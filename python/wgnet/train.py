"""Train and evaluate WGNet.

Stage `sup`  : supervised training on simulated gathers (L1 + edge L1 on the composed field, L1 on the
               dense head alone, L1 sparsity on Gaussian amplitudes).
Stage `phys` : fine-tuning from a `sup` checkpoint with the exact wave-equation adjoint added, gradient
               balanced against the supervised gradient and run through a low-to-full passband schedule.

    python -m wgnet.train --stage sup --out results/base
    python -m wgnet.train --stage phys --init results/base/best.pt --out results/base_phys
    python -m wgnet.train --eval-only results/base_phys/best.pt --eval-test
"""

from __future__ import annotations
import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .data import SHOTS, OpenFWIData, source_positions_m
from .igm import InitialGaussianNetwork, macro_target
from .model import WGNet, WGNetConfig, remap_legacy_state_dict
from .physics import WavePhysics
from .render import ACTIVE_AMP_KM, field_metrics

SHOT_SUBSETS = {5: (0, 1, 2, 3, 4), 3: (0, 2, 4), 2: (0, 4), 1: (2,)}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Batcher:
    """Random mini-batches (with replacement) from the float16 wave memmap and the fields, prefetched in a thread."""

    def __init__(self, waves: np.memmap, fields_km: np.ndarray, batch: int, device: torch.device, seed: int, depth: int = 6) -> None:
        self.waves, self.fields, self.batch, self.device = waves, fields_km, batch, device
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            idx = np.sort(self.rng.integers(0, len(self.waves), size=self.batch))
            w = torch.from_numpy(np.asarray(self.waves[idx], dtype=np.float32))
            v = torch.from_numpy(self.fields[idx])
            self.q.put((idx, w, v))

    def next(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        idx, w, v = self.q.get()
        return w.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True), idx


def augment(traces: torch.Tensor, rms_scale: float, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Shot-subset, additive noise, and dead-trace augmentation. Returns `(traces, source_x_m)`."""
    shots = traces.shape[1]
    src = torch.from_numpy(source_positions_m(shots)).to(traces.device)
    if shots > 1 and rng.random() < 0.25:
        keep = np.sort(rng.choice(shots, size=int(rng.integers(1, shots)), replace=False))
        keep_t = torch.from_numpy(keep).to(traces.device)
        traces, src = traces.index_select(1, keep_t), src.index_select(0, keep_t)
    if rng.random() < 0.5:
        traces = traces + torch.randn_like(traces) * (float(rng.uniform(0.0, 0.02)) * rms_scale)
    if rng.random() < 0.3:
        kill = torch.rand(traces.shape[0], traces.shape[1], 1, traces.shape[3], device=traces.device) < float(rng.uniform(0.0, 0.1))
        traces = traces.masked_fill(kill, 0.0)
    return traces, src


AMP_FREE_KM = 0.02  # |amp| below this is not penalized, so pruned slots keep a live gradient


def supervised_loss(pred, target: torch.Tensor, args, sparsity_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    err = pred.field_km - target
    l1 = err.abs().mean()
    edge = 0.5 * ((err[:, :, 1:] - err[:, :, :-1]).abs().mean() + (err[:, 1:, :] - err[:, :-1, :]).abs().mean())
    dense = (pred.dense_km - target).abs().mean()
    amp = pred.gaussians[..., 5].abs()
    sparsity = F.relu(amp - AMP_FREE_KM).mean()
    total = l1 + args.edge_weight * edge + args.dense_weight * dense + sparsity_weight * sparsity
    active = (amp.detach() >= ACTIVE_AMP_KM).float().sum(dim=1).mean()
    return total, {"l1": l1.detach(), "edge": edge.detach(), "dense_l1": dense.detach(), "active": active}


def sparsity_weight_at(step: int, args) -> float:
    """Ramp the existence penalty in linearly so Gaussians can learn useful amplitudes before being pruned."""
    return args.sparsity * min(1.0, step / max(1, args.sparsity_warmup))


@torch.no_grad()
def evaluate(model: torch.nn.Module, waves: np.memmap, fields_km: np.ndarray, family_idx: np.ndarray, family_ids: list[str],
             device: torch.device, batch: int, shots: int = 5, amp: bool = False, macro_sigma: float = 0.0) -> dict:
    """Metrics against the raw truth, or, with `macro_sigma > 0`, against the low-passed macro model
    (then `mae_raw_mps` additionally reports the error against the raw truth)."""
    model.eval()
    keep = torch.tensor(SHOT_SUBSETS[shots], device=device)
    src = torch.from_numpy(source_positions_m(SHOTS)).to(device).index_select(0, keep)
    acc: dict[str, list[torch.Tensor]] = {}
    n_active = []
    for start in range(0, len(fields_km), batch):
        w = torch.from_numpy(np.asarray(waves[start : start + batch], dtype=np.float32)).to(device).index_select(1, keep)
        v = torch.from_numpy(fields_km[start : start + batch]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(w, src)
        v_target = macro_target(v, macro_sigma) if macro_sigma > 0 else v
        m = field_metrics(pred.field_km.float(), v_target)
        m["dense_mae_mps"] = (pred.dense_km.float() - v_target).abs().mean(dim=(1, 2)) * 1000.0
        if macro_sigma > 0:
            m["mae_raw_mps"] = (pred.field_km.float() - v).abs().mean(dim=(1, 2)) * 1000.0
        for k, val in m.items():
            acc.setdefault(k, []).append(val.cpu())
        n_active.append((pred.gaussians[..., 5].abs() >= ACTIVE_AMP_KM).sum(dim=1).float().cpu())
    per_sample = {k: torch.cat(v) for k, v in acc.items()}
    n_active_t = torch.cat(n_active)
    out = {k: float(v.mean()) for k, v in per_sample.items()}
    out["active_gaussians"] = float(n_active_t.mean())
    out["n"] = int(len(fields_km))
    out["families"] = {}
    fam_t = torch.from_numpy(family_idx)
    for fi, fid in enumerate(family_ids):
        sel = fam_t == fi
        if sel.any():
            out["families"][fid] = {k: float(v[sel].mean()) for k, v in per_sample.items()}
    model.train()
    return out


def format_eval(tag: str, res: dict) -> str:
    raw = f" raw_MAE={res['mae_raw_mps']:.1f}" if "mae_raw_mps" in res else ""
    lines = [f"[{tag}] n={res['n']} MAE={res['mae_mps']:.1f} RMSE={res['rmse_mps']:.1f} SSIM={res['ssim']:.4f} "
             f"(norm MAE={res['mae_norm']:.4f} MSE={res['mse_norm']:.5f}) dense_MAE={res['dense_mae_mps']:.1f} gaussians={res['active_gaussians']:.0f}{raw}"]
    for fid, m in res["families"].items():
        lines.append(f"    {fid:14s} MAE={m['mae_mps']:7.1f} RMSE={m['rmse_mps']:7.1f} SSIM={m['ssim']:.4f}")
    return "\n".join(lines)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1.0 + step) / (10.0 + step))  # warm-up so early evaluations do not average in the init
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.float(), alpha=1.0 - decay)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()})


def build_model(data: OpenFWIData, args, device: torch.device) -> torch.nn.Module:
    cfg = WGNetConfig() if args.model == "base" else WGNetConfig.small()
    cfg.grad_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    cfg.dense_lowres = bool(getattr(args, "dense_lowres", False))
    if args.model == "igm":
        nz, nx = (int(v) for v in args.igm_grid.lower().split("x"))
        return InitialGaussianNetwork(data.train_mean_km, data.rms_scale(), cfg, grid=(nz, nx)).to(device)
    return WGNet(data.train_mean_km, data.rms_scale(), cfg).to(device)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device, use_ema: bool = True) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(remap_legacy_state_dict(ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]))
    return ckpt


def macro_loss(pred, target_macro: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    err = pred.field_km - target_macro
    l1 = err.abs().mean()
    edge = 0.5 * ((err[:, :, 1:] - err[:, :, :-1]).abs().mean() + (err[:, 1:, :] - err[:, :-1, :]).abs().mean())
    return l1 + args.edge_weight * edge, {"l1": l1.detach(), "edge": edge.detach()}


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t))))


def parse_bands(spec: str) -> list[float | None]:
    return [None if b.strip() in ("full", "none") else float(b) for b in spec.split(",")]


def train(args) -> None:
    device = get_device()
    amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    data = OpenFWIData(args.root, args.train_per_family, args.val_per_family, args.test_per_family,
                       families=None if args.families == "all" else tuple(args.families.split(",")), seed=args.seed_data)
    waves = data.ensure_wave_cache(("train", "val") if not args.eval_test else ("train", "val", "test"))
    model = build_model(data, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.1f}M train={len(data.fields_km['train'])} val={len(data.fields_km['val'])}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    ema = EMA(model, args.ema)
    step, best = 0, float("inf")
    if args.init:
        load_checkpoint(Path(args.init), model, device, use_ema=True)
        ema = EMA(model, args.ema)
        print(f"initialized from {args.init}", flush=True)
    last = out_dir / "last.pt"
    if args.resume and last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        opt.load_state_dict(ckpt["optimizer"])
        step, best = int(ckpt["step"]), float(ckpt["best"])
        print(f"resumed {last} at step {step} (best val MAE {best:.2f})", flush=True)

    macro_sigma = args.macro_sigma if args.model == "igm" else 0.0
    physics = WavePhysics() if args.stage == "phys" else None
    bands = parse_bands(args.bands) if physics else []
    rng = np.random.default_rng(args.seed + step)
    batcher = Batcher(waves["train"], data.fields_km["train"], args.batch, device, args.seed + step)
    log = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    eval_model = build_model(data, args, device)
    val_subset = (waves["val"], data.fields_km["val"], data.family_idx["val"])
    if args.val_limit and args.val_limit < len(data.fields_km["val"]):
        sel = np.linspace(0, len(data.fields_km["val"]) - 1, args.val_limit).round().astype(np.int64)
        val_subset = (np.asarray(waves["val"][sel]), data.fields_km["val"][sel], data.family_idx["val"][sel])
    t0, t_last = time.perf_counter(), time.perf_counter()
    model.train()

    while step < args.steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        traces, target, idx = batcher.next()
        traces, src = augment(traces, model.rms_scale, rng)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(traces, src)
        pred = pred._replace(field_km=pred.field_km.float(), dense_km=pred.dense_km.float(), gaussians=pred.gaussians.float())
        if args.model == "igm":
            loss, parts = macro_loss(pred, macro_target(target, args.macro_sigma), args)
        else:
            loss, parts = supervised_loss(pred, target, args, sparsity_weight_at(step, args))

        if physics is not None:
            band = bands[min(len(bands) - 1, int(len(bands) * (step - 1) / max(1, args.steps)))]
            if physics.fmax != band:
                physics.set_band(band)
            nb = min(args.phys_batch, len(idx))
            field = pred.field_km[:nb]
            observed = physics.simulate(data.fields_km["train"][idx[:nb]])
            phys_losses, phys_grads = physics.loss_and_grad(field.detach().cpu().numpy(), observed)
            (g_sup,) = torch.autograd.grad(loss, pred.field_km, retain_graph=True)
            g_phys = torch.from_numpy(phys_grads).to(device) / nb
            scale = args.phys_ratio * g_sup[:nb].norm() / (g_phys.norm() + 1e-12)
            loss = loss + scale * (field * g_phys).sum()
            parts["phys_misfit"], parts["phys_scale"], parts["band_hz"] = float(phys_losses.mean()), float(scale), band or 15.0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        ema.update(model, step)

        if step % args.log_every == 0 or step == 1:
            now = time.perf_counter()
            rec = {"step": step, "loss": loss.item(), "grad_norm": float(gnorm), "lr": lr_at(step, args),
                   "sec_per_step": (now - t_last) / args.log_every, **{k: float(v) for k, v in parts.items()}}
            t_last = now
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ema.apply_to(eval_model)
            res = evaluate(eval_model, *val_subset, data.family_ids, device, args.eval_batch, amp=amp, macro_sigma=macro_sigma)
            res["step"] = step
            print(format_eval(f"val step {step}", res), flush=True)
            log.write(json.dumps({"eval": res}) + "\n")
            log.flush()
            is_best = res["mae_mps"] < best
            best = min(best, res["mae_mps"])
            state = {"model": model.state_dict(), "ema": ema.shadow, "optimizer": opt.state_dict(), "step": step,
                     "best": best, "args": vars(args), "cfg": vars(model.cfg), "val": res}
            torch.save(state, last)
            if is_best:
                torch.save(state, out_dir / "best.pt")
                print(f"  new best val MAE {best:.2f} m/s", flush=True)

    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, best val MAE {best:.2f} m/s", flush=True)
    if args.eval_test:
        load_checkpoint(out_dir / "best.pt", eval_model, device)
        run_test(eval_model, data, waves, device, args, out_dir, amp)


def run_test(model: WGNet, data: OpenFWIData, waves: dict, device: torch.device, args, out_dir: Path, amp: bool) -> None:
    report = {}
    for shots in [int(s) for s in args.eval_shots.split(",")]:
        res = evaluate(model, waves["test"], data.fields_km["test"], data.family_idx["test"], data.family_ids, device, args.eval_batch, shots=shots, amp=amp,
                       macro_sigma=args.macro_sigma if args.model == "igm" else 0.0)
        print(format_eval(f"test {shots}-shot", res), flush=True)
        report[f"{shots}_shot"] = res
    with open(out_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--families", default="all", help="comma-separated OpenFWI family ids or 'all'")
    p.add_argument("--train-per-family", type=int, default=23000)
    p.add_argument("--val-per-family", type=int, default=1000)
    p.add_argument("--test-per-family", type=int, default=6000)
    p.add_argument("--model", choices=["base", "small", "igm"], default="base",
                   help="base/small: WGNet; igm: InitialGaussianNetwork (K Gaussians -> smooth macro model, small encoder)")
    p.add_argument("--igm-grid", default="6x6", help="igm: anchor grid nz x nx (one Gaussian per anchor)")
    p.add_argument("--macro-sigma", type=float, default=6.0, help="igm: Gaussian low-pass of the truth in cells that defines the macro-model target")
    p.add_argument("--grad-checkpoint", action="store_true", help="recompute transformer blocks in backward to save memory")
    p.add_argument("--dense-lowres", action="store_true", help="dense head at 35x35 + bilinear upsample; sharp structure left to the Gaussians")
    p.add_argument("--stage", choices=["sup", "phys"], default="sup")
    p.add_argument("--out", default="results/base")
    p.add_argument("--init", default="", help="checkpoint to initialize from (phys stage)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--edge-weight", type=float, default=0.3)
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--sparsity", type=float, default=0.1)
    p.add_argument("--sparsity-warmup", type=int, default=3000)
    p.add_argument("--phys-ratio", type=float, default=0.5, help="physics gradient norm as a fraction of the supervised one")
    p.add_argument("--phys-batch", type=int, default=8)
    p.add_argument("--bands", default="6,9,12,full", help="passband schedule (Hz) across the phys stage")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-shots", default="5,3,1")
    p.add_argument("--val-limit", type=int, default=0, help="evaluate on an evenly strided subset of the validation split during training (0 = all)")
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--eval-only", default="", help="checkpoint: evaluate on val (and test with --eval-test) and exit")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7201)
    p.add_argument("--seed-data", type=int, default=20261020)
    args = p.parse_args()

    if args.eval_only:
        device = get_device()
        data = OpenFWIData(args.root, args.train_per_family, args.val_per_family, args.test_per_family,
                           families=None if args.families == "all" else tuple(args.families.split(",")), seed=args.seed_data)
        waves = data.ensure_wave_cache(("train", "val", "test") if args.eval_test else ("train", "val"))
        saved = torch.load(Path(args.eval_only), map_location="cpu", weights_only=False)["args"]
        for k in ("model", "igm_grid", "macro_sigma", "dense_lowres"):
            if k in saved:
                setattr(args, k, saved[k])
        model = build_model(data, args, device)
        load_checkpoint(Path(args.eval_only), model, device)
        res = evaluate(model, waves["val"], data.fields_km["val"], data.family_idx["val"], data.family_ids, device, args.eval_batch, amp=device.type == "cuda",
                       macro_sigma=args.macro_sigma if args.model == "igm" else 0.0)
        print(format_eval("val", res), flush=True)
        if args.eval_test:
            run_test(model, data, waves, device, args, Path(args.eval_only).parent, device.type == "cuda")
        return
    train(args)


if __name__ == "__main__":
    main()
