"""Does the InitialGaussianNetwork's macro model make FWI converge? Run the same budgeted multiscale FWI
(L-BFGS-B on the pixel grid, exact adjoint, passband continuation) from three starting models and compare:

    linear1d  the training set's mean 1D velocity profile (the usual blind start)
    igm       the network's Gaussian macro model predicted from the gathers
    oracle    the true model low-passed at --macro-sigma (the best macro model one could hope for)

    python -m wgnet.fwi_eval --ckpt results/igm_6x6/best.pt --per-family 2 --budget 120

Writes `fwi_eval.json` (per model and per family: initial / final MAE in m/s and full-band misfit) and
`fwi_eval.png` next to the checkpoint. The true model is used only to simulate the observed data and to
score the results; the optimizer never sees it.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from .data import N, OpenFWIData, source_positions_m
from .igm import macro_target
from .physics import WavePhysics
from .train import build_model, get_device, load_checkpoint

V_BOUNDS_KM = (1.4, 4.6)


def fwi(start_km: np.ndarray, observed_full: np.ndarray, physics: WavePhysics, bands: list[float | None], budget: int,
        smooth_grad: float = 1.5) -> tuple[np.ndarray, list[dict]]:
    """Budgeted multiscale FWI. The gradient is Gaussian-smoothed by `smooth_grad` cells (a standard
    preconditioner that keeps updates from piling up at the stations; the same for every start).
    Returns the final model and the per-band history of misfit and solve counts."""
    x = start_km.astype(np.float64).reshape(-1)
    per_band = max(2, budget // len(bands))
    history = []
    for band in bands:
        physics.set_band(band)
        observed = physics.bandlimit(observed_full)
        calls = 0

        def objective(vec: np.ndarray):
            nonlocal calls
            calls += 1
            losses, grads = physics.loss_and_grad(vec.reshape(1, N, N).astype(np.float32), observed[None])
            g = gaussian_filter(grads[0], smooth_grad, mode="nearest") if smooth_grad > 0 else grads[0]
            return float(losses[0]), g.reshape(-1).astype(np.float64)

        res = minimize(objective, x, jac=True, method="L-BFGS-B", bounds=[V_BOUNDS_KM] * (N * N),
                       options={"maxfun": per_band, "maxiter": per_band, "maxcor": 10})
        x = res.x
        history.append({"band_hz": band, "misfit": float(res.fun), "solves": calls})
    physics.set_band(None)
    return x.reshape(N, N).astype(np.float32), history


def misfit_full(model_km: np.ndarray, observed_full: np.ndarray, physics: WavePhysics) -> float:
    physics.set_band(None)
    losses, _ = physics.loss_and_grad(model_km[None], observed_full[None])
    return float(losses[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="InitialGaussianNetwork checkpoint")
    p.add_argument("--per-family", type=int, default=2)
    p.add_argument("--budget", type=int, default=120, help="adjoint evaluations per start, split evenly over the bands")
    p.add_argument("--bands", default="5,8,12,full")
    p.add_argument("--smooth-grad", type=float, default=1.5, help="gradient smoothing in cells (0 = off)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    for k in ("train_per_family", "val_per_family", "test_per_family", "families", "seed_data", "model", "igm_grid", "macro_sigma"):
        setattr(args, k, saved[k])
    args.grad_checkpoint = args.dense_lowres = False
    device = get_device()
    data = OpenFWIData(args.root, args.train_per_family, args.val_per_family, args.test_per_family,
                       families=None if args.families == "all" else tuple(args.families.split(",")), seed=args.seed_data)
    waves = data.ensure_wave_cache((args.split,))[args.split]
    model = build_model(data, args, device)
    load_checkpoint(Path(args.ckpt), model, device)
    model.eval()
    physics = WavePhysics()
    bands = [None if b in ("full", "none") else float(b) for b in args.bands.split(",")]
    linear1d = np.repeat(data.train_mean_km.mean(axis=1, keepdims=True), N, axis=1).astype(np.float32)
    src = torch.from_numpy(source_positions_m(5)).to(device)

    fam_idx = data.family_idx[args.split]
    rows = []
    panels = []
    for fi, fid in enumerate(data.family_ids):
        sel = np.flatnonzero(fam_idx == fi)
        sel = sel[np.linspace(0, len(sel) - 1, min(args.per_family, len(sel))).round().astype(int)]
        for j in sel:
            truth = data.fields_km[args.split][j]
            observed = physics.simulate(truth[None])[0]
            with torch.no_grad():
                w = torch.from_numpy(np.asarray(waves[j : j + 1], dtype=np.float32)).to(device)
                igm = model(w, src).field_km[0].float().cpu().numpy()
            starts = {"linear1d": linear1d, "igm": igm, "oracle": macro_target(torch.from_numpy(truth)[None], args.macro_sigma)[0].numpy()}
            row = {"family": fid, "index": int(j)}
            finals = {}
            for name, start in starts.items():
                final, hist = fwi(start, observed, physics, bands, args.budget, args.smooth_grad)
                finals[name] = final
                row[name] = {
                    "init_mae_mps": float(np.abs(start - truth).mean() * 1000.0),
                    "final_mae_mps": float(np.abs(final - truth).mean() * 1000.0),
                    "init_misfit": misfit_full(start, observed, physics),
                    "final_misfit": misfit_full(final, observed, physics),
                    "solves": int(sum(h["solves"] for h in hist)),
                }
            rows.append(row)
            if len(panels) < 10 and j == sel[0]:
                panels.append((fid, truth, starts, finals))
            print(f"{fid:13s} #{j:5d}  " + "  ".join(f"{k}: {row[k]['init_mae_mps']:5.0f}->{row[k]['final_mae_mps']:5.0f} m/s" for k in starts), flush=True)

    summary = {}
    for fid in data.family_ids:
        fam_rows = [r for r in rows if r["family"] == fid]
        summary[fid] = {k: {m: float(np.mean([r[k][m] for r in fam_rows])) for m in ("init_mae_mps", "final_mae_mps", "init_misfit", "final_misfit")}
                        for k in ("linear1d", "igm", "oracle")}
    out_dir = Path(args.ckpt).parent
    with open(out_dir / "fwi_eval.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows, "summary": summary}, f, indent=2)

    print("\nfamily         start      init MAE  final MAE   init misfit  final misfit")
    for fid, s in summary.items():
        for k in ("linear1d", "igm", "oracle"):
            print(f"{fid:13s}  {k:9s}  {s[k]['init_mae_mps']:8.1f}  {s[k]['final_mae_mps']:9.1f}   {s[k]['init_misfit']:.4e}   {s[k]['final_misfit']:.4e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 7, figsize=(17, 2.5 * len(panels)))
    axes = np.atleast_2d(axes)
    for r, (fid, truth, starts, finals) in enumerate(panels):
        vmin, vmax = float(truth.min()), float(truth.max())
        imgs = [("truth", truth), ("1D start", starts["linear1d"]), ("1D -> FWI", finals["linear1d"]),
                ("IGM start", starts["igm"]), ("IGM -> FWI", finals["igm"]), ("oracle start", starts["oracle"]), ("oracle -> FWI", finals["oracle"])]
        for ax, (title, img) in zip(axes[r], imgs):
            ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
            mae = np.abs(img - truth).mean() * 1000.0
            ax.set_title(f"{fid} {title}" if title == "truth" else f"{title}  {mae:.0f} m/s", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "fwi_eval.png", dpi=90)
    print(f"-> {out_dir / 'fwi_eval.json'}, {out_dir / 'fwi_eval.png'}")


if __name__ == "__main__":
    main()
