"""Side-by-side sections of ground truth, prediction, Gaussian residual and error for validation models.

    python -m wgnet.visualize --ckpt results/debug_small/best.pt --per-family 10 --out results/debug_small/viz

Writes one PNG per family (rows = models; columns = truth, prediction, dense head, Gaussian residual, error)
and `overview.png` with the first two models of every family.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import OpenFWIData, source_positions_m
from .render import ACTIVE_AMP_KM, field_metrics
from .train import build_model, get_device, load_checkpoint


@torch.no_grad()
def predict(model, waves, fields, device, batch: int = 16):
    src = torch.from_numpy(source_positions_m(5)).to(device)
    outs = []
    for s in range(0, len(fields), batch):
        w = torch.from_numpy(np.asarray(waves[s : s + batch], dtype=np.float32)).to(device)
        p = model(w, src)
        outs.append({k: getattr(p, k).float().cpu() for k in ("field_km", "dense_km", "residual_km", "gaussians")})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def draw_rows(axes_rows, truth, pred, dense, resid, gauss, metrics, titles_prefix=""):
    for r, ax_row in enumerate(axes_rows):
        vmin, vmax = float(truth[r].min()), float(truth[r].max())
        n_act = int((gauss[r, :, 5].abs() >= ACTIVE_AMP_KM).sum())
        panels = [
            (truth[r], "viridis", vmin, vmax, f"{titles_prefix}truth"),
            (pred[r], "viridis", vmin, vmax, f"pred  MAE={metrics['mae_mps'][r]:.0f} RMSE={metrics['rmse_mps'][r]:.0f} m/s"),
            (dense[r], "viridis", vmin, vmax, "dense head"),
            (resid[r], "coolwarm", -0.5, 0.5, f"Gaussian residual ({n_act} active)"),
            ((pred[r] - truth[r]), "coolwarm", -0.5, 0.5, "error (km/s)"),
        ]
        for ax, (img, cmap, lo, hi, title) in zip(ax_row, panels):
            ax.imshow(img.numpy() if hasattr(img, "numpy") else img, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--per-family", type=int, default=10)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--ema", action="store_true", help="use the EMA weights instead of the live weights")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--model", choices=["base", "small", "igm"], default="")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    for k in ("train_per_family", "val_per_family", "test_per_family", "families", "seed_data"):
        setattr(args, k, saved[k])
    args.model = args.model or saved["model"]
    args.grad_checkpoint = False
    args.dense_lowres = bool(saved.get("dense_lowres", False))
    args.igm_grid = saved.get("igm_grid", "6x6")
    args.macro_sigma = float(saved.get("macro_sigma", 6.0))
    device = get_device()
    data = OpenFWIData(args.root, args.train_per_family, args.val_per_family, args.test_per_family,
                       families=None if args.families == "all" else tuple(args.families.split(",")), seed=args.seed_data)
    waves = data.ensure_wave_cache((args.split,))[args.split]
    model = build_model(data, args, device)
    load_checkpoint(Path(args.ckpt), model, device, use_ema=args.ema)
    model.eval()
    out_dir = Path(args.out or (Path(args.ckpt).parent / "viz"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fam_idx = data.family_idx[args.split]
    overview = []
    for fi, fid in enumerate(data.family_ids):
        sel = np.flatnonzero(fam_idx == fi)
        sel = sel[np.linspace(0, len(sel) - 1, min(args.per_family, len(sel))).round().astype(int)]
        pred = predict(model, waves[sel], data.fields_km[args.split][sel], device)
        truth = torch.from_numpy(data.fields_km[args.split][sel])
        m = field_metrics(pred["field_km"], truth)
        n = len(sel)
        fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
        axes = np.atleast_2d(axes)
        draw_rows(axes, truth, pred["field_km"], pred["dense_km"], pred["residual_km"], pred["gaussians"], m)
        fig.suptitle(f"{fid}  ({args.split}, step {ckpt['step']}, {'EMA' if args.ema else 'live'} weights)  "
                     f"MAE={float(m['mae_mps'].mean()):.0f}  RMSE={float(m['rmse_mps'].mean()):.0f} m/s  SSIM={float(m['ssim'].mean()):.3f}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_dir / f"{fid}.png", dpi=90)
        plt.close(fig)
        overview.append((fid, truth[:2], pred["field_km"][:2], pred["dense_km"][:2], pred["residual_km"][:2], pred["gaussians"][:2], {k: v[:2] for k, v in m.items()}))
        print(f"{fid:14s} MAE={float(m['mae_mps'].mean()):7.1f} RMSE={float(m['rmse_mps'].mean()):7.1f} SSIM={float(m['ssim'].mean()):.3f} -> {out_dir / f'{fid}.png'}", flush=True)

    fig, axes = plt.subplots(2 * len(overview), 5, figsize=(13, 2.6 * 2 * len(overview)))
    for i, (fid, truth, pred, dense, resid, gauss, m) in enumerate(overview):
        draw_rows(axes[2 * i : 2 * i + 2], truth, pred, dense, resid, gauss, m, titles_prefix=f"{fid} ")
    fig.suptitle(f"overview ({args.split}, step {ckpt['step']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_dir / "overview.png", dpi=80)
    print(f"overview -> {out_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
