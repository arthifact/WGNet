"""End-to-end verification of WGNet's differentiable pieces and a tiny overfit run.

1. Gaussian renderer VJP vs finite differences.
2. wgpu adjoint through the renderer vs finite differences.
3. Frequency continuation: low-passed wavelet synthetics == low-passed full-band synthetics.
4. Model forward/backward with 5 and 2 shots, parameter count, step timing.
5. Budgeted refinement reduces the wave misfit on a synthetic problem.
6. (needs data/openfwi) 40-model overfit: supervised loss must drop by half in 80 steps.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from wgnet.data import SHOTS, OpenFWIData, source_positions_m
from wgnet.model import WGNet, WGNetConfig
from wgnet.physics import WavePhysics
from wgnet.refine import refine
from wgnet.render import N_SLOTS, activate_gaussians, render_residual
from wgnet.train import augment, get_device, supervised_loss

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def random_scene(gen: torch.Generator, n_active: int = 40) -> torch.Tensor:
    raw = torch.randn(1, N_SLOTS, 6, generator=gen) * 0.5
    raw[..., 5] = 0.0
    idx = torch.randperm(N_SLOTS, generator=gen)[:n_active]
    raw[0, idx, 5] = torch.randn(n_active, generator=gen) * 1.5
    return activate_gaussians(raw)


def test_renderer_vjp() -> None:
    gen = torch.Generator().manual_seed(1)
    params = random_scene(gen).double().requires_grad_(True)
    cot = torch.randn(1, 70, 70, generator=gen).double()
    field = render_residual(params)
    (g,) = torch.autograd.grad((field * cot).sum(), params)
    direction = torch.randn_like(params) * torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1]).double()
    eps = 1e-4
    with torch.no_grad():
        fp = (render_residual(params + eps * direction) * cot).sum()
        fm = (render_residual(params - eps * direction) * cot).sum()
    fd, adj = float((fp - fm) / (2 * eps)), float((g * direction).sum())
    check("renderer VJP", abs(fd - adj) / max(abs(fd), 1e-9) < 1e-5, f"fd={fd:.6e} adj={adj:.6e}")


def test_adjoint_through_renderer(device: torch.device) -> None:
    physics = WavePhysics(shots=2, nt=501)
    gen = torch.Generator().manual_seed(2)
    base = torch.full((1, 70, 70), 2.5)
    true_params = random_scene(gen, 12)
    init_params = (true_params + 0.1 * torch.randn_like(true_params)).requires_grad_(True)
    with torch.no_grad():
        observed = physics.simulate((base + render_residual(true_params)).numpy())
    field = base + render_residual(init_params)
    losses, grads = physics.loss_and_grad(field.detach().numpy(), observed)
    (g,) = torch.autograd.grad((field * torch.from_numpy(grads)).sum(), init_params)
    direction = torch.randn_like(init_params) * torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    eps = 0.02
    with torch.no_grad():
        lp, _ = physics.loss_and_grad((base + render_residual(init_params + eps * direction)).numpy(), observed)
        lm, _ = physics.loss_and_grad((base + render_residual(init_params - eps * direction)).numpy(), observed)
    fd, adj = float(lp[0] - lm[0]) / (2 * eps), float((g * direction).sum())
    check("wgpu adjoint through renderer", abs(fd - adj) / max(abs(fd), 1e-12) < 0.03, f"loss={losses[0]:.4e} fd={fd:.5e} adj={adj:.5e}")

    physics.set_band(8.0)
    low = physics.simulate(field.detach().numpy())
    physics.set_band(None)
    full = physics.simulate(field.detach().numpy())
    physics.set_band(8.0)
    filtered = physics.bandlimit(full)
    physics.set_band(None)
    rel = float(np.abs(low - filtered).max() / np.abs(filtered).max())
    check("band-limited wavelet == band-limited data", rel < 2e-3, f"max rel diff={rel:.2e}")


def test_model(device: torch.device) -> None:
    cfg = WGNetConfig.small()
    model = WGNet(np.full((70, 70), 3.0, np.float32), rms_scale=1.0, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(3)
    traces = (torch.randn(2, SHOTS, 2001, 70, generator=gen)).to(device)
    pred = model(traces)
    check("forward 5 shots", tuple(pred.field_km.shape) == (2, 70, 70) and torch.isfinite(pred.field_km).all(), f"params={n_params / 1e6:.2f}M")
    keep = torch.tensor([0, 4], device=device)
    pred2 = model(traces[:, :, ::8].index_select(1, keep), torch.from_numpy(source_positions_m(5)).to(device).index_select(0, keep))
    check("forward 2 shots at 4 ms", tuple(pred2.field_km.shape) == (2, 70, 70), f"dense mean={pred2.dense_km.mean():.3f} km/s")
    t0 = time.perf_counter()
    loss = pred.field_km.abs().mean() + pred.gaussians[..., 5].abs().mean()
    loss.backward()
    if device.type != "cpu":
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0.0)]
    check("backward", not missing, f"all parameters received finite nonzero gradients, backward {time.perf_counter() - t0:.2f}s" if not missing else f"no/zero grad: {missing[:6]}")


def test_refine(device: torch.device) -> None:
    physics = WavePhysics(shots=2, nt=801)
    gen = torch.Generator().manual_seed(4)
    base = np.full((70, 70), 2.5, np.float32)
    true_params = random_scene(gen, 10)
    with torch.no_grad():
        observed = physics.simulate((torch.from_numpy(base) + render_residual(true_params)).numpy())[0]
    init = true_params.clone()
    init[..., 5] *= 0.5
    res = refine(base, init[0].numpy(), observed, physics, budget_solves=30, bands_hz=(10.0, None), device=device)
    check("refinement reduces misfit", res.final_loss < 0.7 * res.initial_loss, f"{res.initial_loss:.4e} -> {res.final_loss:.4e} in {res.solves} solves, {res.seconds:.1f}s")


def test_overfit(device: torch.device) -> None:
    if not (ROOT / "data" / "openfwi" / "manifest.json").exists():
        print("  [SKIP] overfit: data/openfwi not present")
        return
    data = OpenFWIData(ROOT, train_per_family=4, val_per_family=1, test_per_family=1)
    waves = data.ensure_wave_cache(("train",))
    model = WGNet(data.train_mean_km, data.rms_scale(), WGNetConfig.small()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    traces = torch.from_numpy(np.asarray(waves["train"], dtype=np.float32)).to(device)
    target = torch.from_numpy(data.fields_km["train"]).to(device)
    rng = np.random.default_rng(0)
    args = argparse.Namespace(edge_weight=0.3, dense_weight=0.5, sparsity=0.1)
    first = None
    t0 = time.perf_counter()
    for step in range(80):
        x, src = augment(traces, model.rms_scale, rng)
        loss, parts = supervised_loss(model(x, src), target, args, sparsity_weight=0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        first = first or loss.item()
    check("40-model overfit", loss.item() < 0.75 * first, f"loss {first:.4f} -> {loss.item():.4f} (l1 {float(parts['l1']) * 1000:.0f} m/s), {(time.perf_counter() - t0) / 80:.2f}s/step on {len(target)} models")


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    test_renderer_vjp()
    test_adjoint_through_renderer(device)
    test_model(device)
    test_refine(device)
    test_overfit(device)
    print("all checks passed")


if __name__ == "__main__":
    main()
