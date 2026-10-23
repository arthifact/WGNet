"""Budgeted test-time refinement of a WGNet prediction against observed shot gathers.

The refinable scene is `field = dense + upsample(coarse) + render(gaussians)`:
    coarse     a 9x9 additive km/s correction, bilinearly upsampled (smooth background updates)
    gaussians  the active slots' activated parameters (sharp local updates)
with `dense` frozen. Projected L-BFGS with Armijo backtracking, every forward and adjoint solve counted
against a fixed budget, and a frequency-continuation schedule over passbands. True geology never enters.
"""

from __future__ import annotations
import time
from typing import NamedTuple, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from .physics import WavePhysics
from .render import ACTIVE_AMP_KM, GRID, clamp_gaussians, render_residual

COARSE = 9
COARSE_MAX_KM = 1.0
GAUSS_SCALES = (1.0, 1.0, 0.2, 0.2, 0.2, 0.1)
COARSE_SCALE = 0.1
LBFGS_PAIRS = 8
LINE_SEARCH_TRIALS = 6
ARMIJO = 1e-4


class RefineResult(NamedTuple):
    initial_field_km: np.ndarray
    final_field_km: np.ndarray
    initial_loss: float
    final_loss: float
    solves: int
    seconds: float
    history: list[dict]
    gaussians: np.ndarray  # [K, 6] refined active slots
    coarse_km: np.ndarray  # [9, 9]


def _lbfgs_direction(grad: np.ndarray, s_hist: list, y_hist: list, rho: list) -> np.ndarray:
    q = grad.copy()
    alphas = []
    for s, y, r in zip(reversed(s_hist), reversed(y_hist), reversed(rho)):
        a = r * float(np.dot(s, q))
        alphas.append(a)
        q -= a * y
    if s_hist:
        sy, yy = float(np.dot(s_hist[-1], y_hist[-1])), float(np.dot(y_hist[-1], y_hist[-1]))
        q *= sy / max(yy, 1e-30)
    for s, y, r, a in zip(s_hist, y_hist, rho, reversed(alphas)):
        q += s * (a - r * float(np.dot(y, q)))
    return -q


def refine(
    dense_km: np.ndarray,
    gaussians: np.ndarray,
    observed_full: np.ndarray,
    physics: WavePhysics,
    budget_solves: int = 160,
    bands_hz: Sequence[float | None] = (8.0, 12.0, None),
    amp_keep_km: float = ACTIVE_AMP_KM,
    device: torch.device | str = "cpu",
) -> RefineResult:
    """Refine one prediction. `dense_km [70, 70]`, `gaussians [1225, 6]` activated, `observed_full [shots, nt, 70]`."""
    t0 = time.perf_counter()
    active = np.flatnonzero(np.abs(gaussians[:, 5]) >= amp_keep_km)
    n_g = len(active)
    g0 = gaussians[active].astype(np.float64)
    dense_t = torch.from_numpy(dense_km).to(device)

    scales = np.concatenate([np.full(COARSE * COARSE, COARSE_SCALE), np.tile(np.array(GAUSS_SCALES), n_g)])
    x = np.concatenate([np.zeros(COARSE * COARSE), g0.reshape(-1)])

    def project(vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        out[: COARSE * COARSE] = np.clip(out[: COARSE * COARSE], -COARSE_MAX_KM, COARSE_MAX_KM)
        g = torch.from_numpy(out[COARSE * COARSE :].reshape(n_g, 6))
        out[COARSE * COARSE :] = clamp_gaussians(g).numpy().reshape(-1)
        return out

    def assemble(vec: np.ndarray, need_grad: bool):
        coarse = torch.tensor(vec[: COARSE * COARSE].reshape(1, 1, COARSE, COARSE), dtype=torch.float32, device=device, requires_grad=need_grad)
        full = torch.zeros(1, gaussians.shape[0], 6, device=device)
        g = torch.tensor(vec[COARSE * COARSE :].reshape(n_g, 6), dtype=torch.float32, device=device, requires_grad=need_grad)
        if n_g:
            full = full.index_copy(1, torch.as_tensor(active, device=device), g.unsqueeze(0))
        field = dense_t + F.interpolate(coarse, size=(GRID, GRID), mode="bilinear", align_corners=True)[0, 0] + render_residual(full)[0]
        return field, coarse, g

    solves = 0
    history: list[dict] = []

    def evaluate(vec: np.ndarray, need_grad: bool, observed: np.ndarray):
        nonlocal solves
        field, coarse, g = assemble(vec, need_grad)
        field_np = field.detach().cpu().numpy()
        if not need_grad:
            solves += 1
            pred = physics.engine.simulate(np.ascontiguousarray(field_np.reshape(-1) * 1000.0, dtype=np.float32))
            diff = pred.reshape(-1).astype(np.float64) - observed.reshape(-1).astype(np.float64)
            return 0.5 * float(diff @ diff) / float(observed.reshape(-1).astype(np.float64) @ observed.reshape(-1).astype(np.float64)), field_np, None
        solves += 2
        losses, grads = physics.loss_and_grad(field_np[None], observed[None])
        (gc, gg) = torch.autograd.grad((field * torch.from_numpy(grads[0]).to(device)).sum(), [coarse, g])
        grad = np.concatenate([gc.cpu().numpy().reshape(-1), gg.cpu().numpy().reshape(-1)]).astype(np.float64)
        return float(losses[0]), field_np, grad * scales

    x = project(x)
    physics.set_band(None)
    initial_loss, initial_field, _ = evaluate(x, False, observed_full)
    budget_per_band = budget_solves // len(bands_hz)

    for band in bands_hz:
        physics.set_band(band)
        observed = physics.bandlimit(observed_full)
        band_budget = solves + budget_per_band
        cur_loss, _, cur_grad = evaluate(x, True, observed)
        s_hist, y_hist, rho = [], [], []
        while solves + 1 < band_budget:
            d = _lbfgs_direction(cur_grad, s_hist, y_hist, rho)
            slope = float(np.dot(cur_grad, d))
            if not np.isfinite(slope) or slope >= 0.0:
                s_hist, y_hist, rho = [], [], []
                d = -cur_grad
                slope = -float(np.dot(cur_grad, cur_grad))
            norm = float(np.linalg.norm(d))
            if norm > 1.0:
                d, slope = d / norm, slope / norm
            step, accepted = 1.0, False
            for _ in range(LINE_SEARCH_TRIALS):
                if solves + 1 >= band_budget:
                    break
                cand = project(x + step * scales * d)
                cand_loss, _, _ = evaluate(cand, False, observed)
                if cand_loss <= cur_loss + ARMIJO * step * slope:
                    if solves + 2 > band_budget:
                        x, cur_loss = cand, cand_loss
                        accepted = True
                        break
                    cand_loss, _, cand_grad = evaluate(cand, True, observed)
                    s, y = (cand - x) / scales, cand_grad - cur_grad
                    sy = float(np.dot(s, y))
                    if sy > 1e-12:
                        if len(s_hist) >= LBFGS_PAIRS:
                            s_hist.pop(0), y_hist.pop(0), rho.pop(0)
                        s_hist.append(s), y_hist.append(y), rho.append(1.0 / sy)
                    x, cur_loss, cur_grad = cand, cand_loss, cand_grad
                    accepted = True
                    break
                step *= 0.5
            history.append({"band_hz": band, "solves": solves, "loss": cur_loss, "accepted": accepted})
            if not accepted:
                break

    physics.set_band(None)
    final_loss, final_field, _ = evaluate(x, False, observed_full)
    return RefineResult(
        initial_field_km=initial_field,
        final_field_km=final_field,
        initial_loss=initial_loss,
        final_loss=final_loss,
        solves=solves,
        seconds=time.perf_counter() - t0,
        history=history,
        gaussians=x[COARSE * COARSE :].reshape(n_g, 6).astype(np.float32),
        coarse_km=x[: COARSE * COARSE].reshape(COARSE, COARSE).astype(np.float32),
    )
