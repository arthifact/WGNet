"""Counterfactual wave-attribution probes: which physical attributes of the gathers a trained model uses.
1. Late-Reflection Mute (tests whether deep geology uses weak late reflections).
2. Direct-Wave Mute (tests whether the model can invert purely from scattered reflections).
3. Far-Offset Aperture Mask (tests whether distant receiver interactions are exploited).
4. Phase Scrambling / Hilbert Envelope (tests whether waveform phase coherency is used).
5. Depth-Resolved Input Jacobian Sensitivity (`||d v_hat(z) / d traces(s, t, r)||`).
"""

from __future__ import annotations
from typing import Callable
import numpy as np
import torch

from .data import DT, H, N, NT, SHOTS
from .render import field_metrics


def _direct_wave_mask(
    shots: int = SHOTS,
    nt: int = NT,
    nrec: int = N,
    dt: float = DT,
    v_direct_mps: float = 2200.0,
    margin_sec: float = 0.14,
    taper_sec: float = 0.03,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return smooth `[1, S, T, R]` mask that is `1.0` inside the direct arrival cone
    `t <= |x_r - x_s| / v_direct + margin_sec` and transitions smoothly to `0.0` after.
    """
    xmax = (nrec - 1) * H
    xs = torch.linspace(0.0, xmax, shots, device=device).view(shots, 1, 1)
    xr = (torch.arange(nrec, device=device, dtype=torch.float32) * H).view(1, 1, nrec)
    t_sec = (torch.arange(nt, device=device, dtype=torch.float32) * dt).view(1, nt, 1)

    t_cutoff = torch.abs(xr - xs) / v_direct_mps + margin_sec  # [S, 1, R]
    # Smooth cosine/sigmoid taper
    direct_cone = torch.sigmoid((t_cutoff - t_sec) / max(taper_sec * 0.25, 1e-4))
    return direct_cone.unsqueeze(0)  # [1, S, T, R]


def mute_late_reflections(traces: torch.Tensor, v_direct_mps: float = 2200.0) -> torch.Tensor:
    """Keep only the direct arrival cone; mute weak late reflections."""
    cone = _direct_wave_mask(
        shots=traces.shape[1],
        nt=traces.shape[2],
        nrec=traces.shape[3],
        v_direct_mps=v_direct_mps,
        device=traces.device,
    )
    return traces * cone


def mute_direct_arrivals(traces: torch.Tensor, v_direct_mps: float = 2200.0) -> torch.Tensor:
    """Mute the early direct arrival cone; keep only late scattered reflections."""
    cone = _direct_wave_mask(
        shots=traces.shape[1],
        nt=traces.shape[2],
        nrec=traces.shape[3],
        v_direct_mps=v_direct_mps,
        device=traces.device,
    )
    return traces * (1.0 - cone)


def mask_far_offsets(traces: torch.Tensor, max_offset_m: float = 150.0) -> torch.Tensor:
    """Zero out receivers farther than `max_offset_m` from each shot."""
    shots, _, nrec = traces.shape[1], traces.shape[2], traces.shape[3]
    xmax = (nrec - 1) * H
    xs = torch.linspace(0.0, xmax, shots, device=traces.device).view(1, shots, 1, 1)
    xr = (torch.arange(nrec, device=traces.device, dtype=torch.float32) * H).view(1, 1, 1, nrec)
    near_mask = (torch.abs(xr - xs) <= max_offset_m).to(traces.dtype)
    return traces * near_mask


def scramble_waveform_phase(traces: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Preserve exact per-trace Fourier amplitude spectrum while randomizing phase across time."""
    spec = torch.fft.rfft(traces, dim=2)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rand_angles = (
        torch.rand(spec.shape, generator=gen, dtype=torch.float32).to(traces.device)
        * (2.0 * np.pi)
    )
    # Keep DC (index 0) phase unchanged
    rand_angles[:, :, 0, :] = 0.0
    scrambled_spec = torch.abs(spec) * torch.exp(1j * rand_angles)
    return torch.fft.irfft(scrambled_spec, n=traces.shape[2], dim=2)


class WaveAttributionSuite:
    """Evaluates a wave-to-geology predictor under controlled physical wave perturbations and Jacobian probes."""

    @staticmethod
    @torch.no_grad()
    def evaluate_blindfold_probes(
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        traces: torch.Tensor,
        target_km: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """Run all counterfactual perturbations and measure full, shallow (`z < 35`), and deep (`z >= 35`) RMSE (m/s).

        Args:
            predict_fn: Callable mapping `traces [B, 5, 2001, 70]` -> `pred_km [B, 70, 70]`.
            traces: `[B, 5, 2001, 70]` test/validation wave recordings.
            target_km: `[B, 70, 70]` ground-truth velocity fields in km/s.
        """
        conditions = {
            "clean_full_wave": traces,
            "late_reflections_muted": mute_late_reflections(traces),
            "direct_arrival_muted": mute_direct_arrivals(traces),
            "far_offsets_masked_150m": mask_far_offsets(traces, max_offset_m=150.0),
            "phase_scrambled": scramble_waveform_phase(traces),
        }

        baseline_pred = predict_fn(traces)
        results: dict[str, dict[str, float]] = {}

        for name, mod_traces in conditions.items():
            pred = baseline_pred if name == "clean_full_wave" else predict_fn(mod_traces)
            overall = {k: float(v.mean()) for k, v in field_metrics(pred, target_km).items()}
            shallow_rmse = float(
                torch.sqrt(((pred[:, :35, :] - target_km[:, :35, :]) * 1000.0).square().mean()).item()
            )
            deep_rmse = float(
                torch.sqrt(((pred[:, 35:, :] - target_km[:, 35:, :]) * 1000.0).square().mean()).item()
            )
            shift_from_clean = float(
                torch.sqrt(((pred - baseline_pred) * 1000.0).square().mean()).item()
            )
            results[name] = {
                "mae_mps": overall["mae_mps"],
                "rmse_mps": overall["rmse_mps"],
                "ssim": overall["ssim"],
                "shallow_rmse_mps": shallow_rmse,
                "deep_rmse_mps": deep_rmse,
                "prediction_shift_mps": shift_from_clean,
            }

        return results

    @staticmethod
    def compute_depth_jacobian_attribution(
        differentiable_predict_fn: Callable[[torch.Tensor], torch.Tensor],
        traces_single: torch.Tensor,
    ) -> dict[str, float | np.ndarray]:
        """Compute input Jacobian sensitivity `||d v_hat / d traces||` for shallow (`z in 5..25`) vs deep (`z in 45..65`) geology.

        Proves whether deep velocity predictions actually depend on late reflection arrival times (`t > 0.45s`)
        and far-offset receivers (`|x_r - x_s| > 200m`).
        """
        if traces_single.ndim == 3:
            traces_single = traces_single.unsqueeze(0)

        cone = _direct_wave_mask(device=traces_single.device)
        late_mask = 1.0 - cone  # 1 in late reflection window, 0 in direct wave cone

        xmax = (N - 1) * H
        xs = torch.linspace(0.0, xmax, SHOTS, device=traces_single.device).view(1, SHOTS, 1, 1)
        xr = (torch.arange(N, device=traces_single.device, dtype=torch.float32) * H).view(1, 1, 1, N)
        far_mask = (torch.abs(xr - xs) > 200.0).to(torch.float32)

        stats: dict[str, float | np.ndarray] = {}
        for region_name, z_slice in [("shallow", slice(5, 25)), ("deep", slice(45, 65))]:
            inp = traces_single.detach().clone().requires_grad_(True)
            pred_km = differentiable_predict_fn(inp)
            scalar_target = pred_km[:, z_slice, 15:55].mean()
            (grad_d,) = torch.autograd.grad(scalar_target, inp, retain_graph=False)
            sens_sq = grad_d.detach().square()
            total_energy = float(sens_sq.sum().item()) + 1e-20
            late_fraction = float((sens_sq * late_mask).sum().item()) / total_energy
            far_fraction = float((sens_sq * far_mask).sum().item()) / total_energy

            stats[f"{region_name}_late_reflection_sensitivity_ratio"] = late_fraction
            stats[f"{region_name}_far_offset_sensitivity_ratio"] = far_fraction
            stats[f"{region_name}_sensitivity_map"] = (
                grad_d.detach().abs().mean(dim=0).cpu().numpy()
            )  # [S, T, R]

        return stats
