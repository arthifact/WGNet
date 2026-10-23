"""Exact wave-equation data misfit as a PyTorch loss, with frequency continuation.

`WavePhysics` owns a native wgpu engine and exposes `loss_and_grad`, the normalized data misfit and its
exact gradient with respect to the velocity field from the engine's discrete adjoint. The passband is controlled with `set_band(fmax_hz)`: the source
wavelet is low-passed with a causal Gaussian kernel, and the same causal kernel applied to observed data
(`bandlimit`) gives exactly the synthetics of the filtered wavelet, by linearity and time invariance. Observed data simulated through the
engine after `set_band` is already band-limited and needs no filtering.
"""

from __future__ import annotations
import math
import numpy as np
import torch

from .data import DT, N, NT, SHOTS, make_engine


def lowpass_kernel(fmax_hz: float, dt: float = DT) -> np.ndarray:
    """Gaussian low-pass with its -3 dB point at `fmax_hz`, truncated at 4 sigma, unit DC gain.
    Applied causally (delay = len - 1 samples) so that filtering the wavelet and filtering the data commute
    exactly with the wave simulation, including at the start and end of the recording."""
    sigma_s = math.sqrt(math.log(2.0) / 2.0) / (math.pi * fmax_hz)
    half = int(math.ceil(4.0 * sigma_s / dt))
    n = np.arange(-half, half + 1)
    h = np.exp(-0.5 * (n * dt / sigma_s) ** 2)
    return (h / h.sum()).astype(np.float32)


class WavePhysics:
    def __init__(self, shots: int = SHOTS, nt: int = NT) -> None:
        self.engine = make_engine(shots, nt)
        self.shots, self.nt = shots, nt
        self.full_wavelet = self.engine.wavelet().copy()
        self.kernel: np.ndarray | None = None
        self.fmax: float | None = None

    def set_band(self, fmax_hz: float | None) -> None:
        """`None` restores the full-band 15 Hz Ricker."""
        if fmax_hz is None:
            self.kernel, self.fmax = None, None
            self.engine.set_wavelet(self.full_wavelet)
            return
        self.kernel = lowpass_kernel(fmax_hz)
        self.fmax = float(fmax_hz)
        w = np.convolve(self.full_wavelet, self.kernel, mode="full")[: self.nt].astype(np.float32)
        self.engine.set_wavelet(np.ascontiguousarray(w))

    def bandlimit(self, traces: np.ndarray) -> np.ndarray:
        """Apply the current passband kernel along time to `[..., nt, nrec]` full-band observed data."""
        if self.kernel is None:
            return traces
        t = torch.from_numpy(np.ascontiguousarray(traces, dtype=np.float32))
        lead = t.shape[:-2]
        x = t.reshape(-1, 1, t.shape[-2], t.shape[-1]).transpose(2, 3).reshape(-1, 1, t.shape[-2])
        k = torch.from_numpy(self.kernel).view(1, 1, -1)
        y = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (len(self.kernel) - 1, 0)), k)
        return y.view(*lead, t.shape[-1], t.shape[-2]).transpose(-1, -2).contiguous().numpy()

    def simulate(self, fields_km: np.ndarray) -> np.ndarray:
        """`[B, 70, 70]` km/s -> `[B, shots, nt, 70]` synthetics with the current passband."""
        out = np.empty((len(fields_km), self.shots, self.nt, N), np.float32)
        for b in range(len(fields_km)):
            out[b] = self.engine.simulate(np.ascontiguousarray(fields_km[b].reshape(-1) * 1000.0, dtype=np.float32))
        return out

    def loss_and_grad(self, fields_km: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-model normalized misfit `0.5 ||p - d||^2 / ||d||^2` and its gradient w.r.t. velocity in km/s."""
        bsz = len(fields_km)
        losses = np.zeros(bsz, np.float32)
        grads = np.zeros((bsz, N, N), np.float32)
        for b in range(bsz):
            vel = np.ascontiguousarray(fields_km[b].reshape(-1) * 1000.0, dtype=np.float32)
            obs = np.ascontiguousarray(observed[b].reshape(-1), dtype=np.float32)
            losses[b], g = self.engine.evaluate_adjoint(vel, obs)
            grads[b] = g * 1000.0  # d/d(km/s) = 1000 * d/d(m/s)
        return losses, grads
