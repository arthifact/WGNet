"""Local-window additive Gaussian renderer on a fixed anchor lattice, plus SSIM and field metrics.

Gaussian layout: one slot per 2x2-pixel cell of the 70x70 grid (35x35 = 1225 slots). Each slot owns
6 activated parameters `(dx, dz, log_sx, log_sz, angle, amp)`:
    dx, dz      offset from the slot anchor in pixels, bounded to +-OFFSET_MAX
    log_sx/sz   log standard deviations in pixels, sigma in [SIGMA_MIN, SIGMA_MAX]
    angle       rotation in radians
    amp         residual velocity in km/s added on top of the dense prediction; |amp| is the slot's
                existence (a slot is active when |amp| >= ACTIVE_AMP_KM), so there is no separate gate
                whose gradient would vanish together with the amplitude
A slot only touches the (2 * WINDOW + 1)^2 pixels around its anchor, so rendering costs
O(slots * window) instead of O(slots * pixels), and the result is a plain `index_add_`.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

GRID = 70
SLOT_SIDE = 35
N_SLOTS = SLOT_SIDE * SLOT_SIDE
WINDOW = 12
WIN_SIDE = 2 * WINDOW + 1
OFFSET_MAX = 2.0
SIGMA_MIN = 0.5
SIGMA_MAX = 6.0
AMP_MAX = 1.0
ACTIVE_AMP_KM = 0.05
N_GAUSS_PARAMS = 6
V_MIN_KM, V_MAX_KM = 1.5, 4.5  # OpenFWI normalization range used for SSIM and normalized metrics


def slot_anchors(device: torch.device | str = "cpu") -> torch.Tensor:
    """`[N_SLOTS, 2]` anchor positions `(x, z)` in pixel-edge coordinates (pixel centers at k + 0.5)."""
    i = torch.arange(SLOT_SIDE, device=device, dtype=torch.float32)
    gz, gx = torch.meshgrid(2.0 * i + 1.0, 2.0 * i + 1.0, indexing="ij")
    return torch.stack([gx.reshape(-1), gz.reshape(-1)], dim=-1)


_WINDOW_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def slot_windows(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-slot pixel window: `(flat_index [N_SLOTS, W^2], centers_xy [N_SLOTS, W^2, 2], valid [N_SLOTS, W^2])`.
    Out-of-grid pixels point at the dummy index GRID*GRID and have valid = 0."""
    key = str(device)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    i = torch.arange(SLOT_SIDE, device=device)
    cz, cx = torch.meshgrid(2 * i, 2 * i, indexing="ij")  # pixel index of each slot's upper-left pixel
    off = torch.arange(-WINDOW, WINDOW + 1, device=device)
    oz, ox = torch.meshgrid(off, off, indexing="ij")
    pz = cz.reshape(-1, 1) + oz.reshape(1, -1)  # [N_SLOTS, W^2]
    px = cx.reshape(-1, 1) + ox.reshape(1, -1)
    valid = (pz >= 0) & (pz < GRID) & (px >= 0) & (px < GRID)
    flat = torch.where(valid, pz * GRID + px, torch.full_like(pz, GRID * GRID))
    centers = torch.stack([px.float() + 0.5, pz.float() + 0.5], dim=-1)
    out = (flat, centers, valid.float())
    _WINDOW_CACHE[key] = out
    return out


def activate_gaussians(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained network outputs `[..., 6]` to bounded physical Gaussian parameters."""
    dx = OFFSET_MAX * torch.tanh(raw[..., 0])
    dz = OFFSET_MAX * torch.tanh(raw[..., 1])
    log_range = math.log(SIGMA_MAX) - math.log(SIGMA_MIN)
    log_sx = math.log(SIGMA_MIN) + log_range * torch.sigmoid(raw[..., 2])
    log_sz = math.log(SIGMA_MIN) + log_range * torch.sigmoid(raw[..., 3])
    angle = (math.pi / 2.0) * torch.tanh(raw[..., 4])
    amp = AMP_MAX * torch.tanh(raw[..., 5])
    return torch.stack([dx, dz, log_sx, log_sz, angle, amp], dim=-1)


def clamp_gaussians(params: torch.Tensor) -> torch.Tensor:
    """Project activated parameters back into their bounds (used by test-time refinement)."""
    out = params.clone()
    out[..., 0:2] = out[..., 0:2].clamp(-OFFSET_MAX, OFFSET_MAX)
    out[..., 2:4] = out[..., 2:4].clamp(math.log(SIGMA_MIN), math.log(SIGMA_MAX))
    out[..., 4] = out[..., 4].clamp(-math.pi / 2.0, math.pi / 2.0)
    out[..., 5] = out[..., 5].clamp(-AMP_MAX, AMP_MAX)
    return out


def render_residual(params: torch.Tensor) -> torch.Tensor:
    """Render activated slot parameters `[B, N_SLOTS, 6]` to a residual field `[B, GRID, GRID]` in km/s."""
    bsz = params.shape[0]
    device = params.device
    flat, centers, valid = slot_windows(device)
    anchors = slot_anchors(device)  # [N_SLOTS, 2]

    cx = (anchors[:, 0] + params[:, :, 0]).unsqueeze(-1)  # [B, N_SLOTS, 1]
    cz = (anchors[:, 1] + params[:, :, 1]).unsqueeze(-1)
    inv_sx = torch.exp(-params[:, :, 2]).unsqueeze(-1)
    inv_sz = torch.exp(-params[:, :, 3]).unsqueeze(-1)
    cos_a = torch.cos(params[:, :, 4]).unsqueeze(-1)
    sin_a = torch.sin(params[:, :, 4]).unsqueeze(-1)
    weight = params[:, :, 5].unsqueeze(-1)

    dx = centers[None, :, :, 0] - cx  # [B, N_SLOTS, W^2]
    dz = centers[None, :, :, 1] - cz
    u = (dx * cos_a + dz * sin_a) * inv_sx
    v = (dz * cos_a - dx * sin_a) * inv_sz
    val = weight * torch.exp(-0.5 * (u * u + v * v)) * valid[None]

    field = torch.zeros(bsz, GRID * GRID + 1, device=device, dtype=params.dtype)
    field.index_add_(1, flat.reshape(-1), val.reshape(bsz, -1))
    return field[:, : GRID * GRID].view(bsz, GRID, GRID)


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

def _gaussian_window(size: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(pred_km: torch.Tensor, target_km: torch.Tensor) -> torch.Tensor:
    """Per-sample SSIM `[B]` on OpenFWI-normalized fields `(v - 1.5) / 3.0` in [0, 1], 11x11 Gaussian window."""
    scale = V_MAX_KM - V_MIN_KM
    x = ((pred_km - V_MIN_KM) / scale).unsqueeze(1)
    y = ((target_km - V_MIN_KM) / scale).unsqueeze(1)
    w = _gaussian_window(device=pred_km.device)
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.conv2d(x, w)
    mu_y = F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x**2
    syy = F.conv2d(y * y, w) - mu_y**2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return (num / den).mean(dim=(1, 2, 3))


@torch.no_grad()
def field_metrics(pred_km: torch.Tensor, target_km: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample `[B]` MAE and RMSE in m/s, SSIM, and OpenFWI-normalized MAE / MSE (velocity / 1500 m/s)."""
    err = (pred_km - target_km) * 1000.0
    mae = err.abs().mean(dim=(1, 2))
    mse = err.square().mean(dim=(1, 2))
    return {
        "mae_mps": mae,
        "rmse_mps": mse.sqrt(),
        "ssim": ssim(pred_km, target_km),
        "mae_norm": mae / 1500.0,
        "mse_norm": mse / 1500.0**2,
    }
