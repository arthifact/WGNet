"""InitialGaussianNetwork: shot gathers -> K anisotropic Gaussians -> smooth macro velocity model.

The macro model is a partition-of-unity radial-basis field
    v(x) = sum_k w_k(x) v_k,    w_k(x) = softmax_k( -1/2 (x - c_k)^T Sigma_k^{-1} (x - c_k) )
so it is smooth by construction, bounded by the node velocities, and fully described by K * 6 numbers
`(x, z, log_sx, log_sz, angle, v)`. Widths are kept at or above SIGMA_MIN cells, so the network cannot
produce anything sharper than a macro model even if asked to.

Each Gaussian is a query anchored on a coarse grid; it cross-attends to the gather tokens of the shared
`GatherEncoder` with the traveltime bias evaluated at its anchor, and predicts a bounded offset from the
anchor, its widths, orientation and velocity. The training target is the true model low-passed with a
Gaussian filter (`macro_target`), the usual definition of a background model for FWI.
"""

from __future__ import annotations
import math
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DecoderBlock, GatherEncoder, Prediction, WGNetConfig, fourier_features
from .render import GRID

SIGMA_MIN = 3.0    # cells
SIGMA_MAX = 40.0   # cells
H_M = 10.0


class MacroScene(NamedTuple):
    params: torch.Tensor  # [B, K, 6]: x, z (pixels), log_sx, log_sz (log pixels), angle (rad), v (km/s)


def render_macro(params: torch.Tensor) -> torch.Tensor:
    """`[B, K, 6]` -> `[B, GRID, GRID]` partition-of-unity Gaussian field in km/s."""
    device = params.device
    i = torch.arange(GRID, device=device, dtype=params.dtype) + 0.5
    pz, px = torch.meshgrid(i, i, indexing="ij")
    px, pz = px.reshape(1, -1, 1), pz.reshape(1, -1, 1)          # [1, P, 1]
    cx, cz = params[:, None, :, 0], params[:, None, :, 1]        # [B, 1, K]
    inv_sx, inv_sz = torch.exp(-params[:, None, :, 2]), torch.exp(-params[:, None, :, 3])
    cos_a, sin_a = torch.cos(params[:, None, :, 4]), torch.sin(params[:, None, :, 4])
    dx, dz = px - cx, pz - cz
    u = (dx * cos_a + dz * sin_a) * inv_sx
    w = (dz * cos_a - dx * sin_a) * inv_sz
    weights = torch.softmax(-0.5 * (u * u + w * w), dim=-1)      # [B, P, K]
    field = (weights * params[:, None, :, 5]).sum(-1)
    return field.view(params.shape[0], GRID, GRID)


def macro_target(v_km: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Gaussian low-pass of `[B, GRID, GRID]` with reflect padding: the macro-model target."""
    if sigma_cells <= 0:
        return v_km
    half = int(math.ceil(3.0 * sigma_cells))
    t = torch.arange(-half, half + 1, device=v_km.device, dtype=v_km.dtype)
    k = torch.exp(-0.5 * (t / sigma_cells) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = v_km.unsqueeze(1)
    x = F.conv2d(F.pad(x, (half, half, 0, 0), mode="reflect"), k)
    x = F.conv2d(F.pad(x, (0, 0, half, half), mode="reflect"), k.transpose(2, 3))
    return x.squeeze(1)


class InitialGaussianNetwork(nn.Module):
    def __init__(self, mean_km: np.ndarray | torch.Tensor, rms_scale: float, cfg: WGNetConfig | None = None,
                 grid: tuple[int, int] = (6, 6)) -> None:
        super().__init__()
        self.cfg = cfg = cfg or WGNetConfig.small()
        d = cfg.d_model
        self.encoder = GatherEncoder(rms_scale, cfg)
        nz, nx = grid
        self.k = nz * nx
        zi = (torch.arange(nz, dtype=torch.float32) + 0.5) * (GRID / nz)
        xi = (torch.arange(nx, dtype=torch.float32) + 0.5) * (GRID / nx)
        az, ax = torch.meshgrid(zi, xi, indexing="ij")
        anchors = torch.stack([ax.reshape(-1), az.reshape(-1)], dim=-1)  # [K, 2] pixels
        self.register_buffer("anchors", anchors)
        self.register_buffer("offset_max", torch.tensor([0.5 * GRID / nx, 0.5 * GRID / nz]))
        mean_t = torch.as_tensor(np.asarray(mean_km), dtype=torch.float32).view(1, 1, GRID, GRID)
        grid_pts = (anchors / GRID * 2.0 - 1.0).view(1, 1, -1, 2)
        mean_at = F.grid_sample(mean_t, grid_pts, mode="bilinear", align_corners=False).view(-1)
        self.register_buffer("base_atanh", torch.atanh(torch.clamp((mean_at - 3.0) * 0.5, -0.999, 0.999)))

        self.query_embed = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.query_pos = nn.Linear(2 * 12, d)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.decoder_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 6)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            # start at sigma ~ one anchor spacing so the initial field is the smooth mean model
            s0 = math.log(GRID / max(nz, nx))
            self.head.bias[2] = self.head.bias[3] = _inv_width(s0)

    @property
    def rms_scale(self) -> float:
        return self.encoder.rms_scale

    def scene(self, raw: torch.Tensor) -> torch.Tensor:
        """Unconstrained `[B, K, 6]` -> physical parameters."""
        xy = self.anchors + self.offset_max * torch.tanh(raw[..., 0:2])
        log_range = math.log(SIGMA_MAX) - math.log(SIGMA_MIN)
        log_s = math.log(SIGMA_MIN) + log_range * torch.sigmoid(raw[..., 2:4])
        angle = (math.pi / 2.0) * torch.tanh(raw[..., 4:5])
        v = 3.0 + 2.0 * torch.tanh(raw[..., 5:6] + self.base_atanh.view(1, -1, 1))
        return torch.cat([xy, log_s, angle, v], dim=-1)

    def forward(self, traces: torch.Tensor, source_x_m: torch.Tensor | None = None) -> Prediction:
        if traces.ndim == 3:
            traces = traces.unsqueeze(0)
        b = traces.shape[0]
        source_x_m = self.encoder.default_sources(traces, source_x_m)
        ctx = self.encoder(traces, source_x_m)
        bias = self.encoder.bias(source_x_m, self.anchors * H_M).to(ctx.dtype)
        q = self.query_embed + self.query_pos(fourier_features(self.anchors / GRID * 2.0 - 1.0))
        q = q.unsqueeze(0).expand(b, -1, -1)
        for blk in self.decoder_blocks:
            q = blk(q, ctx, bias)
        params = self.scene(self.head(self.norm(q)))
        field = render_macro(params)
        return Prediction(field, field, torch.zeros_like(field), params, None)


def _inv_width(log_sigma: float) -> float:
    """Raw value whose activation gives `log_sigma` (inverse of the sigmoid width map)."""
    t = (log_sigma - math.log(SIGMA_MIN)) / (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    t = min(max(t, 1e-4), 1 - 1e-4)
    return math.log(t / (1 - t))
