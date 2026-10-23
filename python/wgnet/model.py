"""WGNet: gather transformer encoder with traveltime-biased cross-attention, dense velocity head,
and a residual Gaussian head on a fixed anchor lattice.

Pipeline for a survey of S shots (any S, any source positions):
    traces [B, S, T, 70]
      -> 3 physical channels (true amplitude, time-gain, time derivative), decimated to 4 ms
      -> per-shot conv stem: [B*S, d, 63 time tokens, 35 receiver tokens]
      -> axial self-attention per shot (time axis, then receiver axis)
      -> 35x35 depth queries cross-attend to all S*63*35 tokens; the first M heads carry a fixed
         traveltime bias -((t_token - tau(query, source, receiver; v_m))^2) for M trial RMS velocities
      -> feature map [B, d, 35, 35]
      -> dense head (70x70 velocity around the training mean) + Gaussian residual head (1225 slots,
         amplitude doubles as existence)
      -> field = dense + render(gaussians)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .render import GRID, N_GAUSS_PARAMS, N_SLOTS, SLOT_SIDE, activate_gaussians, render_residual

H_M = 10.0
XMAX_M = (GRID - 1) * H_M        # 690 m
STATION_DEPTH_M = 10.0           # sources and receivers sit one cell below the surface
TIME_TOKENS = 63
REC_TOKENS = 35
DEC_SAMPLES = 251
DEC_DT = 0.004
RICKER_DELAY_S = 1.5 / 15.0      # peak delay of the 15 Hz Ricker source
V_BANK_MPS = (1600.0, 2200.0, 2900.0, 3800.0)


@dataclass
class WGNetConfig:
    d_model: int = 384
    n_heads: int = 8
    gather_layers: int = 4
    decoder_layers: int = 6
    mlp_ratio: int = 4
    head_channels: int = 128
    dropout: float = 0.0
    grad_checkpoint: bool = False  # recompute transformer blocks in backward (trades ~30% compute for memory)
    dense_lowres: bool = False     # predict the dense field at 35x35 and upsample, so sharp structure must come from Gaussians

    @staticmethod
    def small() -> "WGNetConfig":
        return WGNetConfig(d_model=192, n_heads=6, gather_layers=2, decoder_layers=3, head_channels=96)


class Prediction(NamedTuple):
    field_km: torch.Tensor       # [B, 70, 70] dense + Gaussian residual
    dense_km: torch.Tensor       # [B, 70, 70] dense head alone
    residual_km: torch.Tensor    # [B, 70, 70] rendered Gaussian residual
    gaussians: torch.Tensor      # [B, 1225, 6] activated slot parameters
    feat35: torch.Tensor         # [B, d, 35, 35] decoder features


def fourier_features(x: torch.Tensor, n_freq: int = 6) -> torch.Tensor:
    """`[..., k]` normalized coordinates in roughly [-1, 1] -> `[..., k * 2 * n_freq]` sin/cos features."""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    ang = x.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None, bias: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if ctx is None else ctx
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if bias is not None and x.device.type != "cuda":
            # Without a fused kernel the scores [B, H, N, K] are materialized; keep each chunk under 2^30 elements.
            chunk = max(1, (1 << 30) // (self.heads * n * k.shape[2]))
            o = torch.cat([F.scaled_dot_product_attention(q[i : i + chunk], k[i : i + chunk], v[i : i + chunk], attn_mask=bias, dropout_p=p)
                           for i in range(0, b, chunk)], dim=0)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p)
        return self.out(o.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Sequential):
    def __init__(self, dim: int, ratio: int, dropout: float) -> None:
        super().__init__(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ratio * dim, dim))


class AxialBlock(nn.Module):
    """Pre-norm transformer block on `[B, T, R, d]` gather tokens: attention along time, then along receivers."""

    def __init__(self, dim: int, heads: int, ratio: int, dropout: float) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_t = Attention(dim, heads, dropout)
        self.attn_r = Attention(dim, heads, dropout)
        self.mlp = Mlp(dim, ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, r, d = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b * r, t, d)
        h = h + self.attn_t(self.n1(h))
        h = h.view(b, r, t, d).permute(0, 2, 1, 3).reshape(b * t, r, d)
        h = h + self.attn_r(self.n2(h))
        h = h.view(b, t, r, d)
        return h + self.mlp(self.n3(h))


class DecoderBlock(nn.Module):
    """Depth-query block: self-attention among queries, traveltime-biased cross-attention to gather tokens, MLP."""

    def __init__(self, dim: int, heads: int, ratio: int, dropout: float) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.n_ctx = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads, dropout)
        self.cross_attn = Attention(dim, heads, dropout)
        self.mlp = Mlp(dim, ratio, dropout)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        q = q + self.self_attn(self.n1(q))
        q = q + self.cross_attn(self.n2(q), self.n_ctx(ctx), bias)
        return q + self.mlp(self.n3(q))


class ConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.g1 = nn.GroupNorm(8, ch)
        self.g2 = nn.GroupNorm(8, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.g1(self.c1(x)))
        return x + self.g2(self.c2(h))


class TraveltimeBias(nn.Module):
    """Fixed additive attention bias between depth queries and gather tokens (a physical prior, not learned:
    SDPA kernels do not reliably propagate gradients into a float mask, so a learnable mask would train
    silently on some devices and not on others).

    For query point P = (x_q, z_q), source S, receiver R and trial RMS velocity v_m:
        tau_m = (|P - S| + |P - R|) / v_m + ricker_delay
        bias[h = m] = -(t_token - tau_m)^2 / (2 sigma_t^2)        for the first M heads
        bias[h >= M] = 0                                           remaining heads attend freely
    The bias depends only on the source configuration and is cached per configuration.
    """

    def __init__(self, heads: int, sigma_t_s: float = 0.06) -> None:
        super().__init__()
        self.heads = heads
        self.inv_two_sigma2 = 1.0 / (2.0 * sigma_t_s**2)
        self.register_buffer("v_bank", torch.tensor(V_BANK_MPS), persistent=False)
        self._cache: dict[tuple, torch.Tensor] = {}

    def forward(self, source_x_m: torch.Tensor, query_xz_m: torch.Tensor | None = None) -> torch.Tensor:
        """`query_xz_m [Q, 2]` query points in meters; default: the 35x35 depth grid at cell centers."""
        key = (str(source_x_m.device), tuple(round(float(x), 3) for x in source_x_m.tolist()),
               None if query_xz_m is None else (query_xz_m.shape[0], round(float(query_xz_m.sum()), 2)))
        if key in self._cache:
            return self._cache[key]
        device = source_x_m.device
        m = min(len(V_BANK_MPS), self.heads)
        if query_xz_m is None:
            qi = torch.arange(SLOT_SIDE, device=device, dtype=torch.float32)
            qz, qx = torch.meshgrid(20.0 * qi + 10.0, 20.0 * qi + 10.0, indexing="ij")
            qx, qz = qx.reshape(-1), qz.reshape(-1)  # [Q]
        else:
            qx, qz = query_xz_m[:, 0], query_xz_m[:, 1]
        n_q = qx.shape[0]
        dz2 = (qz - STATION_DEPTH_M) ** 2
        t_tok = (torch.arange(TIME_TOKENS, device=device, dtype=torch.float32) * 4.0) * DEC_DT  # [T]
        x_rec = torch.arange(REC_TOKENS, device=device, dtype=torch.float32) * 20.0            # [R]
        d_src = torch.sqrt((qx[:, None] - source_x_m[None, :]) ** 2 + dz2[:, None])            # [Q, S]
        d_rec = torch.sqrt((qx[:, None] - x_rec[None, :]) ** 2 + dz2[:, None])                 # [Q, R]
        dist = d_src[:, :, None] + d_rec[:, None, :]                                            # [Q, S, R]
        tau = dist[None] / self.v_bank[:m].view(-1, 1, 1, 1) + RICKER_DELAY_S                   # [M, Q, S, R]
        diff = t_tok.view(1, 1, 1, -1, 1) - tau.unsqueeze(-2)                                   # [M, Q, S, T, R]
        bias = torch.zeros(1, self.heads, n_q, diff.shape[2] * TIME_TOKENS * REC_TOKENS, device=device)
        bias[0, :m] = -self.inv_two_sigma2 * diff.square().reshape(m, n_q, -1)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = bias
        return bias


class GatherEncoder(nn.Module):
    """Shot gathers -> tokens: physical channels, conv stem, Fourier positions, per-shot axial attention,
    plus the traveltime bias used by whoever cross-attends to the tokens. Shared by WGNet and the
    InitialGaussianNetwork."""

    def __init__(self, rms_scale: float, cfg: WGNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.rms_scale = float(rms_scale)
        t_norm = torch.linspace(0.0, 1.0, DEC_SAMPLES).view(1, 1, DEC_SAMPLES, 1)
        self.register_buffer("tgc_gain", ((t_norm + 0.05) / 0.12).pow(1.4), persistent=False)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, (5, 3), stride=(2, 1), padding=(2, 1)), nn.SiLU(),
            nn.Conv2d(64, 128, (5, 3), stride=(2, 2), padding=(2, 1)), nn.SiLU(),
            nn.Conv2d(128, d, 3, padding=1),
        )
        self.token_pos = nn.Linear(5 * 12, d)
        self.gather_blocks = nn.ModuleList(
            [AxialBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.gather_layers)]
        )
        self.gather_norm = nn.LayerNorm(d)
        self.bias = TraveltimeBias(cfg.n_heads)

    def preprocess(self, traces: torch.Tensor) -> torch.Tensor:
        """`[B, S, T, 70]` recordings at 0.5 ms (T=2001) or 4 ms (T=251) -> `[B*S, 3, 251, 70]` channels."""
        if traces.shape[2] != DEC_SAMPLES:
            step = (traces.shape[2] - 1) // (DEC_SAMPLES - 1)
            traces = traces[:, :, ::step, :][:, :, :DEC_SAMPLES, :]
        b, s = traces.shape[:2]
        x = traces.reshape(b * s, 1, DEC_SAMPLES, GRID)
        ch0 = torch.asinh(x / self.rms_scale)
        ch1 = torch.asinh(x * self.tgc_gain / (0.12 * self.rms_scale))
        ch2 = torch.asinh(F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 1, 0)) / (0.25 * self.rms_scale))
        return torch.cat([ch0, ch1, ch2], dim=1)

    @staticmethod
    def token_positions(source_x_m: torch.Tensor) -> torch.Tensor:
        """`[S, T, R, 60]` Fourier features of (t, x_r, x_s, offset, midpoint), all scaled to about [-1, 1]."""
        device = source_x_m.device
        t = (torch.arange(TIME_TOKENS, device=device, dtype=torch.float32) * 4.0 * DEC_DT) * 2.0 - 1.0
        xr = torch.arange(REC_TOKENS, device=device, dtype=torch.float32) * 20.0 / XMAX_M * 2.0 - 1.0
        xs = source_x_m / XMAX_M * 2.0 - 1.0
        s, tt, rr = len(xs), TIME_TOKENS, REC_TOKENS
        coords = torch.stack(
            [
                t.view(1, tt, 1).expand(s, tt, rr),
                xr.view(1, 1, rr).expand(s, tt, rr),
                xs.view(s, 1, 1).expand(s, tt, rr),
                (xr.view(1, 1, rr) - xs.view(s, 1, 1)).expand(s, tt, rr) * 0.5,
                (xr.view(1, 1, rr) + xs.view(s, 1, 1)).expand(s, tt, rr) * 0.5,
            ],
            dim=-1,
        )
        return fourier_features(coords)

    @staticmethod
    def default_sources(traces: torch.Tensor, source_x_m: torch.Tensor | None) -> torch.Tensor:
        s, device = traces.shape[1], traces.device
        if source_x_m is None:
            source_x_m = torch.linspace(0.0, XMAX_M, s, device=device) if s > 1 else torch.tensor([XMAX_M / 2], device=device)
        return source_x_m.to(device=device, dtype=torch.float32)

    def forward(self, traces: torch.Tensor, source_x_m: torch.Tensor) -> torch.Tensor:
        """`[B, S, T, 70]` recordings -> `[B, S*63*35, d]` gather tokens."""
        b, s = traces.shape[:2]
        feat = self.stem(self.preprocess(traces))  # [B*S, d, 63, 35]
        d = feat.shape[1]
        tok = feat.permute(0, 2, 3, 1).reshape(b, s, TIME_TOKENS, REC_TOKENS, d)
        tok = tok + self.token_pos(self.token_positions(source_x_m)).unsqueeze(0)
        tok = tok.reshape(b * s, TIME_TOKENS, REC_TOKENS, d)
        ckpt = self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.gather_blocks:
            tok = checkpoint(blk, tok, use_reentrant=False) if ckpt else blk(tok)
        return self.gather_norm(tok).reshape(b, s * TIME_TOKENS * REC_TOKENS, d)


class WGNet(nn.Module):
    def __init__(self, mean_km: np.ndarray | torch.Tensor, rms_scale: float, cfg: WGNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or WGNetConfig()
        d = cfg.d_model
        self.encoder = GatherEncoder(rms_scale, cfg)
        mean_t = torch.as_tensor(np.asarray(mean_km), dtype=torch.float32).view(1, 1, GRID, GRID)
        self.register_buffer("base_atanh", torch.atanh(torch.clamp((mean_t - 3.0) * 0.5, -0.999, 0.999)))

        self.query_embed = nn.Parameter(torch.randn(N_SLOTS, d) * 0.02)
        self.query_pos = nn.Linear(2 * 12, d)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d)

        hc = cfg.head_channels
        self.up35 = nn.Sequential(nn.Conv2d(d, hc, 1), ConvBlock(hc), ConvBlock(hc))
        if cfg.dense_lowres:
            self.up70 = None
            self.dense_head = nn.Conv2d(hc, 1, 3, padding=1)
        else:
            self.up70 = nn.Sequential(nn.Conv2d(hc, hc // 2, 3, padding=1), nn.SiLU(), ConvBlock(hc // 2))
            self.dense_head = nn.Conv2d(hc // 2, 1, 3, padding=1)
        self.gauss_head = nn.Sequential(nn.Conv2d(d, hc, 1), nn.SiLU(), nn.Conv2d(hc, N_GAUSS_PARAMS, 1))
        nn.init.normal_(self.dense_head.weight, std=1e-3)
        nn.init.zeros_(self.dense_head.bias)
        nn.init.normal_(self.gauss_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.gauss_head[-1].bias)

    @property
    def rms_scale(self) -> float:
        return self.encoder.rms_scale

    def forward(self, traces: torch.Tensor, source_x_m: torch.Tensor | None = None) -> Prediction:
        if traces.ndim == 3:
            traces = traces.unsqueeze(0)
        b = traces.shape[0]
        device = traces.device
        source_x_m = self.encoder.default_sources(traces, source_x_m)
        ctx = self.encoder(traces, source_x_m)
        d = ctx.shape[-1]
        ckpt = self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()

        qi = torch.arange(SLOT_SIDE, device=device, dtype=torch.float32) / (SLOT_SIDE - 1) * 2.0 - 1.0
        qz, qx = torch.meshgrid(qi, qi, indexing="ij")
        q = self.query_embed + self.query_pos(fourier_features(torch.stack([qx.reshape(-1), qz.reshape(-1)], -1)))
        q = q.unsqueeze(0).expand(b, -1, -1)
        bias = self.encoder.bias(source_x_m).to(ctx.dtype)
        for blk in self.decoder_blocks:
            q = checkpoint(blk, q, ctx, bias, use_reentrant=False) if ckpt else blk(q, ctx, bias)
        feat35 = self.decoder_norm(q).transpose(1, 2).reshape(b, d, SLOT_SIDE, SLOT_SIDE)

        h = self.up35(feat35)
        if self.up70 is None:
            logit = F.interpolate(self.dense_head(h), size=(GRID, GRID), mode="bilinear", align_corners=False)
        else:
            logit = self.dense_head(self.up70(F.interpolate(h, size=(GRID, GRID), mode="bilinear", align_corners=False)))
        dense = (3.0 + 2.0 * torch.tanh(logit + self.base_atanh)).squeeze(1)

        raw = self.gauss_head(feat35).flatten(2).transpose(1, 2)  # [B, 1225, 6]
        gaussians = activate_gaussians(raw)
        residual = render_residual(gaussians)
        return Prediction(dense + residual, dense, residual, gaussians, feat35)


LEGACY_PREFIXES = ("stem.", "token_pos.", "gather_blocks.", "gather_norm.", "bias.")


def remap_legacy_state_dict(state: dict) -> dict:
    """Checkpoints saved before the GatherEncoder split keep their encoder weights at the top level."""
    if any(k.startswith("encoder.") for k in state):
        return state
    return {("encoder." + k if k.startswith(LEGACY_PREFIXES) else k): v for k, v in state.items()}
