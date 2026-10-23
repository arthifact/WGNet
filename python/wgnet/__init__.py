"""WGNet: wave-to-Gaussian seismic inversion (gather transformer + Gaussian residual scene + exact adjoint)."""

from .model import WGNet, WGNetConfig, Prediction
from .render import render_residual, activate_gaussians, field_metrics
from .physics import WavePhysics
from .refine import refine

__all__ = ["WGNet", "WGNetConfig", "Prediction", "render_residual", "activate_gaussians", "field_metrics", "WavePhysics", "refine"]
