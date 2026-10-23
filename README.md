# WGNet: Wave-to-Gaussian Seismic Inversion

Amortized seismic velocity inversion that outputs an explicit, refinable scene: a dense velocity field plus
a sparse set of anisotropic 2D Gaussians on a fixed anchor lattice, trained with an exact discrete
wave-equation adjoint and refined at test time against the observed shot gathers.

```
shot gathers [S, T, 70]  ──►  gather transformer  ──►  35x35 depth queries  ──►  dense head  ┐
   (any number of shots)       (axial attention,         (cross-attention with                 ├─►  v(x, z)
                                per shot)                 traveltime-biased heads)   Gaussian head ┘
                                                                                          │
                                                               exact wgpu adjoint  ◄──────┘  (training loss and test-time L-BFGS)
```

## Layout

```
crates/
  geology-core/        Ricker wavelet, C-PML profiles, CPU reference solver (tests the GPU kernels)
  wave-gauss-rs/       wgpu forward + exact discrete adjoint (Metal / Vulkan / DX12), PyO3 bindings
    src/shaders/       splatForward.wgsl, splatAdjoint.wgsl
python/
  wgnet/data.py        OpenFWI split (seeded, duplicate-checked) and float16 4 ms shot-gather cache
  wgnet/model.py       WGNet: stem, axial gather blocks, traveltime-biased decoder, dense + Gaussian heads
  wgnet/render.py      local-window additive Gaussian renderer, SSIM, MAE / RMSE metrics
  wgnet/physics.py     exact data misfit + gradient from the adjoint, causal low-pass frequency continuation
  wgnet/refine.py      budgeted projected L-BFGS over (coarse background, active Gaussians) with band schedule
  wgnet/probes.py      counterfactual wave-attribution probes (mutes, offset masks, phase scrambling)
  wgnet/train.py       training / evaluation CLI (supervised stage, physics stage, test reports)
  verify.py            gradient checks, band-limit identity, forward/backward, refinement, overfit smoke test
data/openfwi/          OpenFWI velocity models (symlink or copy; see renderwave's fetch script)
docs/v1_review.md      review of the previous architecture and the reasoning behind this one
```

## Setup

```bash
# Rust toolchain (rustup) and a Python >= 3.11 environment with torch; then
cd python && uv venv --python 3.12 .venv && uv pip install torch numpy maturin -e .
VIRTUAL_ENV=$PWD/.venv maturin develop --release --uv --features extension-module -m ../crates/wave-gauss-rs/Cargo.toml
```

## Run

```bash
cargo test --release                                  # kernel parity and adjoint-vs-finite-difference tests
python/.venv/bin/python python/verify.py              # end-to-end gradient and smoke checks

# supervised stage (defaults: 23000 train / 1000 val / 6000 test models per family, i.e. all of OpenFWI; --val-limit 2000 for faster evals)
python/.venv/bin/python -m wgnet.train --stage sup --out results/base --eval-test
# physics stage from the supervised checkpoint
python/.venv/bin/python -m wgnet.train --stage phys --init results/base/best.pt --out results/base_phys --steps 8000 --eval-test
# evaluate a checkpoint on the full validation and test splits with 5-, 3- and 1-shot surveys
python/.venv/bin/python -m wgnet.train --eval-only results/base_phys/best.pt --eval-test
```

The first run simulates the shot gathers for the chosen split with the wgpu engine (about 50 models/s on an
M2 Ultra) into `data/cache_*/waves_*.f16.bin`. Metrics follow OpenFWI: MAE, RMSE (m/s), SSIM on
`(v - 1500) / 3000`, plus normalized MAE / MSE, per family and overall.

## Design notes

- **Encoder.** Tokens are (time, receiver) cells of each shot gather after a conv stem (63 x 35 per shot),
  with Fourier features of time, receiver, source, offset and midpoint. Axial attention mixes each gather;
  1225 depth queries then cross-attend to every token of every shot, so any number of shots at any
  positions is handled without a fixed-geometry assumption. Four heads carry a fixed traveltime bias
  (two-way time through trial RMS velocities, including the 0.1 s source delay); the rest attend freely.
- **Scene.** `field = dense + render(gaussians)`. One Gaussian slot per 2x2 cell with bounded offset,
  scales, angle, residual amplitude and an existence weight penalized by L1, so Gaussians appear only where
  they buy accuracy (interfaces, faults). Rendering is additive within a 25x25 window per slot.
- **Physics.** `WavePhysics.loss_and_grad` returns the normalized misfit and its exact gradient; in training
  its gradient is scaled to a fixed fraction of the supervised gradient norm, and the passband follows a
  6 -> 9 -> 12 -> 15 Hz schedule via a causal low-pass applied to the wavelet and to the data (an exact
  identity, verified in `verify.py`).
- **Refinement.** `refine()` optimizes a 9x9 background correction and the active Gaussians with projected
  L-BFGS under a solve budget, cycling through passbands. The true model never enters.
