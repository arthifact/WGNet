# WGNet review: positioning, bugs, and a path to SOTA

Reviewed at commit `76dd620` (github.com/arthifact/WGNet). Everything below comes from reading the code, the verification JSON, and the stage 2 gallery. I did not run training.

---

## 1. How WGNet differs from mainstream deep-learning FWI

### What the mainstream looks like (OpenFWI era)

| Family | Examples | Output | Physics in training |
|---|---|---|---|
| Direct image-to-image regression | InversionNet, VelocityGAN, InvLINT, Fourier-DeepONet | dense 70x70 grid | none (pure supervision) |
| Unsupervised / physics-loop | UPFWI | dense grid | differentiable forward solver in the loss |
| Generative priors | DiffusionVel, latent-diffusion FWI, SeisFusion | dense grid (sampled) | optional data-fidelity guidance |
| Implicit / compact parametrizations | IFWI (neural field FWI), Gaussian-splat FWI | continuous or sparse params | classic FWI objective, per-model optimization |

### What WGNet does

1. A **wave encoder** with hand-built seismic inductive biases: asinh/TGC/derivative channels, a multi-velocity NMO hyperbola bank, a learned time-to-depth projection, axial depth/horizon mixing, and a virtual-source projection so any number of shots maps to a fixed K=5 modes.
2. A **sparse geometric decoder**: a position-probability map is dithered, Delaunay-triangulated in Rust, and each triangle becomes one anisotropic Gaussian whose velocity is predicted around the training-mean field. A normalized-softmax renderer turns the Gaussians back into a 70x70 field.
3. A **three-stage curriculum**: geological autoencoder, wave encoder distilled onto the frozen decoder, then joint fine-tuning with an exact discrete adjoint from a native wgpu solver.
4. **Test-time L-BFGS refinement** in Gaussian-parameter space against the observed waves (classic FWI, but over ~6N parameters with a strong learned starting point).

So the method sits between "direct regression" and "implicit/compact FWI": the network amortizes the inversion like InversionNet, but the output is a refinable parametric scene like IFWI or Gaussian splatting, and physics enters both at training time (UPFWI-style) and at test time.

### Strengths (real ones)

- **Exact, verified adjoint.** The wgpu adjoint matches finite differences to 1.6e-5 relative error through the renderer. Most "physics-informed" FWI papers use autograd through a Python solver and never verify this.
- **Refinable output.** Because the scene is a parameter vector, you can run a budgeted FWI on top of the prediction. Grid regressors cannot do this without regularization tricks.
- **Acquisition-agnostic encoder.** The virtual-source projection lets one model run on 1, 3, or 5 shots, and the gallery shows graceful degradation. That is genuinely unusual.
- **Interpretability hooks.** The blindfold probes and depth-Jacobian attribution are a good diagnostic suite that most repos lack.
- **Engineering quality of the native core.** CPU/GPU parity tests, resumable wave cache, deterministic split hashing. This part is solid.

### Weaknesses (structural, not bugs)

- **The Gaussian stage currently adds no accuracy.** In stage 2 the training loss touches only `v_aux_km` and the latent distillation; the decoder is never in the loss. At inference the decoder sees `0.25 * wave_latent + 0.75 * geo_encoder(v_aux)`, i.e. it mostly re-encodes the dense auxiliary prediction. The gallery confirms it: Gaussian RMSE equals direct-field RMSE within a few m/s on every row. Right now the whole Gaussian machinery is a lossy re-rendering of a dense CNN output.
- **It is not compact.** CurveVel_A needs 2644 Gaussians, Style_A 1934. At 6 parameters each that is 11k to 16k numbers for a 4900-pixel field. The "sparse scene" argument fails unless count is regularized.
- **Non-differentiable layout.** Dither plus Delaunay is a hard discretization. Tiny PPM changes flip triangles, so the decoder output is piecewise-discontinuous in its inputs. This limits what stage 3 can learn and makes training noisier than a dense head.
- **Renderer cost is O(pixels x Gaussians) with full softmax.** With 2700 padded Gaussians and batch 16, each of the six intermediate tensors is ~850 MB. That is why stage 1 batch is 8 and training is slow. Real splatting is tile-local.
- **Stage 3 is not shown to help.** The smoke test in the JSON goes 510 -> 519 m/s after the physics stage. Nothing in the repo demonstrates the physics loss improving validation error.
- **Data and evaluation scale.** 400 training models per family (4000 total) versus 24,000 per family in OpenFWI, and every reported number is on a 20-sample validation cohort (2 per family). Model selection on 20 samples is close to random. No test-set numbers, no baseline trained on the same data.

---

## 2. Bugs and defects, ordered by impact

### High

1. **NMO hyperbola bank ignores the source delay.** `stage2_wave_enc.py` builds `tau_bank = dist_2way / v_bank` and compares it to normalized time. The Ricker wavelet is delayed by `1.5 / peak_hz = 0.1 s` (see `ricker_wavelet` in `geology-core`), so every reflection actually arrives 0.1 s later than the bank assumes. That is 6 of the 63 time tokens. The learned `v_bank_mps` can only partially absorb this, because a velocity change scales traveltime multiplicatively while the delay is additive. Same bug in the v1 encoder. Fix: `tau = dist / v + 1.5 / PEAK_HZ`, and normalize by the real window `nt_dec * dt_dec` (1.004 s) instead of 1.0.

2. **Stage 2 never trains through the decoder.** See above. Either put `dec_out.field_km` into the stage 2 loss (end to end through the frozen decoder, which is differentiable in the Gaussian attributes) or drop the pretense and treat the Gaussian stage as a post-processor. As written, the "wave-to-Gaussian" model is a wave-to-dense model with an expensive renderer attached.

3. **Validation cohort of 20 drives checkpoint selection.** `get_validation_cohort_indices(per_family=2)` is used in all three stages for `best_state`. With 1000 validation models cached on disk, there is no reason not to evaluate on all of them every `val_every` steps (it is a forward pass on a 1.4 GB tensor already loaded for training).

4. **Physics loss is unbalanced and unverified.** `WavePhysicsLoss` returns `0.5 * ||p-d||^2 / ||d||^2`, about 0.06 at the start. Weighted by 0.15 and applied every second step, its gradient is tiny next to the field MSE in (km/s)^2. Log the gradient norm of each term, and either normalize the physics gradient to a fixed fraction of the geo gradient or use loss-balancing (GradNorm style). Until stage 3 shows a validation gain, do not ship it.

### Medium

5. **Stage 2 resume silently continues from a half-loaded checkpoint.** `train_staged.py` wraps resume in `try/except Exception` and prints "starting fresh". PyTorch's `load_state_dict` copies every key that matches and raises only at the end, so after an architecture change the model keeps the stale weights for matching keys while the optimizer may or may not have loaded. Catch the error before touching the model, or load into a fresh instance.

6. **Stage 3 is not resumable** despite the module docstring. It always starts from the stage 2 best state and overwrites its checkpoint.

7. **Renderer has no Gaussian-count control.** Teacher density is calibrated to the quadtree leaf count, so layered models produce thousands of Gaussians. Add an explicit mass penalty target (for example a cap per model) and a merge step, or the compactness story collapses.

8. **Refinement budget double-counts a forward.** In `refine.py`, an accepted line-search candidate is evaluated once with `simulate` (counted as one forward) and then again with `evaluate_adjoint` (counted as forward plus adjoint). The adjoint call already runs the forward. Each accepted step therefore costs three solves instead of two, and the loss is computed by two different code paths (float64 numpy versus float32 GPU partial sums), which can flip the Armijo test at the margin. Compute the loss once from the adjoint path when you know you will accept.

9. **Time-axis assumptions are hard-coded.** `traces[:, :, ::8, :] if traces.shape[2] > 251` and the `tgc_gain` buffer of length 251 silently break for any `nt` other than 2001 or 251 (the 501-sample engine used in `verify_and_probe.py` would fail if routed through the encoder). Derive decimation from `dt` and `nt`.

### Low / hygiene

10. Comment says the train-wave GPU tensor is 351 MB; it is 1.4 GB (4000 x 5 x 251 x 70 x 4 bytes).
11. Comment says teacher-layout warm-up is 20% of steps; code uses `max(50, 15%)`.
12. Many docstrings still reference `src/direct/network.ts` and other TypeScript files that are gitignored.
13. `scatter_scale` and `mean_bg_traces` are computed and cached but unused; `temporal_attn` (1.5M floats) is returned on every forward.
14. `verify_and_probe.py` asserts nothing about stages 2 and 3, so the smoke test passes even when stage 3 makes things worse.
15. The Nyquist-rate argument for `::8` is fine for a 15 Hz Ricker (4 ms sampling), but there is no anti-alias filter; add a one-line low-pass if you ever raise `peak_hz`.
16. `max_storage_buffer_binding_size` is cast to `u32`; harmless at 70x70, wrong above 4 GB.

Nothing in the shaders or the Rust solver looked wrong, and the tests cover the parts that matter (forward parity, adjoint versus FD, dithering conservation, teacher-density calibration).

---

## 3. What "SOTA" would require

Published OpenFWI numbers (InversionNet, VelocityGAN, UPFWI, DiffusionVel) are on 24k training models per family with the official 1000-sample, 1 ms, 5-shot data, reported as MAE / RMSE / SSIM on the full test split. Your gallery shows 155 to 212 m/s RMSE on A-families from 4000 training models on your own simulator. Those are not comparable in either direction. The order of work:

### Step 0: make the claim testable (one week)

- Evaluate on the full 1000-model validation and 1000-model test splits, per family, with MAE, RMSE, and SSIM as OpenFWI does.
- Train an InversionNet (and ideally a UPFWI) baseline on exactly your 4000-model split and simulator. This is the only comparison that proves the Gaussian path is worth anything.
- Alternatively, switch the encoder to consume OpenFWI's own seismic data (`[5, 1000, 70]` at 1 ms). Then every published number is directly comparable and you can drop the wave cache for training.

### Step 1: fix the cheap things first

- Source delay in the NMO bank (bug 1). Expect a measurable gain on the B families where reflections are what matters.
- Train stage 2 through the decoder (bug 2).
- Full-split validation and checkpoint selection (bug 3).

### Step 2: make the Gaussian stage earn its place

- **Differentiable layout.** Replace dither+Delaunay with a fixed anchor lattice (say 24x24) whose anchors carry learned offsets, scales, and an existence logit, trained with a straight-through or Gumbel relaxation. 3DGS-style prune/densify can run between epochs. This removes the discontinuity and lets stage 3 move Gaussians.
- **Count regularization.** Penalize active count; target a few hundred Gaussians, not thousands.
- **Local renderer.** Render each pixel from its k nearest Gaussians (k = 16 to 32), or tile the field. Memory drops by two orders of magnitude and batch size can go to 64.
- **Residual decoding.** Have the Gaussians predict a residual on top of `v_aux` instead of the whole field, so the sparse set only needs to encode interfaces.

### Step 3: make the physics stage demonstrably help

- Gradient-norm balancing between geo and wave terms.
- Frequency continuation: start the physics loss on low-passed data (5 to 8 Hz) and widen the band over training. This is what makes FWI converge in practice.
- Report test-time refinement numbers from `refine.py` on the full test split, with a budget sweep (40, 80, 160 solves). This is the one capability dense regressors do not have, so it should be the headline result.

### Step 4: modernize the encoder

- The hand-built NMO bank is a nice prior but a fixed one. A cross-attention block where depth-grid queries attend to (shot, receiver, time) keys with traveltime positional encodings generalizes it and is what the docstring already claims.
- Replace the plain conv stack with a modest transformer or ConvNeXt backbone over shot gathers, EMA weights, longer schedules, and shot/receiver dropout augmentation. These are standard wins on OpenFWI.
- If you want to beat generative methods on the B and Style families, the honest route is a diffusion prior over the Gaussian parameter set or over the geological latent, with the adjoint loss as guidance. The Gaussian parametrization actually makes this cheaper than pixel-space diffusion, which is a real selling point.

### Step 5: scale data

- Simulate the full OpenFWI families (or use their provided waves). With 4000 models you are measuring regularization, not the method.

---

## 4. Suggested write-up framing

The defensible novelty is: "amortized inversion into an explicit, refinable Gaussian scene with an exact discrete adjoint, acquisition-agnostic encoder, and budgeted test-time FWI." Lead with the refinement result and the few-shot generalization, not with raw RMSE, until the baselines in Step 0 exist.

---

## 5. Concrete redesign (what to fix and change)

Priority order. The encoder is where the error lives; the Gaussian decoder is a representation advantage to keep but rebuild; the physics loop is the differentiator only if it demonstrably helps.

### 5.1 Encoder: set transformer over gathers
- Tokenize each shot gather into 16-sample x 2-receiver patches at 4 ms (after a low-pass). Token features: Fourier encodings of source x, receiver x, time, offset, midpoint. All shots in one token set; 1 to 5 shots is native; shot dropout as augmentation. Virtual-source projection removed.
- 35x35 learnable depth queries with (x, z) encodings cross-attend to tokens (Perceiver style).
- Keep the hyperbola as a learned attention bias: -(t - tau(x,z,xs,xr; v_bank))^2 / sigma^2, with the 0.1 s Ricker delay included.
- 8 to 12 layers, dim 384 to 512, 20 to 40M params, AdamW, cosine, EMA, bf16.
- Keep asinh / TGC / derivative input channels.
- Drop the geological-autoencoder latent distillation.

### 5.2 Decoder: differentiable, residual, local
- Dense head predicts the auxiliary field; Gaussians predict a residual on top. All losses on the composed field.
- Replace dither + Delaunay with a fixed anchor lattice (e.g. 32x32 slots): per slot predict bounded offset, log scales, angle, value, existence logit. Soft existence in training with L1 sparsity; hard threshold at inference. Fully batched, no CPU round trip, learned count.
- Local additive renderer: each Gaussian contributes inside a fixed window (e.g. 15x15 cells), opacity-weighted additive blending. O(N x window) instead of O(N x pixels).

### 5.3 Two stages instead of three
- Stage A (supervised, end to end): L1 + MS-SSIM + edge loss + sparsity penalty, on OpenFWI's own seismic data at 24k models per family. Evaluate MAE / RMSE / SSIM per family on the full test split against InversionNet, VelocityGAN, UPFWI, InvLINT.
- Stage B (physics fine-tune): exact wgpu adjoint; physics gradient normalized to a fixed fraction of the supervised gradient norm; frequency continuation from ~5 Hz upward; all shots on a batch subset. Add unlabeled gathers from other distributions with only the adjoint loss (UPFWI-style semi-supervision) for OOD generalization.

### 5.4 Test-time refinement
- L-BFGS over the anchor parametrization, multiscale in frequency, solve-budget sweep (40 / 80 / 160). Report misfit and RMSE after refinement on the full test split. This is the headline result.

### 5.5 Optional: generative prior for hard families
- Small diffusion model over the Gaussian parameter set conditioned on the encoder output, adjoint loss as sampling guidance. Only after the regressor is competitive.

### 5.6 Delete
- Geo-encoder latent bridge (0.25 / 0.75 blend), v1 encoder, virtual-source projection, softmax-normalized renderer, Delaunay path in training.
- Keep: Rust engine, adjoint shaders, data manager, probes, refinement loop.

### 5.7 Hardware
- Stage A at this scale needs a CUDA GPU; MPS is impractical. The wgpu adjoint runs on both.
