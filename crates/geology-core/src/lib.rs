//! Acoustic wave-equation building blocks shared by the CPU reference solver and the wgpu engine:
//! grid constants, the delayed Ricker wavelet, deepwave-style 4th-order C-PML profiles, and a
//! single-shot CPU reference solver used to validate the GPU kernels.

pub const N: usize = 70;
pub const CELLS: usize = N * N;
pub const SPACING_M: f32 = 10.0;
pub const H: f32 = SPACING_M;
pub const FD_PAD: usize = 2;

/// Ricker wavelet delayed by `1.5 / peak_hz` (so the peak sits at index `1.5 / (peak_hz * dt)`).
pub fn ricker_wavelet(samples: usize, dt: f32, peak_hz: f32, out: &mut [f32]) {
    assert_eq!(out.len(), samples);
    let delay = 1.5 / peak_hz;
    for (i, slot) in out.iter_mut().enumerate() {
        let t = (i as f32) * dt - delay;
        let u = std::f32::consts::PI * peak_hz * t;
        let a = u * u;
        *slot = (1.0 - 2.0 * a) * (-a).exp();
    }
}

/// Deepwave's 4th-order C-PML profiles along one axis of `n` padded cells: a, b, and db/dx.
#[allow(clippy::too_many_arguments)]
pub fn cpml_profiles(
    n: usize,
    h: f32,
    dt: f32,
    width: usize,
    max_vel: f32,
    freq: f32,
    a_out: &mut [f32],
    b_out: &mut [f32],
    db_out: &mut [f32],
) {
    assert!(a_out.len() == n && b_out.len() == n && db_out.len() == n);
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let start0 = (FD_PAD + width) as f64;
    let start1 = (n - 1 - FD_PAD - width) as f64;
    let width_f = width as f64;
    let h_f = h as f64;
    let dt_f = dt as f64;
    let sigma0 = -(3.0 * (max_vel as f64) * (0.001f64).ln()) / (2.0 * width_f * h_f);

    for x in 0..n {
        let xf = x as f64;
        let f0 = (start0 - xf) / width_f;
        let f1 = (xf - start1) / width_f;
        let frac = f0.max(f1).clamp(0.0, 1.0);
        let sigma = sigma0 * frac * frac;
        let alpha = std::f64::consts::PI * (freq as f64) * (1.0 - frac);
        let ax = (-(sigma + alpha) * dt_f).exp();
        b[x] = (sigma / (sigma + alpha)) * (ax - 1.0);
        a[x] = if frac == 0.0 { 0.0 } else { ax };
    }

    let mut db = vec![0.0f64; n];
    for x in 2..(n.saturating_sub(2)) {
        db[x] = ((8.0 / 12.0) * (b[x + 1] - b[x - 1]) - (1.0 / 12.0) * (b[x + 2] - b[x - 2])) / h_f;
    }

    for i in 0..n {
        a_out[i] = a[i] as f32;
        b_out[i] = b[i] as f32;
        db_out[i] = db[i] as f32;
    }
}

/// CPU reference 4th-order finite-difference acoustic solver with C-PML for a single shot.
/// `v_vis` is the 70 x 70 visible velocity model in m/s; `out_traces` has length `nt * N`
/// and receives the surface recording at `(x = r * h, z = h)` before each time step.
#[allow(clippy::too_many_arguments)]
pub fn simulate_shot_cpu(
    v_vis: &[f32],
    h: f32,
    dt: f32,
    nt: usize,
    pml: usize,
    vmin: f32,
    vmax: f32,
    peak_hz: f32,
    src_x: f32,
    src_z: f32,
    out_traces: &mut [f32],
) {
    assert_eq!(v_vis.len(), CELLS);
    assert_eq!(out_traces.len(), nt * N);

    let pad = FD_PAD + pml;
    let nx = N + 2 * pad;
    let nz = N + 2 * pad;
    let cells = nx * nz;
    let region = pml + 2 * FD_PAD;
    let rh = 1.0f32 / h;
    let rh2 = rh * rh;
    let dt2 = dt * dt;

    let mut vel = vec![0.0f32; cells];
    for z in 0..nz {
        let vz = (z as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
        for x in 0..nx {
            let vx = (x as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
            vel[z * nx + x] = v_vis[vz * N + vx].clamp(vmin, vmax);
        }
    }

    let (mut ax, mut bx, mut dbx) = (vec![0.0f32; nx], vec![0.0f32; nx], vec![0.0f32; nx]);
    let (mut az, mut bz, mut dbz) = (vec![0.0f32; nz], vec![0.0f32; nz], vec![0.0f32; nz]);
    cpml_profiles(nx, h, dt, pml, vmax, peak_hz, &mut ax, &mut bx, &mut dbx);
    cpml_profiles(nz, h, dt, pml, vmax, peak_hz, &mut az, &mut bz, &mut dbz);

    let mut wavelet = vec![0.0f32; nt];
    ricker_wavelet(nt, dt, peak_hz, &mut wavelet);

    let gx = src_x / h + (pad as f32);
    let gz = src_z / h + (pad as f32);
    let x0 = (gx.floor() as usize).min(nx - 2);
    let z0 = (gz.floor() as usize).min(nz - 2);
    let fx = gx - (x0 as f32);
    let fz = gz - (z0 as f32);
    let src_corners = [
        (z0 * nx + x0, (1.0 - fx) * (1.0 - fz)),
        (z0 * nx + x0 + 1, fx * (1.0 - fz)),
        ((z0 + 1) * nx + x0, (1.0 - fx) * fz),
        ((z0 + 1) * nx + x0 + 1, fx * fz),
    ];
    let rec_z = pad + 1;

    let mut cur = vec![0.0f32; cells];
    let mut prev = vec![0.0f32; cells];
    let mut psi_x = vec![0.0f32; cells];
    let mut psi_x_next = vec![0.0f32; cells];
    let mut psi_z = vec![0.0f32; cells];
    let mut psi_z_next = vec![0.0f32; cells];
    let mut zeta_x = vec![0.0f32; cells];
    let mut zeta_z = vec![0.0f32; cells];

    const C1: f32 = 0.6666666666666666;
    const C2: f32 = 0.08333333333333333;
    const D0: f32 = -2.5;
    const D1: f32 = 1.3333333333333333;
    const D2: f32 = 0.08333333333333333;

    for t in 0..nt {
        let row = &mut out_traces[t * N..(t + 1) * N];
        for r in 0..N {
            row[r] = cur[rec_z * nx + (pad + r)];
        }

        for z in FD_PAD..(nz - FD_PAD) {
            let in_pml_z = z < region || z >= nz - region;
            for x in FD_PAD..(nx - FD_PAD) {
                let i = z * nx + x;
                let u = cur[i];
                let mut w = 0.0f32;

                let um1x = cur[i - 1];
                let up1x = cur[i + 1];
                let um2x = cur[i - 2];
                let up2x = cur[i + 2];
                let d2x = (D0 * u + D1 * (up1x + um1x) - D2 * (up2x + um2x)) * rh2;
                if x < region || x >= nx - region {
                    let d1x = (C1 * (up1x - um1x) - C2 * (up2x - um2x)) * rh;
                    let a = ax[x];
                    let b = bx[x];
                    let ap1 = ax[x + 1] * psi_x[i + 1];
                    let am1 = ax[x - 1] * psi_x[i - 1];
                    let ap2 = ax[x + 2] * psi_x[i + 2];
                    let am2 = ax[x - 2] * psi_x[i - 2];
                    let d1_apsi = (C1 * (ap1 - am1) - C2 * (ap2 - am2)) * rh;
                    let tmp = (1.0 + b) * d2x + dbx[x] * d1x + d1_apsi;
                    w += (1.0 + b) * tmp + a * zeta_x[i];
                    psi_x_next[i] = b * d1x + a * psi_x[i];
                    zeta_x[i] = b * tmp + a * zeta_x[i];
                } else {
                    w += d2x;
                    psi_x_next[i] = 0.0;
                }

                let um1z = cur[i - nx];
                let up1z = cur[i + nx];
                let um2z = cur[i - 2 * nx];
                let up2z = cur[i + 2 * nx];
                let d2z = (D0 * u + D1 * (up1z + um1z) - D2 * (up2z + um2z)) * rh2;
                if in_pml_z {
                    let d1z = (C1 * (up1z - um1z) - C2 * (up2z - um2z)) * rh;
                    let a = az[z];
                    let b = bz[z];
                    let ap1 = az[z + 1] * psi_z[i + nx];
                    let am1 = az[z - 1] * psi_z[i - nx];
                    let ap2 = az[z + 2] * psi_z[i + 2 * nx];
                    let am2 = az[z - 2] * psi_z[i - 2 * nx];
                    let d1_apsi = (C1 * (ap1 - am1) - C2 * (ap2 - am2)) * rh;
                    let tmp = (1.0 + b) * d2z + dbz[z] * d1z + d1_apsi;
                    w += (1.0 + b) * tmp + a * zeta_z[i];
                    psi_z_next[i] = b * d1z + a * psi_z[i];
                    zeta_z[i] = b * tmp + a * zeta_z[i];
                } else {
                    w += d2z;
                    psi_z_next[i] = 0.0;
                }

                let v = vel[i];
                prev[i] = v * v * dt2 * w + 2.0 * u - prev[i];
            }
        }

        let f_t = wavelet[t];
        for &(cell, weight) in &src_corners {
            if weight != 0.0 {
                let v = vel[cell];
                prev[cell] += -(v * v * dt2) * weight * f_t;
            }
        }

        std::mem::swap(&mut cur, &mut prev);
        std::mem::swap(&mut psi_x, &mut psi_x_next);
        std::mem::swap(&mut psi_z, &mut psi_z_next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ricker_peak_and_cpml_profile() {
        let mut w = vec![0.0f32; 2001];
        ricker_wavelet(2001, 0.0005, 15.0, &mut w);
        assert!((w[200] - 1.0).abs() < 1e-6, "peak at t = 1.5 / 15 Hz = 0.1 s");
        assert!(w[0].abs() < 1e-3);

        let n = 114;
        let (mut a, mut b, mut db) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        cpml_profiles(n, 10.0, 0.0005, 20, 5000.0, 15.0, &mut a, &mut b, &mut db);
        assert_eq!(a[55], 0.0);
        assert_eq!(b[55], 0.0);
        assert!(a[5] > 0.0 && b[5] < 0.0);
    }

    #[test]
    fn cpu_wave_solver_propagates_and_stays_finite() {
        let v = vec![2500.0f32; CELLS];
        let nt = 400;
        let mut traces = vec![0.0f32; nt * N];
        simulate_shot_cpu(&v, 10.0, 0.0005, nt, 20, 1000.0, 5000.0, 15.0, 340.0, 10.0, &mut traces);
        let energy: f64 = traces.iter().map(|&x| (x as f64) * (x as f64)).sum();
        assert!(energy > 1e-4, "wavefield must reach surface receivers");
        assert!(traces.iter().all(|x| x.is_finite()));
    }
}
