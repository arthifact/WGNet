//! `wave_gauss_rs`: PyO3 bindings for the native wgpu multi-shot forward / exact discrete adjoint
//! acoustic solver (`PyWaveEngine`) plus the Ricker wavelet and CPU reference solver used to verify it.

pub mod wave_wgpu;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn ricker_wavelet<'py>(py: Python<'py>, samples: usize, dt: f32, peak_hz: f32) -> Bound<'py, PyArray1<f32>> {
    let mut out = vec![0.0f32; samples];
    geology_core::ricker_wavelet(samples, dt, peak_hz, &mut out);
    out.into_pyarray(py)
}

/// Single-shot CPU reference solver. Returns `[nt, 70]` traces.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn simulate_shot_cpu<'py>(
    py: Python<'py>,
    vel_mps: PyReadonlyArray1<'_, f32>,
    h: f32,
    dt: f32,
    nt: usize,
    pml: usize,
    vmin: f32,
    vmax: f32,
    peak_hz: f32,
    src_x: f32,
    src_z: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let slice = vel_mps.as_slice().map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
    if slice.len() != geology_core::CELLS {
        return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
    }
    let mut out = vec![0.0f32; nt * geology_core::N];
    py.allow_threads(|| {
        geology_core::simulate_shot_cpu(slice, h, dt, nt, pml, vmin, vmax, peak_hz, src_x, src_z, &mut out);
    });
    PyArray1::from_vec(py, out).reshape([nt, geology_core::N])
}

#[pyclass]
pub struct PyWaveEngine {
    inner: wave_wgpu::NativeWaveEngine,
}

#[pymethods]
impl PyWaveEngine {
    #[new]
    #[pyo3(signature = (shots=5, nt=2001, dt=0.0005, peak_hz=15.0, pml=20, vmin=1000.0, vmax=5000.0))]
    fn new(shots: usize, nt: usize, dt: f32, peak_hz: f32, pml: usize, vmin: f32, vmax: f32) -> PyResult<Self> {
        let inner = wave_wgpu::NativeWaveEngine::new(shots, nt, dt, peak_hz, pml, vmin, vmax)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn shots(&self) -> usize {
        self.inner.shots
    }

    #[getter]
    fn nt(&self) -> usize {
        self.inner.nt
    }

    #[getter]
    fn nrec(&self) -> usize {
        self.inner.nrec
    }

    /// Current source positions `[shots, 2]` in meters.
    fn sources<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat: Vec<f32> = self.inner.sources.iter().flat_map(|p| [p[0], p[1]]).collect();
        PyArray1::from_vec(py, flat).reshape([self.inner.shots, 2])
    }

    /// Update source positions `[shots, 2]` in meters.
    fn set_sources(&mut self, sources: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let shape = sources.shape();
        if shape[0] != self.inner.shots || shape[1] != 2 {
            return Err(PyValueError::new_err("sources must have shape [shots, 2]"));
        }
        let slice = sources.as_slice().map_err(|_| PyValueError::new_err("sources must be contiguous"))?;
        let pts: Vec<[f32; 2]> = (0..self.inner.shots).map(|s| [slice[2 * s], slice[2 * s + 1]]).collect();
        self.inner.set_sources(&pts);
        Ok(())
    }

    /// Current source wavelet `[nt]`.
    fn wavelet<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.wavelet.clone().into_pyarray(py)
    }

    /// Replace the source wavelet `[nt]` (used for frequency continuation: a low-passed wavelet
    /// yields low-passed synthetics by linearity of the wave equation).
    fn set_wavelet(&mut self, wavelet: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let slice = wavelet.as_slice().map_err(|_| PyValueError::new_err("wavelet must be contiguous"))?;
        if slice.len() != self.inner.nt {
            return Err(PyValueError::new_err("wavelet length must equal nt"));
        }
        self.inner.set_wavelet(slice);
        Ok(())
    }

    /// Simulate all shots for a 70x70 velocity model in m/s. Returns `[shots, nt, nrec]` float32.
    fn simulate<'py>(&self, py: Python<'py>, vel_mps: PyReadonlyArray1<'_, f32>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let slice = vel_mps.as_slice().map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
        if slice.len() != geology_core::CELLS {
            return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
        }
        let traces = py.allow_threads(|| self.inner.simulate(slice));
        PyArray1::from_vec(py, traces).reshape([self.inner.shots, self.inner.nt, self.inner.nrec])
    }

    /// Exact forward + discrete adjoint across all shots. Returns `(loss, grad_mps [70, 70])` where
    /// `loss = 0.5 * ||p - d||^2 / ||d||^2` and `grad_mps = dloss / d(v in m/s)`.
    fn evaluate_adjoint<'py>(
        &self,
        py: Python<'py>,
        vel_mps: PyReadonlyArray1<'_, f32>,
        observed: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<(f32, Bound<'py, PyArray2<f32>>)> {
        let v_slice = vel_mps.as_slice().map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
        let obs_slice = observed.as_slice().map_err(|_| PyValueError::new_err("observed must be contiguous"))?;
        if v_slice.len() != geology_core::CELLS {
            return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
        }
        if obs_slice.len() != self.inner.shots * self.inner.nt * self.inner.nrec {
            return Err(PyValueError::new_err("observed length does not match shots * nt * nrec"));
        }
        let res = py.allow_threads(|| self.inner.evaluate_adjoint(v_slice, obs_slice));
        let grad = PyArray1::from_vec(py, res.grad_mps).reshape([geology_core::N, geology_core::N])?;
        Ok((res.loss, grad))
    }
}

#[pymodule]
fn wave_gauss_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ricker_wavelet, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_shot_cpu, m)?)?;
    m.add_class::<PyWaveEngine>()?;
    Ok(())
}
