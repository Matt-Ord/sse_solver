use ndarray::Array1;
use num_complex::Complex;
use pyo3::prelude::*;
use sse_solver::solvers::{MeasurementError, NormalizedStateMeasurement, StateMeasurement};

#[pyclass(eq, eq_int)]
#[derive(PartialEq)]
enum StateMeasurementPy {
    Normalized,
    Unnormalized,
}

#[pymethods]
impl StateMeasurementPy {
    #[new]
    fn new(normalize: bool) -> Self {
        if normalize {
            StateMeasurementPy::Normalized
        } else {
            StateMeasurementPy::Unnormalized
        }
    }
}

pub trait SDEMeasurementCollection {
    fn add_measurement(&mut self, state: &Array1<Complex<f64>>) -> Result<(), MeasurementError>;
    fn with_capacity(capacity: usize) -> Self;
}

#[pymodule]
fn _measurement(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StateMeasurementPy>()?;
    m.add_class::<BandedData>()?;
    m.add_class::<SplitOperatorData>()?;

    Ok(())
}
