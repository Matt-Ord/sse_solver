use ndarray::{Array1, Array2};
// use ndarray_linalg::Scalar;
use ::sse_solver::{DiagonalNoise, EulerSolver, SSESystem, Solver};
use num_complex::Complex;
use pyo3::prelude::*;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn solve_sse_euler(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: Vec<Complex<f64>>,
    amplitudes: Vec<Complex<f64>>,
    bra: Vec<Complex<f64>>,
    ket: Vec<Complex<f64>>,
    n: usize,
    step: usize,
    dt: f64,
) -> PyResult<Vec<Complex<f64>>> {
    let amplitudes = Array1::from_vec(amplitudes);
    let n_amplitudes = amplitudes.len();
    let noise = DiagonalNoise::from_bra_ket(
        amplitudes,
        Array2::from_shape_vec((n_amplitudes, initial_state.len()), bra).unwrap(),
        Array2::from_shape_vec((n_amplitudes, initial_state.len()), ket).unwrap(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: Array2::from_shape_vec(
            (initial_state.len(), initial_state.len()),
            hamiltonian,
        )
        .unwrap(),
    };

    let initial_state = Array1::from(initial_state);
    let out = EulerSolver::solve(&initial_state, &system, n, step, dt);

    Ok(out.into_raw_vec())
}

/// A Python module implemented in Rust.
#[pymodule]
fn sse_solver(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sse_euler, m)?)?;
    Ok(())
}
