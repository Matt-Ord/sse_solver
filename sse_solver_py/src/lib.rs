use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use sse_solver::{
    solvers::{EulerSolver, MilstenSolver, Order2WeakSolver, Solver},
    sparse::BandedArray,
    sse_system::{FullNoise, SSESystem},
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_euler(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: Vec<Vec<Complex<f64>>>,
    operators: Vec<Vec<Vec<Complex<f64>>>>,
    n: usize,
    step: usize,
    dt: f64,
) -> PyResult<Vec<Complex<f64>>> {
    if initial_state.len() != hamiltonian.len() || hamiltonian[1].len() != hamiltonian.len() {
        return Err(PyAssertionError::new_err("Hamiltonian has bad shape"));
    }
    if initial_state.len() != operators[1].len() || operators[1].len() != operators[2].len() {
        return Err(PyAssertionError::new_err("Hamiltonian has bad shape"));
    }
    let noise = FullNoise::from_operators(
        &Array3::from_shape_vec(
            (operators.len(), initial_state.len(), initial_state.len()),
            operators.into_iter().flatten().flatten().collect(),
        )
        .unwrap(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: Array2::from_shape_vec(
            (initial_state.len(), initial_state.len()),
            hamiltonian.into_iter().flatten().collect(),
        )
        .unwrap(),
    };

    let initial_state = Array1::from(initial_state);
    let out = EulerSolver::solve(&initial_state, &system, n, step, dt);

    Ok(out.into_raw_vec())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_euler_banded(
    initial_state: Vec<Complex<f64>>,
    hamiltonian_diagonal: Vec<Vec<Complex<f64>>>,
    hamiltonian_offset: Vec<usize>,
    operators_diagonals: Vec<Vec<Vec<Complex<f64>>>>,
    operators_offsets: Vec<Vec<usize>>,
    n: usize,
    step: usize,
    dt: f64,
) -> PyResult<Vec<Complex<f64>>> {
    let shape = [initial_state.len(), initial_state.len()];
    let noise = FullNoise::from_banded(
        &operators_diagonals
            .iter()
            .zip(operators_offsets.iter())
            .map(|(diagonals, offsets)| BandedArray::from_sparse(diagonals, offsets, &shape))
            .collect::<Vec<_>>(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: BandedArray::from_sparse(&hamiltonian_diagonal, &hamiltonian_offset, &shape),
    };

    let initial_state = Array1::from(initial_state);
    let out = EulerSolver::solve(&initial_state, &system, n, step, dt);

    Ok(out.into_raw_vec())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_milsten_banded(
    initial_state: Vec<Complex<f64>>,
    hamiltonian_diagonal: Vec<Vec<Complex<f64>>>,
    hamiltonian_offset: Vec<usize>,
    operators_diagonals: Vec<Vec<Vec<Complex<f64>>>>,
    operators_offsets: Vec<Vec<usize>>,
    n: usize,
    step: usize,
    dt: f64,
) -> PyResult<Vec<Complex<f64>>> {
    let shape = [initial_state.len(), initial_state.len()];
    let noise = FullNoise::from_banded(
        &operators_diagonals
            .iter()
            .zip(operators_offsets.iter())
            .map(|(diagonals, offsets)| BandedArray::from_sparse(diagonals, offsets, &shape))
            .collect::<Vec<_>>(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: BandedArray::from_sparse(&hamiltonian_diagonal, &hamiltonian_offset, &shape),
    };

    let initial_state = Array1::from(initial_state);
    let out = MilstenSolver::solve(&initial_state, &system, n, step, dt);

    Ok(out.into_raw_vec())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_second_order_banded(
    initial_state: Vec<Complex<f64>>,
    hamiltonian_diagonal: Vec<Vec<Complex<f64>>>,
    hamiltonian_offset: Vec<usize>,
    operators_diagonals: Vec<Vec<Vec<Complex<f64>>>>,
    operators_offsets: Vec<Vec<usize>>,
    n: usize,
    step: usize,
    dt: f64,
) -> PyResult<Vec<Complex<f64>>> {
    let shape = [initial_state.len(), initial_state.len()];
    let noise = FullNoise::from_banded(
        &operators_diagonals
            .iter()
            .zip(operators_offsets.iter())
            .map(|(diagonals, offsets)| BandedArray::from_sparse(diagonals, offsets, &shape))
            .collect::<Vec<_>>(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: BandedArray::from_sparse(&hamiltonian_diagonal, &hamiltonian_offset, &shape),
    };

    let initial_state = Array1::from(initial_state);
    let out = Order2WeakSolver::solve(&initial_state, &system, n, step, dt);

    Ok(out.into_raw_vec())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_euler_bra_ket(
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

    let noise = FullNoise::from_bra_ket(
        amplitudes,
        &Array2::from_shape_vec((n_amplitudes, initial_state.len()), bra).unwrap(),
        &Array2::from_shape_vec((n_amplitudes, initial_state.len()), ket).unwrap(),
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
fn _solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sse_euler, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_euler_bra_ket, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_euler_banded, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_milsten_banded, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_second_order_banded, m)?)?;
    Ok(())
}
