use std::thread;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use sse_solver::{
    solvers::{
        EulerSolver, LocalizedSolver, MilstenSolver, NormalizedEulerSolver,
        Order2ExplicitWeakSolver, Solver,
    },
    sparse::BandedArray,
    sse_system::{FullNoise, SSESystem},
    system::SDESystem,
};

enum SSEMethod {
    Euler,
    NormalizedEuler,
    Milsten,
    Order2ExplicitWeak,
    LocalizedSolver,
}

#[pyclass]
struct SimulationConfig {
    n: usize,
    step: usize,
    dt: f64,
    n_trajectories: usize,
    method: SSEMethod,
}

#[pymethods]
impl SimulationConfig {
    #[new]
    #[pyo3(signature = (*, n, step, dt, n_trajectories=1,method))]
    fn new(n: usize, step: usize, dt: f64, n_trajectories: usize, method: &str) -> Self {
        let method_enum = match method {
            "Euler" => SSEMethod::Euler,
            "NormalizedEuler" => SSEMethod::NormalizedEuler,
            "Milsten" => SSEMethod::Milsten,
            "Order2ExplicitWeak" => SSEMethod::Order2ExplicitWeak,
            "LocalizedSolver" => SSEMethod::LocalizedSolver,
            _ => panic!(),
        };
        SimulationConfig {
            n,
            step,
            dt,
            n_trajectories,
            method: method_enum,
        }
    }
}

impl SimulationConfig {
    fn simulate_single_system<T: SDESystem>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
    ) -> Array2<Complex<f64>> {
        match self.method {
            SSEMethod::Euler => {
                EulerSolver::solve(initial_state, system, self.n, self.step, self.dt)
            }
            SSEMethod::NormalizedEuler => {
                NormalizedEulerSolver::solve(initial_state, system, self.n, self.step, self.dt)
            }
            SSEMethod::Milsten => {
                MilstenSolver::solve(initial_state, system, self.n, self.step, self.dt)
            }
            SSEMethod::Order2ExplicitWeak => {
                Order2ExplicitWeakSolver::solve(initial_state, system, self.n, self.step, self.dt)
            }
            SSEMethod::LocalizedSolver => {
                LocalizedSolver::solve(initial_state, system, self.n, self.step, self.dt)
            }
        }
    }

    fn simulate_system<T: SDESystem + std::marker::Sync>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
    ) -> Vec<Complex<f64>> {
        thread::scope(move |s| {
            let threads = (0..self.n_trajectories)
                .map(|_| s.spawn(move || self.simulate_single_system(initial_state, system)))
                .collect::<Vec<_>>();

            threads
                .into_iter()
                .flat_map(|t| t.join().unwrap().into_iter())
                .collect::<Vec<_>>()
        })
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: Vec<Vec<Complex<f64>>>,
    operators: Vec<Vec<Vec<Complex<f64>>>>,
    config: PyRef<SimulationConfig>,
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
    Ok(config.simulate_system(&initial_state, &system))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_banded(
    initial_state: Vec<Complex<f64>>,
    hamiltonian_diagonal: Vec<Vec<Complex<f64>>>,
    hamiltonian_offset: Vec<usize>,
    operators_diagonals: Vec<Vec<Vec<Complex<f64>>>>,
    operators_offsets: Vec<Vec<usize>>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    if operators_diagonals.len() != operators_offsets.len() {
        return Err(PyAssertionError::new_err("Bad Operators"));
    }
    if operators_diagonals[0].len() != operators_offsets[0].len() {
        return Err(PyAssertionError::new_err(
            "Number of offsets does not match number of diagonals",
        ));
    }
    if hamiltonian_diagonal.len() != hamiltonian_offset.len() {
        return Err(PyAssertionError::new_err(
            "Number of offsets does not match number of diagonals",
        ));
    }
    if operators_diagonals[0][0].len() != hamiltonian_diagonal[0].len()
        || hamiltonian_diagonal[0].len() != initial_state.len()
    {
        return Err(PyAssertionError::new_err(
            "Bad Hamiltonian or operator size",
        ));
    }

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

    Ok(config.simulate_system(&initial_state, &system))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse_bra_ket(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: Vec<Complex<f64>>,
    amplitudes: Vec<Complex<f64>>,
    bra: Vec<Complex<f64>>,
    ket: Vec<Complex<f64>>,
    config: PyRef<SimulationConfig>,
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
    Ok(config.simulate_system(&initial_state, &system))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_bra_ket, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_banded, m)?)?;
    m.add_class::<SimulationConfig>()?;
    Ok(())
}
