use std::thread;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use sse_solver::{
    solvers::{
        EulerSolver, MilstenSolver, NormalizedEulerSolver, Order2ExplicitWeakSolverRedux, Solver,
    },
    sparse::BandedArray,
    sse_system::{FullNoise, SSESystem},
    system::SDESystem,
};

#[cfg(feature = "localized")]
use sse_solver::solvers::LocalizedSolver;

#[derive(Clone, Copy, Hash)]
enum SSEMethod {
    Euler,
    NormalizedEuler,
    Milsten,
    Order2ExplicitWeak,
}

#[pyclass]
struct SimulationConfig {
    #[pyo3(get, set)]
    n: usize,
    #[pyo3(get, set)]
    step: usize,
    #[pyo3(get, set)]
    dt: f64,
    #[pyo3(get, set)]
    n_trajectories: usize,
    #[pyo3(get, set)]
    n_realizations: usize,
    method: SSEMethod,
}

#[pymethods]
impl SimulationConfig {
    #[new]
    #[pyo3(signature = (*, n, step, dt, n_trajectories=1,method,n_realizations=1))]
    fn new(
        n: usize,
        step: usize,
        dt: f64,
        n_trajectories: usize,
        method: &str,
        n_realizations: usize,
    ) -> Self {
        let method_enum = match method {
            "Euler" => SSEMethod::Euler,
            "NormalizedEuler" => SSEMethod::NormalizedEuler,
            "Milsten" => SSEMethod::Milsten,
            "Order2ExplicitWeak" => SSEMethod::Order2ExplicitWeak,
            _ => panic!(),
        };
        SimulationConfig {
            n,
            step,
            dt,
            n_trajectories,
            method: method_enum,
            n_realizations,
        }
    }

    #[getter]
    fn method(&self) -> String {
        match self.method {
            SSEMethod::Euler => "Euler".to_owned(),
            SSEMethod::NormalizedEuler => "NormalizedEuler".to_owned(),
            SSEMethod::Milsten => "Milsten".to_owned(),
            SSEMethod::Order2ExplicitWeak => "Order2ExplicitWeak".to_owned(),
        }
    }
}

impl SimulationConfig {
    fn simulate_single_system<T: SDESystem>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
    ) -> Array2<Complex<f64>> {
        match (self.n_realizations, self.method) {
            (1, SSEMethod::Euler) => {
                EulerSolver::default().solve(initial_state, system, self.n, self.step, self.dt)
            }
            (n_realizations, SSEMethod::Euler) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: EulerSolver::default(),
                    n_realizations,
                }
                .solve(initial_state, system, self.n, self.step, self.dt);
                panic!()
            }
            (1, SSEMethod::NormalizedEuler) => NormalizedEulerSolver::default().solve(
                initial_state,
                system,
                self.n,
                self.step,
                self.dt,
            ),
            (n_realizations, SSEMethod::NormalizedEuler) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: EulerSolver::default(),
                    n_realizations,
                }
                .solve(initial_state, system, self.n, self.step, self.dt);
                panic!()
            }
            (1, SSEMethod::Milsten) => {
                MilstenSolver {}.solve(initial_state, system, self.n, self.step, self.dt)
            }
            (n_realizations, SSEMethod::Milsten) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: MilstenSolver {},
                    n_realizations,
                }
                .solve(initial_state, system, self.n, self.step, self.dt);
                panic!()
            }
            (1, SSEMethod::Order2ExplicitWeak) => Order2ExplicitWeakSolverRedux {}.solve(
                initial_state,
                system,
                self.n,
                self.step,
                self.dt,
            ),
            (n_realizations, SSEMethod::Order2ExplicitWeak) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: Order2ExplicitWeakSolverRedux {},
                    n_realizations,
                }
                .solve(initial_state, system, self.n, self.step, self.dt);
                panic!()
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
