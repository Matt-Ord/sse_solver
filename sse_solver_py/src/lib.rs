use std::thread;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use sse_solver::solvers::{Measurement, OperatorMeasurement, StateMeasurement};
use sse_solver::sparse::PlannedSplitScatteringArray;
use sse_solver::{
    solvers::{
        EulerSolver, MilstenSolver, NormalizedSolver, Order2ExplicitWeakSolverRedux, Solver,
    },
    sparse::{BandedArray, SplitScatteringArray},
    system::sse::{FullNoise, SSESystem},
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
    NormalizedOrder2ExplicitWeak,
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
            "NormalizedOrder2ExplicitWeak" => SSEMethod::NormalizedOrder2ExplicitWeak,
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
            SSEMethod::NormalizedOrder2ExplicitWeak => "NormalizedOrder2ExplicitWeak".to_owned(),
        }
    }
}

impl SimulationConfig {
    fn simulate_single_system<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
    ) -> Vec<M::Out> {
        match (self.n_realizations, self.method) {
            (1, SSEMethod::Euler) => EulerSolver::default().solve(
                initial_state,
                system,
                measurement,
                self.n,
                self.step,
                self.dt,
            ),
            (n_realizations, SSEMethod::Euler) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: EulerSolver::default(),
                    n_realizations,
                }
                .solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                );
                panic!()
            }
            (1, SSEMethod::NormalizedEuler) => NormalizedSolver(EulerSolver::default()).solve(
                initial_state,
                system,
                measurement,
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
                .solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                );
                panic!()
            }
            (1, SSEMethod::Milsten) => MilstenSolver {}.solve(
                initial_state,
                system,
                measurement,
                self.n,
                self.step,
                self.dt,
            ),
            (n_realizations, SSEMethod::Milsten) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: MilstenSolver {},
                    n_realizations,
                }
                .solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                );
                panic!()
            }
            (1, SSEMethod::Order2ExplicitWeak) => Order2ExplicitWeakSolverRedux {}.solve(
                initial_state,
                system,
                measurement,
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
                .solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                );
                panic!()
            }
            (1, SSEMethod::NormalizedOrder2ExplicitWeak) => {
                NormalizedSolver(Order2ExplicitWeakSolverRedux {}).solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                )
            }
            (n_realizations, SSEMethod::NormalizedOrder2ExplicitWeak) => {
                #[cfg(feature = "localized")]
                return LocalizedSolver {
                    solver: NormalizedSolver(Order2ExplicitWeakSolverRedux {}),
                    n_realizations,
                }
                .solve(
                    initial_state,
                    system,
                    measurement,
                    self.n,
                    self.step,
                    self.dt,
                );
                panic!()
            }
        }
    }

    fn simulate_system<T: SDESystem + Sync, M: Measurement<Out: Send> + Sync>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
    ) -> Vec<M::Out> {
        thread::scope(move |s| {
            let threads = (0..self.n_trajectories)
                .map(|_| {
                    s.spawn(move || self.simulate_single_system(initial_state, system, measurement))
                })
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
    noise_operators: Vec<Vec<Vec<Complex<f64>>>>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    if initial_state.len() != hamiltonian.len() || hamiltonian[1].len() != hamiltonian.len() {
        return Err(PyAssertionError::new_err("Hamiltonian has bad shape"));
    }
    if initial_state.len() != noise_operators[1].len()
        || noise_operators[1].len() != noise_operators[2].len()
    {
        return Err(PyAssertionError::new_err("Hamiltonian has bad shape"));
    }
    let noise = FullNoise::from_operators(
        &Array3::from_shape_vec(
            (
                noise_operators.len(),
                initial_state.len(),
                initial_state.len(),
            ),
            noise_operators.into_iter().flatten().flatten().collect(),
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
    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}

#[pyclass]
#[derive(Clone)]
struct BandedData {
    #[pyo3(get)]
    diagonals: Vec<Vec<Complex<f64>>>,
    #[pyo3(get)]
    offsets: Vec<usize>,
    #[pyo3(get)]
    shape: [usize; 2],
}

#[pymethods]
impl BandedData {
    #[new]
    fn new(diagonals: Vec<Vec<Complex<f64>>>, offsets: Vec<usize>, shape: [usize; 2]) -> Self {
        assert!(diagonals.len() == offsets.len());
        BandedData {
            diagonals,
            offsets,
            shape,
        }
    }
}

impl From<&BandedData> for BandedArray<Complex<f64>> {
    fn from(value: &BandedData) -> Self {
        BandedArray::from_sparse(&value.diagonals, &value.offsets, &value.shape)
    }
}

#[pyfunction]
fn solve_sse_banded(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: &BandedData,
    noise_operators: Vec<BandedData>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let shape = [initial_state.len(), initial_state.len()];
    assert!(hamiltonian.shape == shape);
    noise_operators
        .iter()
        .for_each(|op| assert!(op.shape == shape));

    let noise = FullNoise::from_banded(
        &noise_operators
            .iter()
            .map(BandedArray::from)
            .collect::<Vec<_>>(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: BandedArray::from(hamiltonian),
    };

    let initial_state = Array1::from(initial_state);

    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}
#[pyclass]
#[derive(Clone)]
struct SplitOperatorData {
    #[pyo3(get)]
    a: Option<Vec<Complex<f64>>>,
    #[pyo3(get)]
    b: Option<Vec<Complex<f64>>>,
    #[pyo3(get)]
    c: Vec<Complex<f64>>,
    #[pyo3(get)]
    d: Option<Vec<Complex<f64>>>,
}

#[pymethods]
impl SplitOperatorData {
    #[new]
    #[pyo3(signature = (*, a, b, c, d))]
    fn new(
        a: Option<Vec<Complex<f64>>>,
        b: Option<Vec<Complex<f64>>>,
        c: Vec<Complex<f64>>,
        d: Option<Vec<Complex<f64>>>,
    ) -> Self {
        SplitOperatorData { a, b, c, d }
    }
}

impl From<SplitOperatorData> for SplitScatteringArray<Complex<f64>> {
    fn from(value: SplitOperatorData) -> Self {
        SplitScatteringArray::from_parts(
            value.a.map(|d| d.into()),
            value.b.map(|d| d.into()),
            value.c.into(),
            value.d.map(|d| d.into()),
        )
    }
}

fn solve_sse_split_operator_for_measurement<
    M: Sync + Measurement<Out: Send + IntoIterator<Item = Complex<f64>>>,
>(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: &SplitOperatorData,
    noise_operators: Vec<SplitOperatorData>,
    config: PyRef<SimulationConfig>,
    measurement: &M,
) -> PyResult<Vec<Complex<f64>>> {
    assert!(hamiltonian.c.len() == initial_state.len());
    noise_operators
        .iter()
        .for_each(|op| assert!(op.c.len() == initial_state.len()));

    let noise = FullNoise::from_split(
        &noise_operators
            .into_iter()
            .map(SplitScatteringArray::from)
            .collect::<Vec<_>>(),
    );
    let system = SSESystem {
        noise,
        hamiltonian: SplitScatteringArray::from(hamiltonian.clone()),
    };

    let initial_state = Array1::from(initial_state);

    Ok(config
        .simulate_system(&initial_state, &system, measurement)
        .into_iter()
        .flat_map(|d| d.into_iter())
        .collect())
}

#[pyfunction]
fn solve_sse_split_operator(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: &SplitOperatorData,
    noise_operators: Vec<SplitOperatorData>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    solve_sse_split_operator_for_measurement(
        initial_state,
        hamiltonian,
        noise_operators,
        config,
        &StateMeasurement {},
    )
}

#[pyfunction]
fn solve_sse_measured_split_operator(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: &SplitOperatorData,
    noise_operators: Vec<SplitOperatorData>,
    measurement_operators: Vec<SplitOperatorData>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    measurement_operators
        .iter()
        .for_each(|op| assert!(op.c.len() == initial_state.len()));

    let measurement = measurement_operators
        .into_iter()
        .map(SplitScatteringArray::from)
        .map(PlannedSplitScatteringArray::from)
        .map(|operator| OperatorMeasurement { operator })
        .collect::<Vec<_>>();

    solve_sse_split_operator_for_measurement(
        initial_state,
        hamiltonian,
        noise_operators,
        config,
        &measurement,
    )
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
    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_bra_ket, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_banded, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_split_operator, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sse_measured_split_operator, m)?)?;
    m.add_class::<SimulationConfig>()?;
    m.add_class::<BandedData>()?;
    m.add_class::<SplitOperatorData>()?;
    Ok(())
}
