use std::thread;

use ndarray::{Array1, Array2, Array3, array};
use num_complex::Complex;
use pyo3::IntoPyObject;
use pyo3::types::PyList;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use sse_solver::solvers::{
    DynamicErrorStepSolver, FixedStepSolver, Measurement, OperatorMeasurement, Solver,
    StateMeasurement, Stepper,
};
use sse_solver::sparse::PlannedSplitScatteringArray;
use sse_solver::system::harmonic_langevin::{
    HarmonicLangevinParameters, PeriodicLangevinParameters, get_langevin_system,
    get_stable_quantum_langevin_system,
};
use sse_solver::system::simple_stochastic::{SimpleStochasticFn, SimpleStochasticSDESystem};
use sse_solver::{
    solvers::{
        EulerStepper, MilsteinStepper, NormalizedStepper, Order2ExplicitWeakR5Stepper,
        Order2ExplicitWeakStepper,
    },
    sparse::{BandedArray, SplitScatteringArray},
    system::SDESystem,
    system::sse::{FullNoise, SSESystem},
};

#[cfg(feature = "localized")]
use sse_solver::solvers::LocalizedStepper;

#[derive(Clone, Copy, Hash)]
enum SSEMethod {
    Euler,
    NormalizedEuler,
    Milstein,
    NormalizedMilstein,
    Order2ExplicitWeak,
    NormalizedOrder2ExplicitWeak,
    Order2ExplicitWeakR5,
    NormalizedOrder2ExplicitWeakR5,
}

#[pyclass]
struct SimulationConfig {
    #[pyo3(get, set)]
    times: Vec<f64>,
    #[pyo3(get, set)]
    dt: f64,
    #[pyo3(get, set)]
    delta: Option<(Option<f64>, f64, Option<f64>)>,
    #[pyo3(get, set)]
    n_trajectories: usize,
    #[pyo3(get, set)]
    n_realizations: usize,
    method: SSEMethod,
}

#[pymethods]
impl SimulationConfig {
    #[new]
    #[pyo3(signature = (*, times, dt, delta=None, n_trajectories=1, method, n_realizations=1))]
    fn new(
        times: Vec<f64>,
        dt: f64,
        delta: Option<(Option<f64>, f64, Option<f64>)>,
        n_trajectories: usize,
        method: &str,
        n_realizations: usize,
    ) -> Self {
        let method_enum = match method {
            "Euler" => SSEMethod::Euler,
            "NormalizedEuler" => SSEMethod::NormalizedEuler,
            "Milstein" => SSEMethod::Milstein,
            "NormalizedMilstein" => SSEMethod::NormalizedMilstein,
            "Order2ExplicitWeak" => SSEMethod::Order2ExplicitWeak,
            "NormalizedOrder2ExplicitWeak" => SSEMethod::NormalizedOrder2ExplicitWeak,
            "Order2ExplicitWeakR5" => SSEMethod::Order2ExplicitWeakR5,
            "NormalizedOrder2ExplicitWeakR5" => SSEMethod::NormalizedOrder2ExplicitWeakR5,
            _ => panic!("Invalid method"),
        };
        SimulationConfig {
            times,
            dt,
            delta,
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
            SSEMethod::Milstein => "Milstein".to_owned(),
            SSEMethod::NormalizedMilstein => "NormalizedMilstein".to_owned(),
            SSEMethod::Order2ExplicitWeak => "Order2ExplicitWeak".to_owned(),
            SSEMethod::NormalizedOrder2ExplicitWeak => "NormalizedOrder2ExplicitWeak".to_owned(),
            SSEMethod::Order2ExplicitWeakR5 => "Order2ExplicitWeakR5".to_owned(),
            SSEMethod::NormalizedOrder2ExplicitWeakR5 => {
                "NormalizedOrder2ExplicitWeakR5".to_owned()
            }
        }
    }
}

enum DynStepper {
    Euler(EulerStepper),
    NormalizedEuler(NormalizedStepper<EulerStepper>),
    Milstein(MilsteinStepper),
    NormalizedMilstein(NormalizedStepper<MilsteinStepper>),
    Order2ExplicitWeak(Order2ExplicitWeakStepper),
    NormalizedOrder2ExplicitWeak(NormalizedStepper<Order2ExplicitWeakStepper>),
    Order2ExplicitWeakR5(Order2ExplicitWeakR5Stepper),
    NormalizedOrder2ExplicitWeakR5(NormalizedStepper<Order2ExplicitWeakR5Stepper>),
}

impl Stepper for DynStepper {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        match self {
            DynStepper::Euler(s) => s.step(state, system, t, dt),
            DynStepper::NormalizedEuler(s) => s.step(state, system, t, dt),
            DynStepper::Milstein(s) => s.step(state, system, t, dt),
            DynStepper::NormalizedMilstein(s) => s.step(state, system, t, dt),
            DynStepper::Order2ExplicitWeak(s) => s.step(state, system, t, dt),
            DynStepper::NormalizedOrder2ExplicitWeak(s) => s.step(state, system, t, dt),
            DynStepper::Order2ExplicitWeakR5(s) => s.step(state, system, t, dt),
            DynStepper::NormalizedOrder2ExplicitWeakR5(s) => s.step(state, system, t, dt),
        }
    }
}

impl SimulationConfig {
    fn get_stepper(&self) -> DynStepper {
        match self.method {
            SSEMethod::Euler => DynStepper::Euler(EulerStepper::default()),
            SSEMethod::NormalizedEuler => DynStepper::NormalizedEuler(NormalizedStepper {
                inner: EulerStepper::default(),
                calculate_error: true,
            }),
            SSEMethod::Milstein => DynStepper::Milstein(MilsteinStepper {}),
            SSEMethod::NormalizedMilstein => DynStepper::NormalizedMilstein(NormalizedStepper {
                inner: MilsteinStepper {},
                calculate_error: true,
            }),
            SSEMethod::Order2ExplicitWeak => {
                DynStepper::Order2ExplicitWeak(Order2ExplicitWeakStepper {
                    error_measure: None,
                })
            }
            SSEMethod::NormalizedOrder2ExplicitWeak => {
                DynStepper::NormalizedOrder2ExplicitWeak(NormalizedStepper {
                    inner: Order2ExplicitWeakStepper {
                        error_measure: None,
                    },
                    calculate_error: true,
                })
            }
            SSEMethod::Order2ExplicitWeakR5 => {
                DynStepper::Order2ExplicitWeakR5(Order2ExplicitWeakR5Stepper {
                    error_measure: None,
                })
            }
            SSEMethod::NormalizedOrder2ExplicitWeakR5 => {
                DynStepper::NormalizedOrder2ExplicitWeakR5(NormalizedStepper {
                    inner: Order2ExplicitWeakR5Stepper {
                        error_measure: None,
                    },
                    calculate_error: true,
                })
            }
        }
    }
    fn simulate_single_system<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
    ) -> Vec<M::Out> {
        let stepper = self.get_stepper();
        if self.n_realizations == 1 {
            if let Some(delta) = self.delta {
                return DynamicErrorStepSolver {
                    max_error: delta.2,
                    target_error: delta.1,
                    min_error: delta.0,
                    stepper,
                    dt_guess: self.dt,
                    n_average: 11,
                }
                .solve(initial_state, system, measurement, &self.times);
            }
            FixedStepSolver {
                stepper,
                target_dt: self.dt,
            }
            .solve(initial_state, system, measurement, &self.times)
        } else {
            #[cfg(feature = "localized")]
            return LocalizedStepper {
                solver,
                n_realizations: self.n_realizations,
            }
            .solve(
                initial_state,
                system,
                measurement,
                self.n,
                self.dt * self.step as f64,
            );
            panic!()
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

fn warn_about_complex(py: Python<'_>) -> PyResult<()> {
    // Attempt to import the 'warnings' module
    let warnings = PyModule::import(py, "warnings")?;

    // The message you want to display
    let message = "This function has recently been changed to use real noise and is probably not working as intended with complex noise.";

    // Call warnings.warn(message)
    warnings.call_method1("warn", (message,))?;

    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_sse(
    initial_state: Vec<Complex<f64>>,
    hamiltonian: Vec<Vec<Complex<f64>>>,
    noise_operators: Vec<Vec<Vec<Complex<f64>>>>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    warn_about_complex(config.py())?;
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
    warn_about_complex(config.py())?;
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
    warn_about_complex(config.py())?;
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
    warn_about_complex(config.py())?;
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
/// Solve a simple stochastic system defined by coherent and incoherent functions
fn solve_simple_stochastic(
    initial_state: Vec<Complex<f64>>,
    coherent: Py<PyAny>,
    incoherent: Vec<Py<PyAny>>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let system = SimpleStochasticSDESystem {
        coherent: Box::new(move |t: f64, state: &Array1<Complex<f64>>| {
            Python::attach(|py| {
                let args = (
                    t.into_pyobject(py).unwrap(),
                    PyList::new(py, state.iter()).unwrap(),
                );
                let result = coherent.call1(py, args).unwrap();
                let array: Vec<Complex<f64>> = result.extract(py).unwrap();
                Array1::from(array)
            })
        }),
        incoherent: incoherent
            .into_iter()
            .map(|f| {
                Box::new(move |t: f64, state: &Array1<Complex<f64>>| {
                    Python::attach(|py| {
                        let args = (
                            t.into_pyobject(py).unwrap(),
                            PyList::new(py, state.iter()).unwrap(),
                        );
                        let result = f.call1(py, args).unwrap();
                        let array: Vec<Complex<f64>> = result.extract(py).unwrap();
                        Array1::from(array)
                    })
                }) as Box<SimpleStochasticFn>
            })
            .collect::<Vec<_>>(),
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
struct HarmonicLangevinSystemParameters {
    #[pyo3(get, set)]
    dimensionless_mass: f64,
    #[pyo3(get, set)]
    dimensionless_omega: f64,
    #[pyo3(get, set)]
    dimensionless_lambda: f64,
    #[pyo3(get, set)]
    kbt_div_hbar: f64,
}

#[pymethods]
impl HarmonicLangevinSystemParameters {
    #[new]
    #[pyo3(signature = (*, dimensionless_mass, dimensionless_omega, dimensionless_lambda, kbt_div_hbar))]
    fn new(
        dimensionless_mass: f64,
        dimensionless_omega: f64,
        dimensionless_lambda: f64,
        kbt_div_hbar: f64,
    ) -> Self {
        HarmonicLangevinSystemParameters {
            dimensionless_mass,
            dimensionless_omega,
            dimensionless_lambda,
            kbt_div_hbar,
        }
    }
}

#[pyfunction]
fn solve_harmonic_langevin(
    initial_state: Complex<f64>,
    params: PyRef<HarmonicLangevinSystemParameters>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let system = get_langevin_system(&HarmonicLangevinParameters {
        dimensionless_mass: params.dimensionless_mass,
        dimensionless_omega: params.dimensionless_omega,
        dimensionless_lambda: params.dimensionless_lambda,
        kbt_div_hbar: params.kbt_div_hbar,
    });

    let initial_state = array![initial_state];
    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}

#[pyfunction]
fn solve_harmonic_stable_quantum_langevin(
    initial_state: (Complex<f64>, Complex<f64>),
    params: PyRef<HarmonicLangevinSystemParameters>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let system = get_stable_quantum_langevin_system(&HarmonicLangevinParameters {
        dimensionless_mass: params.dimensionless_mass,
        dimensionless_omega: params.dimensionless_omega,
        dimensionless_lambda: params.dimensionless_lambda,
        kbt_div_hbar: params.kbt_div_hbar,
    });

    let initial_state = array![initial_state.0, initial_state.1];
    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}

#[pyclass]
struct PeriodicLangevinSystemParameters {
    #[pyo3(get, set)]
    dimensionless_mass: f64,
    #[pyo3(get, set)]
    dimensionless_potential: Vec<Complex<f64>>,
    #[pyo3(get, set)]
    dk_times_lengthscale: f64,
    #[pyo3(get, set)]
    dimensionless_lambda: f64,
    #[pyo3(get, set)]
    kbt_div_hbar: f64,
}

#[pymethods]
impl PeriodicLangevinSystemParameters {
    #[new]
    #[pyo3(signature = (*, dimensionless_mass, dimensionless_potential, dk_times_lengthscale, dimensionless_lambda, kbt_div_hbar))]
    fn new(
        dimensionless_mass: f64,
        dimensionless_potential: Vec<Complex<f64>>,
        dk_times_lengthscale: f64,
        dimensionless_lambda: f64,
        kbt_div_hbar: f64,
    ) -> Self {
        PeriodicLangevinSystemParameters {
            dimensionless_mass,
            dimensionless_potential,
            dk_times_lengthscale,
            dimensionless_lambda,
            kbt_div_hbar,
        }
    }
}
#[pyfunction]
fn solve_periodic_langevin(
    initial_state: Complex<f64>,
    params: PyRef<PeriodicLangevinSystemParameters>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let system = get_langevin_system(&PeriodicLangevinParameters {
        dimensionless_mass: params.dimensionless_mass,
        dimensionless_potential: params.dimensionless_potential.clone(),
        dk_times_lengthscale: params.dk_times_lengthscale,
        dimensionless_lambda: params.dimensionless_lambda,
        kbt_div_hbar: params.kbt_div_hbar,
    });

    let initial_state = array![initial_state];
    Ok(config
        .simulate_system(&initial_state, &system, &StateMeasurement {})
        .iter()
        .flat_map(|d| d.iter())
        .cloned()
        .collect())
}

#[pyfunction]
fn solve_periodic_stable_quantum_langevin(
    initial_state: (Complex<f64>, Complex<f64>),
    params: PyRef<PeriodicLangevinSystemParameters>,
    config: PyRef<SimulationConfig>,
) -> PyResult<Vec<Complex<f64>>> {
    let system = get_stable_quantum_langevin_system(&PeriodicLangevinParameters {
        dimensionless_mass: params.dimensionless_mass,
        dimensionless_potential: params.dimensionless_potential.clone(),
        dk_times_lengthscale: params.dk_times_lengthscale,
        dimensionless_lambda: params.dimensionless_lambda,
        kbt_div_hbar: params.kbt_div_hbar,
    });

    let initial_state = array![initial_state.0, initial_state.1];
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
    m.add_function(wrap_pyfunction!(solve_simple_stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(solve_harmonic_langevin, m)?)?;
    m.add_function(wrap_pyfunction!(solve_harmonic_stable_quantum_langevin, m)?)?;
    m.add_function(wrap_pyfunction!(solve_periodic_langevin, m)?)?;
    m.add_function(wrap_pyfunction!(solve_periodic_stable_quantum_langevin, m)?)?;
    m.add_class::<SimulationConfig>()?;
    m.add_class::<BandedData>()?;
    m.add_class::<SplitOperatorData>()?;
    m.add_class::<HarmonicLangevinSystemParameters>()?;
    m.add_class::<PeriodicLangevinSystemParameters>()?;

    Ok(())
}
