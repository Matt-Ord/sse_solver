use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use crate::{sparse::Tensor, system::SDESystem};

pub trait Measurement {
    type Out;
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out;
}

pub struct StateMeasurement {}

impl Measurement for StateMeasurement {
    type Out = Array1<Complex<f64>>;
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out {
        state.clone()
    }
}

pub struct OperatorMeasurement<T> {
    pub operator: T,
}

impl<T: Tensor> Measurement for OperatorMeasurement<T> {
    type Out = Complex<f64>;
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out {
        let conj_state = state.map(num_complex::Complex::conj);
        conj_state.dot(&self.operator.dot(state))
    }
}

impl<M: Measurement> Measurement for Vec<M> {
    type Out = Vec<M::Out>;
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out {
        self.iter().map(|m| m.measure(state)).collect()
    }
}

impl<M0: Measurement, M1: Measurement> Measurement for (M0, M1) {
    type Out = (M0::Out, M1::Out);
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out {
        (self.0.measure(state), self.1.measure(state))
    }
}
pub trait Stepper {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>>;
}

pub trait Solver {
    #[allow(clippy::cast_precision_loss)]
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        n: usize,
        dt: f64,
    ) -> Vec<M::Out>;
}

pub struct FixedStep<S> {
    pub stepper: S,
    pub n_substeps: usize,
}

impl<S: Stepper> FixedStep<S> {
    fn integrate<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        current_t: &mut f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        #[allow(clippy::cast_precision_loss)]
        let step_dt = dt / self.n_substeps as f64;
        let mut out = state.clone();
        for _n in 0..self.n_substeps {
            out += &self.stepper.step(&out, system, *current_t, step_dt);
            *current_t += step_dt;
        }
        out
    }
}
impl<S: Stepper> Solver for FixedStep<S> {
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        n: usize,
        dt: f64,
    ) -> Vec<M::Out> {
        let mut out = Vec::with_capacity(n);
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;
        for _step_n in 1..n {
            out.push(measurement.measure(&current));
            current = self.integrate(&current, system, &mut current_t, dt);
        }
        out.push(measurement.measure(&current));

        out
    }
}

pub struct DynamicStep<S> {
    pub stepper: S,
    pub min_delta: Option<f64>,
    pub max_delta: Option<f64>,
    pub target_delta: f64,
    pub n_substeps_guess: usize,
}

impl<S> DynamicStep<S> {
    fn get_initial_dt(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &impl SDESystem,
        current_t: f64,
        dt: f64,
    ) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let step_dt_guess = dt / self.n_substeps_guess as f64;

        let step = system.get_coherent_step(step_dt_guess.into(), initial_state, current_t);
        let current_delta = step.norm_l2();
        step_dt_guess * self.target_delta / current_delta
    }
}
impl<S: Stepper> Solver for DynamicStep<S> {
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        n: usize,
        dt: f64,
    ) -> Vec<M::Out> {
        let mut out = Vec::with_capacity(n);
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;

        let mut step_dt = self.get_initial_dt(initial_state, system, current_t, dt);

        for _step_n in 1..n {
            out.push(measurement.measure(&current));

            let mut res_dt = dt;
            while res_dt > step_dt {
                let step = self.stepper.step(&current, system, current_t, step_dt);

                let current_delta = step.norm_l2();
                if (self.min_delta.is_none_or(|d| d < current_delta))
                    && self.max_delta.is_none_or(|d| current_delta < d)
                {
                    current += &step;
                    current_t += step_dt;
                    res_dt -= step_dt;
                }

                // Adjust the step size - note we are conservative about increasing the step size
                // immediately to the target delta, as the increments are stochastic
                let optimal_dt = step_dt * self.target_delta / current_delta;
                step_dt *= if optimal_dt > step_dt {
                    0.8 + (0.2 * optimal_dt)
                } else {
                    optimal_dt
                };
            }
            current += &self.stepper.step(&current, system, current_t, res_dt);
            current_t += res_dt;
        }
        out.push(measurement.measure(&current));

        out
    }
}
