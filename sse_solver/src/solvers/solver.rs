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

pub struct NormalizedStateMeasurement {}

impl Measurement for NormalizedStateMeasurement {
    type Out = Array1<Complex<f64>>;
    fn measure(&self, state: &Array1<Complex<f64>>) -> Self::Out {
        state / state.norm_l2()
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
    ) -> (Array1<Complex<f64>>, Option<f64>);
}

pub trait Solver {
    #[allow(clippy::cast_precision_loss)]
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        times: &[f64],
    ) -> Vec<M::Out>;
}

pub struct FixedStep<S> {
    pub stepper: S,
    pub target_dt: f64,
}

impl<S: Stepper> FixedStep<S> {
    fn integrate<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        current_t: &mut f64,
        end_t: f64,
    ) -> Array1<Complex<f64>> {
        let delta_t = end_t - *current_t;
        let n_substeps = (delta_t / self.target_dt).ceil();
        let dt = delta_t / n_substeps;
        #[allow(clippy::cast_possible_truncation)]
        let n_substeps = n_substeps as i64;
        let mut out = state.clone();

        if delta_t < 1e-3 * self.target_dt {
            return out;
        }

        for _n in 0..n_substeps {
            out += &self.stepper.step(&out, system, *current_t, dt).0;
            *current_t += dt;
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
        times: &[f64],
    ) -> Vec<M::Out> {
        let mut out = Vec::with_capacity(times.len());
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;
        for t in times {
            current = self.integrate(&current, system, &mut current_t, *t);
            out.push(measurement.measure(&current));
        }

        out
    }
}

pub struct DynamicErrorStepSolver<S> {
    pub stepper: S,
    pub min_error: Option<f64>,
    pub max_error: Option<f64>,
    pub target_error: f64,
    pub dt_guess: f64,
    pub n_average: usize,
}

fn get_nth_unstable_by<T: Clone, F>(v: &[T], n: usize, f: F) -> T
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let mut owned = v.to_owned();
    let (_, n_th, _) = owned.select_nth_unstable_by(n, f);
    n_th.clone()
}

fn get_median(previous_dt: &[f64]) -> f64 {
    // We should use .div_floor when `int_roundings`` is stabilized
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    get_nth_unstable_by(
        previous_dt,
        (previous_dt.len() as f64 / 2.0).floor() as usize,
        |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    )
}

impl<S: Stepper> DynamicErrorStepSolver<S> {
    fn get_initial_dt(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &impl SDESystem,
        current_t: f64,
    ) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let step_dt_guess = self.dt_guess;

        let (step, error) = self
            .stepper
            .step(initial_state, system, current_t, step_dt_guess);
        let current_delta = error.unwrap_or(step.norm_l2());

        step_dt_guess * self.target_error / current_delta
    }
}

impl<S: Stepper> Solver for DynamicErrorStepSolver<S> {
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        times: &[f64],
    ) -> Vec<M::Out> {
        let mut out = Vec::with_capacity(times.len());
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;

        let initial_dt = self.get_initial_dt(initial_state, system, current_t);
        let mut previous_dt = vec![initial_dt; self.n_average];
        let mut step_idx = 0;

        for t in times {
            let mut res_dt = t - current_t;
            let step_dt = get_median(&previous_dt);
            while res_dt > step_dt {
                let (step, error) = self.stepper.step(&current, system, current_t, step_dt);

                let current_delta = error.unwrap_or(step.norm_l2());
                if (self.min_error.is_none_or(|d| d < current_delta))
                    && self.max_error.is_none_or(|d| current_delta < d)
                {
                    current += &step;
                    current_t += step_dt;
                    res_dt -= step_dt;
                    step_idx += 1;
                }

                // Adjust the step size - note we are conservative about increasing the step size
                // immediately to the target delta, as the increments are stochastic
                let len = previous_dt.len();
                previous_dt[step_idx % len] = step_dt * self.target_error / current_delta;
            }
            // Behaves poorly for really small time steps
            // If we simply ignore a small step it will have no effect on the result
            if res_dt > step_dt * 1e-3 {
                current += &self.stepper.step(&current, system, current_t, res_dt).0;
                current_t += res_dt;
            }

            out.push(measurement.measure(&current));
        }

        out
    }
}
