use ndarray::Array1;
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
    fn integrate<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        current_t: &mut f64,
        dt: f64,
    ) -> Array1<Complex<f64>>;

    #[allow(clippy::cast_precision_loss)]
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

pub struct FixedStep<S> {
    pub stepper: S,
    pub n_substeps: usize,
}

impl<S: Stepper> Solver for FixedStep<S> {
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

pub struct DynamicStep<S> {
    pub stepper: S,
    pub min_delta: f64,
    pub max_delta: f64,
    pub n_substeps_guess: usize,
}

impl<S: Stepper> Solver for DynamicStep<S> {
    fn integrate<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        current_t: &mut f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        #[allow(clippy::cast_precision_loss)]
        let mut step_dt = dt / self.n_substeps_guess as f64;
        let mut out = state.clone();
        let mut res_dt = dt;
        let target_delta = 0.5 * (self.min_delta + self.max_delta);

        while res_dt > step_dt {
            let step = self.stepper.step(&out, system, *current_t, step_dt);

            let current_delta = step
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .fold(0f64, f64::max)
                .sqrt();
            if (self.min_delta < current_delta) && current_delta < self.max_delta {
                out += &step;
                *current_t += step_dt;
                res_dt -= step_dt;
            }

            step_dt *= target_delta / current_delta;
        }

        out += &self.stepper.step(&out, system, *current_t, res_dt);
        out
    }
}
