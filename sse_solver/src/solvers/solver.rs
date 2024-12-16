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

pub trait Solver {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>>;

    fn integrate<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        current_t: &mut f64,
        n_step: usize,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut out = state.clone();
        for _n in 0..n_step {
            out = self.step(&out, system, *current_t, dt);
            *current_t += dt;
        }
        out
    }

    #[allow(clippy::cast_precision_loss)]
    fn solve<T: SDESystem, M: Measurement>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        measurement: &M,
        n: usize,
        step: usize,
        dt: f64,
    ) -> Vec<M::Out> {
        let mut out = Vec::with_capacity(n);
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;
        for _step_n in 1..n {
            out.push(measurement.measure(&current));
            current = self.integrate(&current, system, &mut current_t, step, dt);
        }
        out.push(measurement.measure(&current));

        out
    }
}
