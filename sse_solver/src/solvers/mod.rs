use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use crate::system::SDESystem;

mod order_1;
pub use order_1::{EulerStepper, MilsteinStepper};

mod order_2;
pub use order_2::{
    ExplicitWeakR5Stepper as Order2ExplicitWeakR5Stepper,
    ExplicitWeakStepper as Order2ExplicitWeakStepper,
};

pub mod solver;
pub use solver::{
    DynamicErrorStepSolver, FixedStep as FixedStepSolver, Measurement, NormalizedStateMeasurement,
    OperatorMeasurement, Solver, StateMeasurement, Stepper,
};
#[cfg(feature = "localized")]
pub mod localized;
#[cfg(feature = "localized")]
pub use localized::*;

#[derive(Default, Debug)]
pub struct NormalizedStepper<S> {
    pub inner: S,
    pub calculate_error: bool,
}

impl<S: Stepper> Stepper for NormalizedStepper<S> {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        let (step, inner_err) = self.inner.step(state, system, t, dt);
        let next = state + step;
        let next_norm = next.norm_l2();
        // Normalize the state
        let next_normalized = &next / next_norm;

        let err = if self.calculate_error {
            Some(next_norm)
        } else {
            inner_err
        };
        (next_normalized - state, err)
    }
}
