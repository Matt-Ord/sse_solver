use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use crate::system::SDESystem;

mod order_1;
pub use order_1::{EulerStepper, MilstenStepper};

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

#[derive(Default)]
pub struct NormalizedStepper<S>(pub S);

impl<S: Stepper> Stepper for NormalizedStepper<S> {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        let (step, err) = self.0.step(state, system, t, dt);
        let mut next = state + step;
        // Normalize the state
        next /= Complex {
            re: next.norm_l2(),
            im: 0f64,
        };
        (next - state, err)
    }
}
