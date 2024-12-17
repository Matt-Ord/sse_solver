use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use crate::system::SDESystem;

mod order_1;
pub use order_1::{EulerSolver, MilstenSolver};

mod order_2;
pub use order_2::{
    ExplicitWeakR5Solver as Order2ExplicitWeakR5Solver,
    ExplicitWeakSolver as Order2ExplicitWeakSolver,
};

pub mod solver;
pub use solver::*;
#[cfg(feature = "localized")]
pub mod localized;
#[cfg(feature = "localized")]
pub use localized::*;

#[derive(Default)]
pub struct NormalizedSolver<S>(pub S);

impl<S: Solver> Solver for NormalizedSolver<S> {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut out = self.0.step(state, system, t, dt);
        // Normalize the state
        out /= Complex {
            re: out.norm_l2(),
            im: 0f64,
        };
        out
    }
}
