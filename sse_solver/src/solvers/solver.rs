use ndarray::{Array1, Array2};
use num_complex::Complex;

use crate::system::SDESystem;

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
    fn solve<T: SDESystem>(
        &self,
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        n: usize,
        step: usize,
        dt: f64,
    ) -> Array2<Complex<f64>> {
        let mut out = Array2::zeros([0, initial_state.len()]);
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;
        for _step_n in 1..n {
            out.push_row(current.view()).unwrap();
            current = self.integrate(&current, system, &mut current_t, step, dt);
        }
        out.push_row(current.view()).unwrap();

        out
    }
}
