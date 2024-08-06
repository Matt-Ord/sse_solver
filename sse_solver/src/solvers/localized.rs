use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use num_complex::Complex;

use ndarray::{s, Axis};
use ndarray_linalg::{Eigh, UPLO};
use rand::seq::SliceRandom;

use crate::system::SDESystem;

use super::Solver;

fn calculate_inner_products(states: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let n = states.nrows();
    let mut result = Array2::<Complex<f64>>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let row_i = states.slice(s![i, ..]);
            let row_j = states.slice(s![j, ..]);
            let dot_product = row_i.dot(&row_j);
            result[[i, j]] = dot_product;
        }
    }
    result
}

fn get_weighted_state_vector(
    state_list_data: &Array2<Complex<f64>>,
    weights: &Array1<Complex<f64>>,
) -> Array1<Complex<f64>> {
    state_list_data
        .rows()
        .into_iter()
        .zip(weights)
        .map(|(v, w)| &v * *w)
        .reduce(|a, b| a + b)
        .unwrap()
}

fn select_random_localized_state(states: &Array2<Complex<f64>>) -> Array1<Complex<f64>> {
    let mut op = calculate_inner_products(states);
    op /= Complex {
        re: op.norm(),
        im: 0_f64,
    };

    let (probabilities, eigenstates) = op.eigh(UPLO::Lower).unwrap();

    let mut rng = rand::thread_rng();
    let mut transformation = eigenstates
        .axis_iter(Axis(0))
        .zip(probabilities)
        .collect::<Vec<(_, f64)>>()
        .choose_weighted(&mut rng, |item| item.1)
        .unwrap()
        .0
        .to_owned();

    transformation /= Complex {
        re: 2_f64.sqrt(),
        im: 0_f64,
    };
    get_weighted_state_vector(states, &transformation)
}

#[allow(clippy::module_name_repetitions)]
pub struct LocalizedSolver<S> {
    pub solver: S,
    pub n_realizations: usize,
}

impl<S: Solver> Solver for LocalizedSolver<S> {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut states = Array2::zeros((self.n_realizations, state.len()));
        for i in 0..self.n_realizations {
            states
                .slice_mut(s![i, ..])
                .assign(&(state + &self.solver.step(state, system, t, dt)));
        }
        select_random_localized_state(&states)
    }
}
