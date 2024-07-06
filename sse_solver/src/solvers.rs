use core::f64;
use std::collections::HashMap;

use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Eigh, Norm, UPLO};
use num_complex::Complex;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

use crate::{
    distribution::{StandardComplexNormal, VMatrix},
    system::{SDEStep, SDESystem},
};

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

#[derive(Default)]
pub struct EulerSolver {}

impl Solver for EulerSolver {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        // The basic euler method
        // Y_n+1 = Y_n + a dt + \sum_k b_k dW
        // where dW are normalized gaussian random variables,  <dW_k* dW_k'> = dt

        let rng = rand::thread_rng();
        let sqt_dt = dt.sqrt();
        let step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: rng
                .sample_iter::<Complex<_>, _>(StandardComplexNormal)
                .map(|d| d * sqt_dt)
                .take(system.n_incoherent())
                .collect(),
        };

        state + system.get_step(&step, state, t)
    }
}

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
    let prob_eigen_map: Vec<(_, f64)> = eigenstates
        .axis_iter(Axis(0))
        .zip(probabilities.into_iter())
        .map(|b| b)
        .collect();
    let mut rng = rand::thread_rng();
    let mut transformation = prob_eigen_map
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
pub struct LocalizedSolver<S: Solver> {
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

#[derive(Default)]
pub struct NormalizedEulerSolver(EulerSolver);

impl Solver for NormalizedEulerSolver {
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

pub struct MilstenSolver {}

impl Solver for MilstenSolver {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        // The explicit milsten scheme for commuting noise
        // Y_k(n+1) = Y_k(n) + \underline{a}_k dt + \frac{1}{2} \sum_j (b^j(t, \bar{Y}(n))_k + b^j(t, Y(n))_k)dW^j
        // where dW are normalized gaussian random variables,  <dW_k* dW_k'> = dt

        // Pre-compute the system parts, since we use them twice (for supporting value and actual step)
        let parts = system.get_parts(state, t);

        let rng = rand::thread_rng();
        let sqrt_dt = dt.sqrt();

        let noise = rng
            .sample_iter::<Complex<_>, _>(StandardComplexNormal)
            .map(|d| d * sqrt_dt)
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let mut out = state.to_owned();
        // The non-supported part of the step
        // Y_k(n+1) = Y_k(n) + a dt + \sum_j \frac{1}{2}  b^j(t, Y(n))_k dW^j - sqrt(dt)(b^j(t, Y(n))_k)
        let simple_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: noise.iter().map(|d| 0.5f64 * (d + sqrt_dt)).collect(),
        };
        out += &T::get_step_from_parts(&parts, &simple_step);

        // Parts for the second supporting value required to calculate b'b term
        // according to eq 11.1.4 in <https://doi.org/10.1007/978-3-662-12616-5>
        // \bar{Y}(n) = Y(n) + a dt + b sqrt(dt)
        // as suggested in the above book, we drop the \underline{a}_k term
        // \bar{Y}(n) = Y(n) + \sum_j b^j dW^j
        let second_supporting_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: (0..system.n_incoherent())
                .map(|_| Complex {
                    re: sqrt_dt,
                    im: 0f64,
                })
                .collect(),
        };
        let second_supporting_state =
            state + T::get_step_from_parts(&parts, &second_supporting_step);
        // Add in the contribution to bb' from this supporting state (1/sqrt(dt) b(\bar{Y}))
        out += &system.get_incoherent_steps(
            &(0..system.n_incoherent())
                .map(|_| Complex {
                    re: -0.5f64 * sqrt_dt,
                    im: 0f64,
                })
                .collect::<Vec<_>>(),
            &second_supporting_state,
            t,
        );

        // Parts for the first supporting value <https://doi.org/10.1007/978-3-662-12616-5>
        // \bar{Y}(n) = Y(n) + \underline{a}_k dt  + \sum_j b^j dW^j
        // as suggested in the above book, eqn 11.1.14, we drop the \underline{a}_k term
        // \bar{Y}(n) = Y(n) + \sum_j b^j dW^j
        let first_supporting_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: noise.iter().map(|d| (d + 0.5f64 * sqrt_dt)).collect(),
        };
        let mut first_supporting_state =
            state + T::get_step_from_parts(&parts, &first_supporting_step);
        first_supporting_state += &system.get_incoherent_steps(
            &(0..system.n_incoherent())
                .map(|_| Complex {
                    re: -0.5f64 * sqrt_dt,
                    im: 0f64,
                })
                .collect::<Vec<_>>(),
            &second_supporting_state,
            t,
        );

        // Add in the parts from the first supporting state \bar{Y}(n)
        // \frac{1}{2} \sum_j (b^j(t, \bar{Y}(n))_k)dW^j
        out += &system.get_incoherent_steps(
            &noise.iter().map(|d| 0.5f64 * d).collect::<Vec<_>>(),
            &first_supporting_state,
            t,
        );

        out
    }
}

/// See 15.1.3, although there is a typo if one compares to 15.4.13
pub struct Order2ExplicitWeakSolver {}

impl Solver for Order2ExplicitWeakSolver {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let sqrt_dt = dt.sqrt();

        let mut rng = rand::thread_rng();
        let v = &rng.sample(VMatrix {
            dt,
            n: system.n_incoherent(),
        });

        let noise = rng
            .sample_iter::<Complex<_>, _>(StandardComplexNormal)
            .map(|d| d * sqrt_dt)
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let parts = system.get_parts(state, t);

        // Calculate all supporting states

        let y_supporting_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: noise,
        };

        let y_supporting_state = state + T::get_step_from_parts(&parts, &y_supporting_step);
        let noise = y_supporting_step.incoherent;

        // Calculate the final state

        let mut out = state.to_owned();
        // 1/2 dt a(\bar{Y})
        out += &system.get_coherent_step(
            Complex {
                re: 0.5f64 * dt,
                im: 0f64,
            },
            &y_supporting_state,
            t,
        );
        // 1/2 dt a(Y) + 1/2 \sum_j b^j dw^j (2 - N_incoherent)
        out += &T::get_step_from_parts(
            &parts,
            &SDEStep {
                coherent: Complex {
                    re: 0.5f64 * dt,
                    im: 0f64,
                },
                #[allow(clippy::cast_precision_loss)]
                incoherent: noise
                    .iter()
                    .map(|dw| dw * (0.5 - (0.5 * (system.n_incoherent() as f64 - 1.0) / sqrt_dt)))
                    .collect(),
            },
        );

        let operators = system.operators_from_parts(&parts);

        let u_plus_supporting_states = operators
            .incoherent
            .iter()
            .map(|incoherent| state + (incoherent * sqrt_dt))
            .collect::<Vec<_>>();

        // 1/4 \sum_j \sum_r b^j(Ur+) dw^j + (dw^j dw^r + vrj) / sqrt(dt)
        for (j, dwj) in noise.iter().enumerate() {
            for (r, (u_plus_supporting_state, dwr)) in
                u_plus_supporting_states.iter().zip(&noise).enumerate()
            {
                if j == r {
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj + (((dwj * dwj) + v[[r, j]]) / sqrt_dt)),
                        &((&operators.coherent * dt) + u_plus_supporting_state),
                        t,
                    );
                } else {
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj + ((dwj * dwr) + v[[r, j]])) / sqrt_dt,
                        u_plus_supporting_state,
                        t,
                    );
                }
            }
        }

        let u_minus_supporting_states = operators
            .incoherent
            .iter()
            .map(|incoherent| state - (incoherent * sqrt_dt))
            .collect::<Vec<_>>();

        // 1/4 \sum_j \sum_r b^j(Ur-) dw^j - (dw^j dw^r + vrj) / sqrt(dt)
        for (j, dwj) in noise.iter().enumerate() {
            for (r, (u_minus_supporting_state, dwr)) in
                u_minus_supporting_states.iter().zip(&noise).enumerate()
            {
                if j == r {
                    // R supporting value terms
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj - (((dwj * dwj) + v[[r, j]]) / sqrt_dt)),
                        &((&operators.coherent * dt) + u_minus_supporting_state),
                        t,
                    );
                } else {
                    // U supporting value terms
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj - ((dwj * dwr) + v[[r, j]])) / sqrt_dt,
                        u_minus_supporting_state,
                        t,
                    );
                }
            }
        }

        out
    }
}

/// See 15.4.13
pub struct Order2ImplicitWeakSolver {}

impl Solver for Order2ImplicitWeakSolver {
    #[allow(clippy::too_many_lines)]
    fn step<T: SDESystem>(
        &self,
        _state: &Array1<Complex<f64>>,
        _system: &T,
        _t: f64,
        _dt: f64,
    ) -> Array1<Complex<f64>> {
        todo!()
    }
}
