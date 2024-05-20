use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use num_complex::Complex;
use rand::Rng;

use crate::{
    distribution::StandardComplexNormal,
    system::{SDEStep, SDESystem},
};

pub trait Solver<T: SDESystem> {
    fn step(state: &Array1<Complex<f64>>, system: &T, t: f64, dt: f64) -> Array1<Complex<f64>>;

    fn integrate(
        state: &Array1<Complex<f64>>,
        system: &T,
        t_start: f64,
        n_step: usize,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut out = state.clone();
        let mut current_t = t_start.to_owned();
        for _n in 0..n_step {
            out = Self::step(&out, system, current_t, dt);
            current_t += dt;
        }
        out
    }
    #[allow(clippy::cast_precision_loss)]
    fn solve(
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
            current = Self::integrate(&current, system, current_t, step, dt);
            current_t += dt * step as f64;
            // TODO: we maybe shouldn't be doing this ...
            current /= Complex {
                re: current.norm_l2(),
                im: 0f64,
            };
        }
        out.push_row(current.view()).unwrap();

        out
    }
}

pub struct EulerSolver {}

impl<T: SDESystem> Solver<T> for EulerSolver {
    fn step(state: &Array1<Complex<f64>>, system: &T, t: f64, dt: f64) -> Array1<Complex<f64>> {
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

        let mut out = state.to_owned();
        system.apply_step(&mut out, &step, state, t);
        out
    }
}

pub struct MilstenSolver {}

impl<T: SDESystem> Solver<T> for MilstenSolver {
    fn step(state: &Array1<Complex<f64>>, system: &T, t: f64, dt: f64) -> Array1<Complex<f64>> {
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
        // Y_k(n+1) = Y_k(n) + a dt + \sum_j \frac{1}{2}  b^j(t, Y(n))_k dW^j - 1/sqrt(dt)(b^j(t, Y(n))_k)
        let simple_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: noise.iter().map(|d| (d / 2f64) - sqrt_dt).collect(),
        };
        T::apply_step_from_parts(&mut out, &parts, &simple_step);

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
        let mut second_supporting_state = state.to_owned();
        T::apply_step_from_parts(
            &mut second_supporting_state,
            &parts,
            &second_supporting_step,
        );
        // Add in the contribution to bb' from this supporting state (1/sqrt(dt) b(\bar{Y}))
        system.apply_incoherent_step(
            &mut out,
            &(0..system.n_incoherent())
                .map(|_| Complex {
                    re: sqrt_dt,
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
        let mut first_supporting_state = state.to_owned();
        T::apply_incoherent_step_from_parts(&mut first_supporting_state, &parts.into(), &noise);

        // Add in the parts from the first supporting state \bar{Y}(n)
        // \frac{1}{2} \sum_j (b^j(t, \bar{Y}(n))_k)dW^j
        system.apply_incoherent_step(
            &mut out,
            &simple_step.incoherent,
            &first_supporting_state,
            t,
        );

        out
    }
}

pub struct Order2WeakSolver {}

impl<T: SDESystem> Solver<T> for Order2WeakSolver {
    fn step(_state: &Array1<Complex<f64>>, _system: &T, _t: f64, _dt: f64) -> Array1<Complex<f64>> {
        todo!()
    }
}
