use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use rand::Rng;

use crate::{
    distribution::{StandardComplexNormal, V as VDistribution, W as WDistribution},
    system::{SDEStep, SDESystem},
};

pub mod solver;
pub use solver::*;
#[cfg(feature = "localized")]
pub mod localized;
#[cfg(feature = "localized")]
pub use localized::*;

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
        let v = &rng.sample(VDistribution {
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

        let operators = T::operators_from_parts(&parts);

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
                let pair_dw = (dwj * dwr + v[(r, j)]) / sqrt_dt;
                if j == r {
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj + pair_dw),
                        &((&operators.coherent * dt) + u_plus_supporting_state),
                        t,
                    );
                } else {
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj + pair_dw),
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
                let pair_dw = (dwj * dwr + v[(r, j)]) / sqrt_dt;
                if j == r {
                    // R supporting value terms
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj - pair_dw),
                        &((&operators.coherent * dt) + u_minus_supporting_state),
                        t,
                    );
                } else {
                    // U supporting value terms
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj - pair_dw),
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

/// The Order 2 General Weak Taylor Scheme defined in 5.1 of
/// <https://www.jstor.org/stable/27862707>
///
/// This method scales poorly for large n operators compared
/// to the other methods discussed in the paper
pub struct Order2ExplicitWeakSolverRedux {}

impl Solver for Order2ExplicitWeakSolverRedux {
    #[allow(clippy::too_many_lines)]
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut rng = rand::thread_rng();
        // Sample the V distribution
        let dv = &rng.sample(VDistribution {
            dt,
            n: system.n_incoherent(),
        });

        // Sample the W distribution (this is called I in the paper)
        let dw = &rng
            .sample_iter(WDistribution::new(dt))
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let parts = system.get_parts(state, t);

        // Y_n + a(Y_n) dt + \sum_j b^j w_j
        let h0_step = T::get_step_from_parts(
            &parts,
            &SDEStep {
                coherent: Complex { re: dt, im: 0f64 },
                incoherent: dw.iter().map(|d| Complex { re: *d, im: 0f64 }).collect(),
            },
        );

        // Build out the supporting states required for the calculation

        // H_0 = Y_n + a(Y_n) dt + b^j w_j
        let h0 = state + h0_step;

        let operators = T::operators_from_parts(&parts);

        let sqrt_dt = dt.sqrt();
        // H bar_+-^k = Y_n +- b^j \sqrt(dt)
        let h_bar_plus = operators
            .incoherent
            .iter()
            .map(|incoherent| state + (incoherent * sqrt_dt))
            .collect::<Vec<_>>();
        let h_bar_minus = operators
            .incoherent
            .iter()
            .map(|incoherent| state - (incoherent * sqrt_dt))
            .collect::<Vec<_>>();

        // H_+-^k = Y_n + a(Y_n) dt +- b^j \sqrt(dt)
        let h_plus = h_bar_plus
            .iter()
            .map(|h_bar| h_bar + (&operators.coherent * dt))
            .collect::<Vec<_>>();
        let h_minus = h_bar_minus
            .iter()
            .map(|h_bar| h_bar + (&operators.coherent * dt))
            .collect::<Vec<_>>();

        // Y_n+1 = Y_n + (a(Y_n) + a(H_0)) dt / 2 + ...
        let mut out = state
            + ((operators.coherent
                + system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h0, t))
                * (dt / 2.0));

        for k in 0..system.n_incoherent() {
            for l in 0..system.n_incoherent() {
                // Add H - type increment
                // If k=l this uses a H support, otherwise H bar
                //  (b^k(H_+^l) * (1/4) * (w_k + (w_k w_l + Vkl)/sqrt_dt)
                // + b^k(H_-^l) * (1/4) * (w_k - (w_k w_l + Vkl)/sqrt_dt)
                // +- b^k(Y_n) * (1/2 *w_k) (plus if k=l, else minus)
                let dj = (dw[k] * dw[l] + dv[(k, l)]) / sqrt_dt;
                let h_plus_type_support = if k == l { &h_plus[l] } else { &h_bar_plus[l] };
                let h_minus_type_support = if k == l { &h_minus[l] } else { &h_bar_minus[l] };
                out += &system.get_incoherent_step(
                    k,
                    Complex {
                        re: 0.25 * (dw[k] + dj),
                        im: 0.0,
                    },
                    h_plus_type_support,
                    t,
                );
                out += &system.get_incoherent_step(
                    k,
                    Complex {
                        re: 0.25 * (dw[k] - dj),
                        im: 0.0,
                    },
                    h_minus_type_support,
                    t,
                );

                if k == l {
                    out += &(&operators.incoherent[k] * (0.5 * dw[k]));
                } else {
                    out -= &(&operators.incoherent[k] * (0.5 * dw[k]));
                }
            }
        }
        out
    }
}
