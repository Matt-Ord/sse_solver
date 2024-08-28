use ndarray::Array1;
use ndarray_linalg::{Norm, Scalar};
use num_complex::Complex;

use rand::Rng;
use rand_distr::Distribution;

use crate::{
    distribution::{
        FourPointComplexW as FourPointComplexWDistribution,
        NinePointComplexW as NinePointComplexWDistribution, StandardComplexNormal,
        V as VDistribution,
    },
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
            incoherent: &rng
                .sample_iter::<Complex<_>, _>(StandardComplexNormal)
                .map(|d| d * sqt_dt)
                .take(system.n_incoherent())
                .collect(),
        };

        state + system.get_step(&step, state, t)
    }
}

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
            incoherent: &noise.iter().map(|d| 0.5f64 * (d + sqrt_dt)).collect(),
        };
        out += &T::get_step_from_parts(&parts, &simple_step);

        // Parts for the second supporting value required to calculate b'b term
        // according to eq 11.1.4 in <https://doi.org/10.1007/978-3-662-12616-5>
        // \bar{Y}(n) = Y(n) + a dt + b sqrt(dt)
        // as suggested in the above book, we drop the \underline{a}_k term
        // \bar{Y}(n) = Y(n) + \sum_j b^j dW^j
        let second_supporting_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: &(0..system.n_incoherent())
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
            incoherent: &noise.iter().map(|d| (d + 0.5f64 * sqrt_dt)).collect(),
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

        // let noise = rng
        //     .sample_iter::<Complex<_>, _>(StandardComplexNormal)
        //     .map(|d| d * sqrt_dt)
        //     .take(system.n_incoherent())
        //     .collect::<Vec<_>>();

        let noise = &rng
            .sample_iter(NinePointComplexWDistribution::new(dt))
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let parts = system.get_parts(state, t);

        // Calculate all supporting states

        let y_supporting_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: noise,
        };

        let y_supporting_state = state + T::get_step_from_parts(&parts, &y_supporting_step);

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
                incoherent: &noise
                    .iter()
                    .map(|dw| 0.5 * dw * (2.0 - system.n_incoherent() as f64))
                    .collect(),
            },
        );

        let operators = T::get_operators_from_parts(&parts);

        let u_plus_supporting_states = operators
            .incoherent
            .iter()
            .map(|incoherent| state + (incoherent * sqrt_dt))
            .collect::<Vec<_>>();

        // 1/4 \sum_j \sum_r b^j(Ur+) dw^j + (dw^j dw^r + vrj) / sqrt(dt)
        for (j, dwj) in noise.iter().enumerate() {
            for (r, (u_plus_supporting_state, dwr)) in
                u_plus_supporting_states.iter().zip(noise).enumerate()
            {
                let pair_dw = (dwj * dwr + v[(j, r)]) / sqrt_dt;
                if j == r {
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj + pair_dw),
                        // U bar plus
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
                u_minus_supporting_states.iter().zip(noise).enumerate()
            {
                let pair_dw = (dwj * dwr + v[(j, r)]) / sqrt_dt;
                if j == r {
                    // R supporting value terms
                    out += &system.get_incoherent_step(
                        j,
                        0.25f64 * (dwj - pair_dw),
                        // U bar minus
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
            .sample_iter(NinePointComplexWDistribution::new(dt))
            .take(system.n_incoherent())
            .map(|s| Complex { re: s, im: 0.0 })
            .collect::<Vec<_>>();

        let parts = system.get_parts(state, t);

        // H_0 = Y_n + a(Y_n) dt + \sum_j b^j w_j
        let h0 = state
            + T::get_step_from_parts(
                &parts,
                &SDEStep {
                    coherent: Complex { re: dt, im: 0f64 },
                    incoherent: dw,
                },
            );

        let operators = T::get_operators_from_parts(&parts);

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

        // Y_n+1 = Y_n + (a(Y_n) + a(H_0)) dt / 2 + ...
        let mut step = (&operators.coherent
            + system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h0, t))
            * (dt * 0.5);

        for k in 0..system.n_incoherent() {
            for l in 0..system.n_incoherent() {
                // Add H - type increment
                // If k=l this uses a H support, otherwise H bar
                //  (b^k(H_+^l) * (1/4) * (w_k + (w_k w_l + Vkl)/sqrt_dt)
                // + b^k(H_-^l) * (1/4) * (w_k - (w_k w_l + Vkl)/sqrt_dt)
                // +- b^k(Y_n) * (1/2 *w_k) (plus if k=l, else minus)
                let dj = (dw[k] * dw[l] + dv[(k, l)]) / sqrt_dt;
                let h_plus_type_support = if k == l {
                    // H_+-^k = Y_n + a(Y_n) dt +- b^j \sqrt(dt)
                    &(&h_bar_plus[l] + (&operators.coherent * dt))
                } else {
                    &h_bar_plus[l]
                };
                step += &system.get_incoherent_step(k, 0.25 * (dw[k] + dj), h_plus_type_support, t);

                let h_minus_type_support = if k == l {
                    // H_+-^k = Y_n + a(Y_n) dt +- b^j \sqrt(dt)s
                    &(&h_bar_minus[l] + (&operators.coherent * dt))
                } else {
                    &h_bar_minus[l]
                };
                step +=
                    &system.get_incoherent_step(k, 0.25 * (dw[k] - dj), h_minus_type_support, t);

                if k == l {
                    step += &(&operators.incoherent[k] * (0.5 * dw[k]));
                } else {
                    step -= &(&operators.incoherent[k] * (0.5 * dw[k]));
                }
            }
        }
        state + step
    }
}

/// The R5 Solver defined in
/// <https://www.jstor.org/stable/27862707>
///
/// See 5.4 for the SRK formula
/// See Table 5.2 for the various weights
///
/// Note in this implimentation we assume a(t) = a(0)
pub struct Order2ExplicitWeakR5Solver {}

impl Order2ExplicitWeakR5Solver {
    // Note: for explicit solvers, A and B are lower diagonals
    pub const A0: [[f64; 3]; 3] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [25.0 / 144.0, 35.0 / 144.0, 0.0],
    ];
    pub const A1: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.0, 0.0]];
    pub const A2: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

    pub const B0: [[f64; 3]; 3] = [
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 0.0, 0.0],
        [-5.0 / 6.0, 0.0, 0.0],
    ];
    pub const B1: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]];
    pub const B2: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];

    pub const C0: [f64; 3] = [0.0, 1.0, 5.0 / 12.0];
    pub const C1: [f64; 3] = [0.0, 0.25, 0.25];
    pub const C2: [f64; 3] = [0.0, 0.0, 0.0];

    pub const ALPHA: [f64; 3] = [0.1, 3.0 / 14.0, 24.0 / 35.0];

    pub const BETA1: [f64; 3] = [1.0, -1.0, -1.0];
    pub const BETA2: [f64; 3] = [0.0, 1.0, -1.0];
    pub const BETA3: [f64; 3] = [0.5, -0.25, -0.25];
    pub const BETA4: [f64; 3] = [0.0, 0.5, -0.5];
}

#[inline]
fn get_supporting_state_lazy(
    state: &Array1<Complex<f64>>,
    increment: &[(&Array1<Complex<f64>>, f64)],
) -> Array1<Complex<f64>> {
    let mut out = state.clone();
    for (s, ds) in increment {
        if (ds - 0.0f64).abs() < 1e-100 {
            out += &(Complex { re: *ds, im: 0.0 } * *s);
        }
    }
    out
}

impl Solver for Order2ExplicitWeakR5Solver {
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let rng = rand::thread_rng();

        let i_hat = NinePointComplexWDistribution::new(dt)
            .sample_iter(rng.clone())
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let i_bar = FourPointComplexWDistribution::new(dt)
            .sample_iter(rng.clone())
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let sqrt_dt = dt.sqrt();

        let h_00 = state;

        // a(t, H_0^0) dt
        let h_00_step = &system.get_operators(h_00, t);
        let h_00_coherent = &h_00_step.coherent;
        //NOTE: here since H_k0 == H_00, we just use incoherent ops of H_00
        let h_k0_incoherent = &h_00_step.incoherent;
        let h_hat_k0_incoherent = &h_00_step.incoherent;

        let mut h_01 = get_supporting_state_lazy(state, &[(h_00_coherent, Self::A0[1][0] * dt)]);
        if Self::B0[1][0] != 0.0 {
            h_k0_incoherent
                .iter()
                .zip(&i_hat)
                .for_each(|(b, i)| h_01 += &((i * Self::B0[1][0]) * b));
        }

        let h_k1 = h_k0_incoherent
            .iter()
            .map(|b_0| {
                get_supporting_state_lazy(
                    state,
                    &[
                        (h_00_coherent, Self::A1[1][0] * dt),
                        (b_0, Self::B1[1][0] * sqrt_dt),
                    ],
                )
            })
            .collect::<Vec<_>>();

        let h_hat_k1 = (0..system.n_incoherent())
            .map(|k| {
                let mut out =
                    get_supporting_state_lazy(state, &[(h_00_coherent, Self::A2[1][0] * dt)]);
                if Self::B2[1][0] != 0.0 {
                    h_k0_incoherent.iter().enumerate().for_each(|(l, b)| {
                        if l == k {
                            return;
                        }
                        let factor = if l > k {
                            i_hat[k] * i_hat[l].conj() - sqrt_dt * i_bar[k]
                        } else {
                            i_hat[k] * i_hat[l].conj() + sqrt_dt * i_bar[l]
                        };

                        out += &(((0.5 * Self::B2[1][0] * factor) / sqrt_dt) * b);
                    });
                }
                out
            })
            .collect::<Vec<_>>();

        let h_01_coherent = &system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h_01, t);
        let h_k1_incoherent = &h_k1
            .iter()
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t))
            .collect::<Vec<_>>();

        let mut h_02 = get_supporting_state_lazy(
            state,
            &[
                (h_00_coherent, Self::A0[2][0] * dt),
                (h_01_coherent, Self::A0[2][1] * dt),
                // TODO: find a way to suppoort this better...
                // (b_0, Self::B0[2][0] * dt),
                // (b_1, Self::B0[2][1] * dt),
            ],
        );
        if Self::B0[2][0] != 0.0 {
            h_k0_incoherent
                .iter()
                .zip(&i_hat)
                .for_each(|(b, i)| h_02 += &((i * Self::B0[2][0]) * b));
        }
        if Self::B0[2][1] != 0.0 {
            h_k1_incoherent
                .iter()
                .zip(&i_hat)
                .for_each(|(b, i)| h_02 += &((i * Self::B0[2][1]) * b));
        }

        let h_k2 = h_k0_incoherent
            .iter()
            .zip(h_k1_incoherent)
            .map(|(b_0, b_1)| {
                get_supporting_state_lazy(
                    state,
                    &[
                        (h_00_coherent, Self::A1[2][0] * dt),
                        (h_01_coherent, Self::A1[2][1] * dt),
                        (b_0, Self::B1[2][0] * dt),
                        (b_1, Self::B1[2][1] * dt),
                    ],
                )
            })
            .collect::<Vec<_>>();

        let h_hat_k2 = (0..system.n_incoherent())
            .map(|k| {
                let mut out = get_supporting_state_lazy(
                    state,
                    &[
                        (h_00_coherent, Self::A2[2][0] * dt),
                        (h_01_coherent, Self::A2[2][1] * dt),
                    ],
                );
                if Self::B2[2][0] != 0.0 {
                    h_k0_incoherent.iter().enumerate().for_each(|(l, b)| {
                        if l == k {
                            return;
                        }
                        let factor = if l > k {
                            i_hat[k] * i_hat[l].conj() - sqrt_dt * i_bar[k]
                        } else {
                            i_hat[k] * i_hat[l].conj() + sqrt_dt * i_bar[l]
                        };

                        out += &(((0.5 * Self::B2[2][0] * factor) / sqrt_dt) * b);
                    });
                }

                if Self::B2[2][1] != 0.0 {
                    h_k1_incoherent.iter().enumerate().for_each(|(l, b)| {
                        if l == k {
                            return;
                        }
                        let factor = if l > k {
                            i_hat[k] * i_hat[l].conj() - sqrt_dt * i_bar[k]
                        } else {
                            i_hat[k] * i_hat[l].conj() + sqrt_dt * i_bar[l]
                        };

                        out += &(((0.5 * Self::B2[2][1] * factor) / sqrt_dt) * b);
                    });
                }
                out
            })
            .collect::<Vec<_>>();

        let h_02_coherent = &system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h_02, t);
        // Y_(n+1) = Y_(n) + \sum_i alpha_i a(t_n+c_i^0h_n, H_i^0)h_n
        let mut out = state
            + &(Complex {
                re: Self::ALPHA[0] * dt,
                im: 0.0,
            } * h_00_coherent)
            + &(Complex {
                re: Self::ALPHA[1] * dt,
                im: 0.0,
            } * h_01_coherent)
            + &(Complex {
                re: Self::ALPHA[2] * dt,
                im: 0.0,
            } * h_02_coherent);

        // Y_(n+1) += \sum_i \sum_k b^k(t, H_k) (i hat_k beta_i(1) + i_k,k hat / sqrt(dt))
        h_k0_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[0] * i_hat_k)
                    + ((0.5 * Self::BETA2[0] / sqrt_dt) * (i_hat_k.abs().square() - dt));
                out += &(factor * b_k);
            });

        h_k1_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[1] * i_hat_k)
                    + ((0.5 * Self::BETA2[1] / sqrt_dt) * (i_hat_k.abs().square() - dt));
                out += &(factor * b_k);
            });

        let h_k2_incoherent = h_k2
            .iter()
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t))
            .collect::<Vec<_>>();

        h_k2_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[2] * i_hat_k)
                    + ((0.5 * Self::BETA2[2] / sqrt_dt) * (i_hat_k.abs().square() - dt));
                out += &(factor * b_k);
            });

        // Y_(n+1) += \sum_i \sum_k b^k(t, H_hat_k) (i hat_k beta_i(3) + beta_i(4)*sqrt(dt))
        h_hat_k0_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[0] * i_hat_k) + (Self::BETA4[0] * sqrt_dt);
                out += &(factor * b_k);
            });
        let h_hat_k1_incoherent = h_hat_k1
            .iter()
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t));

        h_hat_k1_incoherent
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[1] * i_hat_k) + (Self::BETA4[1] * sqrt_dt);
                out += &(factor * b_k);
            });

        let h_hat_k2_incoherent = h_hat_k2
            .iter()
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t));

        h_hat_k2_incoherent
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[2] * i_hat_k) + (Self::BETA4[2] * sqrt_dt);
                out += &(factor * b_k);
            });
        out
    }
}
