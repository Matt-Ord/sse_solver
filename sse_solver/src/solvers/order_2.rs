use ndarray::Array1;
use ndarray_linalg::Scalar;
use num_complex::Complex;

use rand::Rng;
use rand_distr::Distribution;

use crate::{
    distribution::{ThreePointW, TwoPointW, V as VDistribution},
    system::{SDEStep, SDESystem},
};

use super::Solver;

/// The Order 2 General Weak Taylor Scheme defined in 5.1 of
/// <https://www.jstor.org/stable/27862707>
///
/// This method scales poorly for large n operators compared
/// to the other methods discussed in the paper
pub struct ExplicitWeakSolver {}

impl Solver for ExplicitWeakSolver {
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
            .sample_iter(ThreePointW::new(dt))
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
pub struct ExplicitWeakR5Solver {}

impl ExplicitWeakR5Solver {
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

struct Increment<'a> {
    i_hat_noise: &'a Vec<Complex<f64>>,
    i_bar_noise: &'a Vec<Complex<f64>>,
    dt: f64,
    sqrt_dt: f64,
}

impl<'a> Increment<'a> {
    fn n_incoherent(&self) -> usize {
        self.i_hat_noise.len()
    }

    fn i_bar_pair_noise(&self, i: usize, j: usize) -> Complex<f64> {
        let product = self.i_hat_noise[i] * self.i_hat_noise[j];
        if i == j {
            return 0.5 * (product - self.dt);
        }
        if i < j {
            0.5 * (product - self.sqrt_dt * self.i_bar_noise[i])
        } else {
            0.5 * (product + self.sqrt_dt * self.i_bar_noise[j])
        }
    }
}

impl ExplicitWeakR5Solver {
    // Note: for explicit solvers, A and B are lower diagonals
    #[inline]
    #[must_use]
    fn get_h_0_state<const N: usize>(
        state: &Array1<Complex<f64>>,
        coherent_steps: [&Array1<Complex<f64>>; N],
        incoherent_steps: [&Vec<Array1<Complex<f64>>>; N],
        increment: &Increment,
    ) -> Array1<Complex<f64>> {
        debug_assert_eq!(N, incoherent_steps.len());

        let mut out = state.clone();
        for (i, step) in coherent_steps.into_iter().enumerate() {
            if Self::A0[N][i] != 0.0 {
                out += &(Complex {
                    re: Self::A0[N][i] * increment.dt,
                    im: 0.0,
                } * step);
            }
        }

        for (i, step) in incoherent_steps.into_iter().enumerate() {
            if Self::B0[N][i] != 0.0 {
                step.iter()
                    .zip(increment.i_hat_noise)
                    .for_each(|(b, i_hat)| out += &((Self::B0[N][i] * i_hat) * b));
            }
        }

        out
    }

    #[inline]
    #[must_use]
    fn get_h_k_states<const N: usize>(
        state: &Array1<Complex<f64>>,
        coherent_steps: [&Array1<Complex<f64>>; N],
        incoherent_steps: [&Vec<Array1<Complex<f64>>>; N],
        increment: &Increment,
    ) -> Vec<Array1<Complex<f64>>> {
        (0..increment.n_incoherent())
            .map(|ik| {
                let mut out = state.clone();
                for (i, step) in coherent_steps.into_iter().enumerate() {
                    if Self::A1[N][i] != 0.0 {
                        out += &(Complex {
                            re: Self::A1[N][i] * increment.dt,
                            im: 0.0,
                        } * step);
                    }
                }
                for (i, steps) in incoherent_steps.into_iter().enumerate() {
                    if Self::B1[N][i] != 0.0 {
                        out += &(Complex {
                            re: Self::B1[N][i] * increment.dt.sqrt(),
                            im: 0.0,
                        } * &steps[ik]);
                    }
                }
                out
            })
            .collect()
    }
    #[inline]
    #[must_use]
    fn get_h_hat_k_states<const N: usize>(
        state: &Array1<Complex<f64>>,
        coherent_steps: [&Array1<Complex<f64>>; N],
        incoherent_steps: [&Vec<Array1<Complex<f64>>>; N],
        increment: &Increment,
    ) -> Vec<Array1<Complex<f64>>> {
        let sqrt_dt = increment.dt.sqrt();

        (0..increment.n_incoherent())
            .map(|ik| {
                let mut out = state.clone();
                for (i, step) in coherent_steps.into_iter().enumerate() {
                    if Self::A2[N][i] != 0.0 {
                        out += &(Complex {
                            re: Self::A2[N][i] * increment.dt,
                            im: 0.0,
                        } * step);
                    }
                }
                for (i, steps) in incoherent_steps.into_iter().enumerate() {
                    if Self::B2[N][i] != 0.0 {
                        steps.iter().enumerate().for_each(|(il, step)| {
                            if il == ik {
                                return;
                            }
                            let factor = increment.i_bar_pair_noise(il, ik);

                            out += &((Self::B2[N][i] * factor / sqrt_dt) * step);
                        });
                    }
                }
                out
            })
            .collect()
    }
}

impl Solver for ExplicitWeakR5Solver {
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let rng = rand::thread_rng();

        let i_hat = ThreePointW::new(dt)
            .sample_iter(rng.clone())
            .take(system.n_incoherent())
            .map(|d| Complex { re: d, im: 0.0 })
            .collect::<Vec<_>>();

        let i_bar = TwoPointW::new(dt)
            .sample_iter(rng.clone())
            .take(system.n_incoherent())
            .map(|d| Complex { re: d, im: 0.0 })
            .collect::<Vec<_>>();

        let increment = Increment {
            dt,
            i_bar_noise: &i_bar,
            i_hat_noise: &i_hat,
            sqrt_dt: dt.sqrt(),
        };

        let h_00 = &Self::get_h_0_state::<0>(state, [], [], &increment);

        let h_00_coherent =
            &system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, h_00, t + (Self::C0[0] * dt));

        // NOTE: here since H_k0 == The initial state, we don't pre-calculate them
        let h_k0_incoherent = &(0..system.n_incoherent())
            .map(|idx| {
                system.get_incoherent_step(
                    idx,
                    Complex { re: 1.0, im: 0.0 },
                    h_00,
                    t + (Self::C1[0] * dt),
                )
            })
            .collect::<Vec<_>>();

        let h_01 = Self::get_h_0_state::<1>(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_k1 = Self::get_h_k_states(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_hat_k1 =
            Self::get_h_hat_k_states(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_01_coherent =
            &system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h_01, t + (Self::C0[1] * dt));

        let h_k1_incoherent = &h_k1
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                system.get_incoherent_step(
                    idx,
                    Complex { re: 1.0, im: 0.0 },
                    s,
                    t + (Self::C1[1] * dt),
                )
            })
            .collect::<Vec<_>>();

        let h_02 = Self::get_h_0_state::<2>(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );
        let h_k2 = Self::get_h_k_states(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );
        let h_hat_k2 = Self::get_h_hat_k_states(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );

        let h_02_coherent =
            &system.get_coherent_step(Complex { re: 1.0, im: 0.0 }, &h_02, t + (Self::C0[2] * dt));

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
                    + ((0.5 * Self::BETA2[0] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                out += &(factor * b_k);
            });

        h_k1_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[1] * i_hat_k)
                    + ((Self::BETA2[1] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                out += &(factor * b_k);
            });

        let h_k2_incoherent = h_k2
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                system.get_incoherent_step(
                    idx,
                    Complex { re: 1.0, im: 0.0 },
                    s,
                    t + (Self::C1[2] * dt),
                )
            })
            .collect::<Vec<_>>();

        h_k2_incoherent
            .iter()
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[2] * i_hat_k)
                    + ((0.5 * Self::BETA2[2] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                out += &(factor * b_k);
            });

        let h_hat_k0_incoherent =
            (0..system.n_incoherent())
                .zip(std::iter::repeat(h_00))
                .map(|(idx, s)| {
                    system.get_incoherent_step(
                        idx,
                        Complex { re: 1.0, im: 0.0 },
                        s,
                        t + (Self::C2[0] * dt),
                    )
                });

        // Y_(n+1) += \sum_i \sum_k b^k(t, H_hat_k) (i hat_k beta_i(3) + beta_i(4)*sqrt(dt))
        h_hat_k0_incoherent
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[0] * i_hat_k) + (Self::BETA4[0] * increment.sqrt_dt);
                out += &(factor * b_k);
            });

        let h_hat_k1_incoherent = h_hat_k1.iter().enumerate().map(|(idx, s)| {
            system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t + (Self::C2[1] * dt))
        });

        h_hat_k1_incoherent
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[1] * i_hat_k) + (Self::BETA4[1] * increment.sqrt_dt);
                out += &(factor * b_k);
            });

        let h_hat_k2_incoherent = h_hat_k2.iter().enumerate().map(|(idx, s)| {
            system.get_incoherent_step(idx, Complex { re: 1.0, im: 0.0 }, s, t + (Self::C2[2] * dt))
        });

        h_hat_k2_incoherent
            .zip(i_hat.iter())
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[2] * i_hat_k) + (Self::BETA4[2] * increment.sqrt_dt);
                out += &(factor * b_k);
            });
        out
    }
}
