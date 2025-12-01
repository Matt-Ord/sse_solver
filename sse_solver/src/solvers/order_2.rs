use ndarray::Array1;
use ndarray_linalg::Norm;
use num_complex::Complex;

use rand::Rng;
use rand_distr::Distribution;

use crate::{
    distribution::{ThreePointW, TwoPointW, V as VDistribution},
    system::{SDEStep, SDESystem},
};

use super::Stepper;

pub enum ErrorMeasure {
    L1,
    L2,
    Max,
}
/// The Order 2 General Weak Taylor Scheme defined in 5.1 of
/// <https://www.jstor.org/stable/27862707>
///
/// This method scales poorly for large n operators compared
/// to the other methods discussed in the paper
pub struct ExplicitWeakStepper {
    pub error_measure: Option<ErrorMeasure>,
}

impl Stepper for ExplicitWeakStepper {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        let mut rng = rand::rng();
        // Sample the V distribution
        let dv = &rng.sample(VDistribution {
            dt,
            n: system.n_incoherent(),
        });

        // Sample the W distribution (this is called I in the paper)
        let dw = &rng
            .sample_iter(ThreePointW::new(dt))
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        let parts = system.get_parts(state, t);
        let euler_step = T::get_step_from_parts(
            &parts,
            &SDEStep {
                coherent: dt,
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
        // Where H_0 = Y_n + a(Y_n) dt + \sum_j b^j w_j is the result of a single euler step
        // H_0 = Y_n + a(Y_n) dt + \sum_j b^j w_j
        let mut step = (&operators.coherent
            + system.get_coherent_step(1.0, &(state + &euler_step), t))
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

        let error = match self.error_measure {
            Some(ErrorMeasure::L1) => Some((&step - euler_step).norm_l1()),
            Some(ErrorMeasure::L2) => Some((&step - euler_step).norm_l2()),
            Some(ErrorMeasure::Max) => Some((&step - euler_step).norm_max()),
            None => None,
        };

        (step, error)
    }
}

/// The R5 Stepper defined in
/// <https://www.jstor.org/stable/27862707>
///
/// See 5.4 for the SRK formula
/// See Table 5.2 for the various weights
///
/// Note in this implimentation we assume a(t) = a(0)
pub struct ExplicitWeakR5Stepper {
    pub error_measure: Option<ErrorMeasure>,
}

impl ExplicitWeakR5Stepper {
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

struct Increment {
    i_hat_noise: Vec<f64>,
    i_bar_noise: Vec<f64>,
    dt: f64,
    sqrt_dt: f64,
}

impl Increment {
    fn n_incoherent(&self) -> usize {
        self.i_hat_noise.len()
    }

    fn i_bar_pair_noise(&self, k: usize, l: usize) -> f64 {
        let product = self.i_hat_noise[k] * self.i_hat_noise[l];
        if k == l {
            return 0.5 * (product - self.dt);
        }
        if k < l {
            0.5 * (product - (self.sqrt_dt * self.i_bar_noise[k]))
        } else {
            0.5 * (product + (self.sqrt_dt * self.i_bar_noise[l]))
        }
    }

    fn new(n_incoherent: usize, dt: f64) -> Self {
        let rng = rand::rng();
        let i_hat_noise = ThreePointW::new(dt)
            .sample_iter(rng.clone())
            .take(n_incoherent)
            .collect::<Vec<_>>();
        let i_bar_noise = TwoPointW::new(dt)
            .sample_iter(rng.clone())
            .take(n_incoherent)
            .collect::<Vec<_>>();
        Self {
            i_hat_noise,
            i_bar_noise,
            dt,
            sqrt_dt: dt.sqrt(),
        }
    }
}

impl ExplicitWeakR5Stepper {
    // Note: for explicit solvers, A and B are lower diagonals
    #[inline]
    #[must_use]
    fn get_h_0_state<const I: usize>(
        state: &Array1<Complex<f64>>,
        coherent_steps: [&Array1<Complex<f64>>; I],
        incoherent_steps: [&Vec<Array1<Complex<f64>>>; I],
        increment: &Increment,
    ) -> Array1<Complex<f64>> {
        debug_assert_eq!(I, incoherent_steps.len());

        let mut out = state.clone();
        for (j, step) in coherent_steps.into_iter().enumerate() {
            if Self::A0[I][j] != 0.0 {
                out += &(step * (Self::A0[I][j] * increment.dt));
            }
        }

        for (j, step) in incoherent_steps.iter().enumerate() {
            if Self::B0[I][j] != 0.0 {
                step.iter()
                    .zip(&increment.i_hat_noise)
                    .for_each(|(b, i_hat_l)| out += &(b * (Self::B0[I][j] * i_hat_l)));
            }
        }

        out
    }

    #[inline]
    fn get_h_k_states<'a, const I: usize>(
        state: &'a Array1<Complex<f64>>,
        coherent_steps: [&'a Array1<Complex<f64>>; I],
        incoherent_steps: [&'a Vec<Array1<Complex<f64>>>; I],
        increment: &'a Increment,
    ) -> impl Iterator<Item = Array1<Complex<f64>>> + 'a {
        (0..increment.n_incoherent()).map(move |ik| {
            let mut out = state.to_owned();
            for (j, step) in coherent_steps.into_iter().enumerate() {
                if Self::A1[I][j] != 0.0 {
                    out += &(step * (Self::A1[I][j] * increment.dt));
                }
            }
            for (j, steps) in incoherent_steps.iter().enumerate() {
                if Self::B1[I][j] != 0.0 {
                    out += &(&steps[ik] * (Self::B1[I][j] * increment.sqrt_dt));
                }
            }
            out
        })
    }
    #[inline]
    fn get_h_hat_k_states<'a, const I: usize>(
        state: &'a Array1<Complex<f64>>,
        coherent_steps: [&'a Array1<Complex<f64>>; I],
        incoherent_steps: [&'a Vec<Array1<Complex<f64>>>; I],
        increment: &'a Increment,
    ) -> impl Iterator<Item = Array1<Complex<f64>>> + 'a {
        (0..increment.n_incoherent()).map(move |ik| {
            let mut out = state.clone();
            for (j, step) in coherent_steps.into_iter().enumerate() {
                if Self::A2[I][j] != 0.0 {
                    out += &(step * (Self::A2[I][j] * increment.dt));
                }
            }
            for (j, steps) in incoherent_steps.iter().enumerate() {
                if Self::B2[I][j] != 0.0 {
                    steps.iter().enumerate().for_each(|(il, step)| {
                        if il == ik {
                            return;
                        }
                        let factor = increment.i_bar_pair_noise(ik, il) / increment.sqrt_dt;

                        out += &(step * (Self::B2[I][j] * factor));
                    });
                }
            }
            out
        })
    }
}

impl Stepper for ExplicitWeakR5Stepper {
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        let increment = Increment::new(system.n_incoherent(), dt);

        let h_00 = &Self::get_h_0_state::<0>(state, [], [], &increment);

        let h_00_coherent = &system.get_coherent_step(1.0, h_00, t + (Self::C0[0] * dt));

        // NOTE: here since H_k0 == The initial state, we don't pre-calculate them
        let h_k0_incoherent = &(0..system.n_incoherent())
            .map(|idx| system.get_incoherent_step(idx, 1.0, state, t + (Self::C1[0] * dt)))
            .collect::<Vec<_>>();

        let h_01 = Self::get_h_0_state::<1>(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_k1 = Self::get_h_k_states::<1>(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_hat_k1 =
            Self::get_h_hat_k_states::<1>(state, [h_00_coherent], [h_k0_incoherent], &increment);

        let h_01_coherent = &system.get_coherent_step(1.0, &h_01, t + (Self::C0[1] * dt));

        let h_k1_incoherent = &h_k1
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, 1.0, &s, t + (Self::C1[1] * dt)))
            .collect::<Vec<_>>();

        let h_02 = Self::get_h_0_state::<2>(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );
        let h_k2 = Self::get_h_k_states::<2>(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );
        let h_hat_k2 = Self::get_h_hat_k_states::<2>(
            state,
            [h_00_coherent, h_01_coherent],
            [h_k0_incoherent, h_k1_incoherent],
            &increment,
        );

        let h_02_coherent = &system.get_coherent_step(1.0, &h_02, t + (Self::C0[2] * dt));

        // Y_(n+1) = Y_(n) + \sum_i alpha_i a(t_n+c_i^0h_n, H_i^0)h_n
        let mut step = &(h_00_coherent * (Self::ALPHA[0] * dt))
            + &(h_01_coherent * (Self::ALPHA[1] * dt))
            + &(h_02_coherent * (Self::ALPHA[2] * dt));

        // Y_(n+1) += \sum_i \sum_k b^k(t, H_k) (i hat_k beta_i(1) + i_k,k hat / sqrt(dt))
        h_k0_incoherent
            .iter()
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[0] * i_hat_k)
                    + ((0.5 * Self::BETA2[0] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                step += &(b_k * factor);
            });

        h_k1_incoherent
            .iter()
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[1] * i_hat_k)
                    + ((0.5 * Self::BETA2[1] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                step += &(b_k * factor);
            });

        let h_k2_incoherent = h_k2
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, 1.0, &s, t + (Self::C1[2] * dt)))
            .collect::<Vec<_>>();

        h_k2_incoherent
            .iter()
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA1[2] * i_hat_k)
                    + ((0.5 * Self::BETA2[2] / increment.sqrt_dt) * ((i_hat_k * i_hat_k) - dt));
                step += &(b_k * factor);
            });

        let h_hat_k0_incoherent = (0..increment.n_incoherent())
            .map(|idx| system.get_incoherent_step(idx, 1.0, state, t + (Self::C2[0] * dt)));

        // Y_(n+1) += \sum_i \sum_k b^k(t, H_hat_k) (i hat_k beta_i(3) + beta_i(4)*sqrt(dt))
        h_hat_k0_incoherent
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[0] * i_hat_k) + (Self::BETA4[0] * increment.sqrt_dt);
                step += &(b_k * factor);
            });

        let h_hat_k1_incoherent = h_hat_k1
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, 1.0, &s, t + (Self::C2[1] * dt)));

        h_hat_k1_incoherent
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[1] * i_hat_k) + (Self::BETA4[1] * increment.sqrt_dt);
                step += &(b_k * factor);
            });

        let h_hat_k2_incoherent = h_hat_k2
            .enumerate()
            .map(|(idx, s)| system.get_incoherent_step(idx, 1.0, &s, t + (Self::C2[2] * dt)));

        h_hat_k2_incoherent
            .zip(&increment.i_hat_noise)
            .for_each(|(b_k, i_hat_k)| {
                let factor = (Self::BETA3[2] * i_hat_k) + (Self::BETA4[2] * increment.sqrt_dt);
                step += &(b_k * factor);
            });

        let euler_step = system.get_step(
            &SDEStep {
                coherent: dt,
                incoherent: &increment.i_hat_noise.into_iter().collect(),
            },
            state,
            t,
        );
        let error = match self.error_measure {
            Some(ErrorMeasure::L1) => Some((&step - euler_step).norm_l1()),
            Some(ErrorMeasure::L2) => Some((&step - euler_step).norm_l2()),
            Some(ErrorMeasure::Max) => Some((&step - euler_step).norm_max()),
            None => None,
        };

        (step, error)
    }
}
