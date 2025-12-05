use ndarray::{Array1, array};
use ndarray_linalg::Scalar;
use num_complex::Complex;

use crate::system::simple_stochastic::{SimpleStochasticFn, SimpleStochasticSDESystem};

pub trait LangevinParameters {
    fn dimensionless_mass(&self) -> f64;
    fn dimensionless_lambda(&self) -> f64;
    fn kbt_div_hbar(&self) -> f64;
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64;
    fn get_potential_coefficient(&self, idx: u32, alpha: Complex<f64>, ratio: Complex<f64>) -> f64;
}

#[derive(Clone)]
pub struct HarmonicLangevinParameters {
    pub dimensionless_mass: f64,
    /// The dimensionless frequency
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Omega}{K_b T}$
    /// ```
    pub dimensionless_omega: f64,
    /// The dimensionless friction coefficient
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Lambda}{K_b T}$
    /// ```
    pub dimensionless_lambda: f64,
    pub kbt_div_hbar: f64,
}

impl LangevinParameters for HarmonicLangevinParameters {
    fn dimensionless_mass(&self) -> f64 {
        self.dimensionless_mass
    }
    fn dimensionless_lambda(&self) -> f64 {
        self.dimensionless_lambda
    }
    fn kbt_div_hbar(&self) -> f64 {
        self.kbt_div_hbar
    }
    #[inline]
    fn get_potential_coefficient(
        &self,
        idx: u32,
        alpha: Complex<f64>,
        _ratio: Complex<f64>,
    ) -> f64 {
        if idx == 1 {
            return self.dimensionless_omega.square() * alpha.re / (2.0 * self.dimensionless_mass);
        }
        if idx == 2 {
            return self.dimensionless_omega.square() / (4.0 * self.dimensionless_mass);
        }
        0.0
    }
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64 {
        self.get_potential_coefficient(1, alpha, Complex { re: 1.0, im: 0.0 })
    }
}

#[derive(Clone)]
pub struct PeriodicLangevinParameters {
    pub dimensionless_mass: f64,
    /// The dimensionless potential, given as rfft coefficients
    /// ```latex
    /// $\displaystyle V(x) = K_b T (c_0 + \sum_{n=1}^{N} c_n 2 \cos{\left(n x\right)})$
    /// ```
    pub dimensionless_potential: Vec<Complex<f64>>,
    pub dk_times_lengthscale: f64,
    /// The dimensionless friction coefficient
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Lambda}{K_b T}$
    /// ```
    pub dimensionless_lambda: f64,
    pub kbt_div_hbar: f64,
}

impl LangevinParameters for PeriodicLangevinParameters {
    fn dimensionless_mass(&self) -> f64 {
        self.dimensionless_mass
    }
    fn dimensionless_lambda(&self) -> f64 {
        self.dimensionless_lambda
    }
    fn kbt_div_hbar(&self) -> f64 {
        self.kbt_div_hbar
    }
    /// Calculate the coefficient
    /// ```latex
    /// \begin{align}
    //     C_N
    //     &= \sum_{i=-\infty}^\infty e^{ik n_i x_0}V_i^F(ik n_i)^{N}e^{-(k n_i|\mu+\nu|)^2/4}
    //     \\
    //     &= \sum_{i=1}^\infty (e^{ik n_i x_0}V_i^F(ik n_i)^{N} + e^{-ik n_i x_0}{V_i^F}^*(-ik n_i)^{N}) e^{-(k n_i|\mu+\nu|)^2/4}
    // \end{align}
    /// ```
    fn get_potential_coefficient(&self, idx: u32, alpha: Complex<f64>, ratio: Complex<f64>) -> f64 {
        let k = self.dk_times_lengthscale;
        let x_0 = 2.0.sqrt() * alpha.re;
        let mut out = 0.0;
        for (n_i, v_k) in self.dimensionless_potential.iter().enumerate().skip(1) {
            #[allow(clippy::cast_precision_loss)]
            let phase = k * n_i as f64;

            let prefactor =
                2.0 * (-0.25 * self.dimensionless_mass * phase.square() / ratio.re).exp();

            let inner = (Complex::i() * phase * x_0).exp() * v_k * (Complex::i() * phase).powu(idx);

            out += prefactor * inner.re;
        }
        out
    }
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64 {
        let k = self.dk_times_lengthscale;
        let x_0 = 2.0.sqrt() * alpha.re;
        let mut out = 0.0;
        for (n_i, v_k) in self.dimensionless_potential.iter().enumerate().skip(1) {
            #[allow(clippy::cast_precision_loss)]
            let phase = k * n_i as f64;
            // Classically, the wavefunction has a width much less than the potential period
            // So the exponential prefactor is approximately 1
            let prefactor = 2.0;

            let inner = (Complex::i() * phase * x_0).exp() * v_k * (Complex::i() * phase).powu(1);

            out += prefactor * inner.re;
        }
        out
    }
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in a harmonic potential with Langevin dynamics.
///
/// We simulate alpha, where alpha is the coherent state parameter.
/// The SDE is given by:
/// ```latex
/// $d\alpha = \displaystyle \frac{K_{B} T \left(4 M^{2} \operatorname{im}{\left(\alpha\right)} - 2 i M \Lambda \operatorname{im}{\left(\alpha\right)} - i \Omega^{2} \operatorname{re}{\left(\alpha\right)}\right)}{2 M \text{hbar}} + \frac{i \sqrt{K_{B} T} \sqrt{\Lambda} \xi(t)}{\sqrt{M} \sqrt{\text{hbar}}}$
/// ```
#[must_use]
pub fn get_langevin_system<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> SimpleStochasticSDESystem {
    let prefactor = params.kbt_div_hbar() / (2.0 * params.dimensionless_mass());
    let alpha_im_factor = Complex {
        re: 4.0 * params.dimensionless_mass().square() * prefactor,
        im: -2.0 * params.dimensionless_mass() * params.dimensionless_lambda() * prefactor,
    };
    let force_prefactor = (prefactor * params.dimensionless_lambda()).sqrt();
    let params_coherent = params.clone();
    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];

            let c1 = params_coherent.get_classical_potential_coefficient(alpha);
            let potential_factor = Complex { re: 0.0, im: -c1 } * params_coherent.kbt_div_hbar();

            array![alpha_im_factor * alpha.im + potential_factor]
        }),
        incoherent: vec![Box::new(move |_t, _state| {
            array![Complex {
                re: 0.0,
                im: force_prefactor
            }]
        })],
    }
}

fn get_quantum_re_force_prefactor<T: LangevinParameters>(
    params: &T,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda() / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass().sqrt();
    Complex {
        re: prefactor * sqrt_m * ((2.0 / ratio.re) - 1.0),
        im: -2.0 * ratio.im * prefactor / (sqrt_m * ratio.re),
    }
}

fn get_quantum_im_force_prefactor<T: LangevinParameters>(
    params: &T,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda() / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass().sqrt();
    Complex {
        re: sqrt_m * (ratio.im / ratio.re) * prefactor,
        im: prefactor * (2.0 - ratio.re - ratio.im * (ratio.im / ratio.re)) / (sqrt_m),
    }
}

fn build_quantum_incoherent_terms<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> Vec<Box<SimpleStochasticFn>> {
    let params0 = params.clone();
    let params1 = params.clone();
    vec![
        Box::new(move |_t, state| {
            let ratio = state[1];
            let re_prefactor = get_quantum_re_force_prefactor(&params0, ratio);

            let mut out = Array1::zeros(state.len());
            out[0] = re_prefactor;
            out
        }),
        Box::new(move |_t, state| {
            let ratio = state[1];
            let im_prefactor = get_quantum_im_force_prefactor(&params1, ratio);

            let mut out = Array1::zeros(state.len());
            out[0] = im_prefactor;
            out
        }),
    ]
}

fn get_ratio_derivative<T: LangevinParameters>(
    params: &T,
    alpha: Complex<f64>,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = params.kbt_div_hbar() * 0.125;
    let r2 = ratio.square() * (params.dimensionless_lambda() + Complex { re: 0.0, im: 8.0 });
    let r1 = ratio * (4.0 * params.dimensionless_lambda());
    let c2 = params.get_potential_coefficient(2, alpha, ratio);
    let r0 = -4.0 * params.dimensionless_lambda()
        + c2 * Complex {
            re: 0.0,
            im: -8.0 * params.dimensionless_mass(),
        };
    prefactor * (r2 + r1 + r0)
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in the most stable squeezed state
/// in a harmonic potential with Calderia-Leggett quantum Langevin dynamics.
#[must_use]
pub fn get_stable_quantum_langevin_system<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> SimpleStochasticSDESystem {
    let prefactor = params.kbt_div_hbar() / (2.0 * params.dimensionless_mass());
    let alpha_im_factor = Complex {
        re: 4.0 * params.dimensionless_mass().square() * prefactor,
        im: -2.0 * params.dimensionless_mass() * params.dimensionless_lambda() * prefactor,
    };

    let params_coherent = params.clone();
    let params_incoherent = params.clone();

    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];
            let ratio = state[1];

            let c1 = params_coherent.get_potential_coefficient(1, alpha, ratio);
            let potential_factor = Complex { re: 0.0, im: -c1 } * params_coherent.kbt_div_hbar();

            array![
                alpha_im_factor * alpha.im + potential_factor,
                get_ratio_derivative(&params_coherent, alpha, ratio),
            ]
        }),
        incoherent: build_quantum_incoherent_terms(&params_incoherent),
    }
}
