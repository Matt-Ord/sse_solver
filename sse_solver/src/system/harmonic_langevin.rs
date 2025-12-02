use ndarray::array;
use ndarray_linalg::Scalar;
use num_complex::Complex;

use crate::system::simple_stochastic::SimpleStochasticSDESystem;

#[derive(Clone)]
pub struct HarmonicLangevinParameters {
    pub dimensionless_mass: f64,
    pub dimensionless_omega: f64,
    pub dimensionless_lambda: f64,
    pub kbt_div_hbar: f64,
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
pub fn get_harmonic_langevin_system(
    params: &HarmonicLangevinParameters,
) -> SimpleStochasticSDESystem {
    let prefactor = params.kbt_div_hbar / (2.0 * params.dimensionless_mass);
    let alpha_re_factor = Complex {
        re: 0.0,
        im: -params.dimensionless_omega.square() * prefactor,
    };
    let alpha_im_factor = Complex {
        re: 4.0 * params.dimensionless_mass.square() * prefactor,
        im: -2.0 * params.dimensionless_mass * params.dimensionless_lambda * prefactor,
    };
    let force_prefactor = (prefactor * params.dimensionless_lambda).sqrt();
    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];

            array![alpha_re_factor * alpha.re + alpha_im_factor * alpha.im]
        }),
        incoherent: vec![Box::new(move |_t, _state| {
            array![Complex {
                re: 0.0,
                im: force_prefactor
            }]
        })],
    }
}

fn get_quantum_re_force_prefactor(
    params: &HarmonicLangevinParameters,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar * params.dimensionless_lambda / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass.sqrt();

    Complex {
        re: prefactor * sqrt_m * ((2.0 / ratio.re) - 1.0),
        im: -2.0 * ratio.im * prefactor / (sqrt_m * ratio.re),
    }
}

fn get_quantum_im_force_prefactor(
    params: &HarmonicLangevinParameters,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar * params.dimensionless_lambda / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass.sqrt();

    Complex {
        re: sqrt_m * (ratio.im / ratio.re) * prefactor,
        im: prefactor * (2.0 - ratio.re - ratio.im * (ratio.im / ratio.re)) / (sqrt_m),
    }
}

fn get_quantum_ratio_derivative(
    params: &HarmonicLangevinParameters,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = params.kbt_div_hbar / (4.0);
    let r2 = ratio.square() * (params.dimensionless_lambda + Complex { re: 0.0, im: 8.0 });
    let r1 = ratio * (4.0 * params.dimensionless_lambda);
    let r0 = -4.0 * params.dimensionless_lambda
        + params.dimensionless_omega.square() * Complex { re: 0.0, im: -2.0 };
    prefactor * (r2 + r1 + r0)
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in a harmonic potential with Calderia-Leggett quantum Langevin dynamics.
#[must_use]
pub fn get_harmonic_quantum_langevin_system(
    params: &HarmonicLangevinParameters,
) -> SimpleStochasticSDESystem {
    let prefactor = params.kbt_div_hbar / (2.0 * params.dimensionless_mass);
    let alpha_re_factor = Complex {
        re: 0.0,
        im: -params.dimensionless_omega.square() * prefactor,
    };
    let alpha_im_factor = Complex {
        re: 4.0 * params.dimensionless_mass.square() * prefactor,
        im: -2.0 * params.dimensionless_mass * params.dimensionless_lambda * prefactor,
    };

    let params_coherent = params.clone();
    let params0 = params.clone();
    let params1 = params.clone();

    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];
            let ratio = state[1];

            array![
                alpha_re_factor * alpha.re + alpha_im_factor * alpha.im,
                get_quantum_ratio_derivative(&params_coherent, ratio),
            ]
        }),
        incoherent: vec![
            Box::new(move |_t, state| {
                let ratio = state[1];
                let re_prefactor = get_quantum_re_force_prefactor(&params0, ratio);
                array![re_prefactor, Complex { re: 0.0, im: 0.0 }]
            }),
            Box::new(move |_t, state| {
                let ratio = state[1];
                let im_prefactor = get_quantum_im_force_prefactor(&params1, ratio);
                array![im_prefactor, Complex { re: 0.0, im: 0.0 }]
            }),
        ],
    }
}
