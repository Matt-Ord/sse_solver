use std::sync::Arc;

use ndarray::array;
use ndarray_linalg::Scalar;
use num_complex::Complex;

use crate::system::simple_stochastic::SimpleStochasticSDESystem;

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
        coherent: Arc::new(move |_t, state| {
            let alpha = state[0];

            array![alpha_re_factor * alpha.re + alpha_im_factor * alpha.im]
        }),
        incoherent: vec![Arc::new(move |_t, _state| {
            array![Complex {
                re: 0.0,
                im: force_prefactor
            }]
        })],
    }
}
