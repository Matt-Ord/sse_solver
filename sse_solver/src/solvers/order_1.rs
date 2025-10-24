use ndarray::Array1;
use num_complex::Complex;

use rand::Rng;

use crate::{
    distribution::StandardComplexNormal,
    system::{SDEStep, SDESystem},
};

use super::Stepper;

#[derive(Default)]
#[allow(clippy::module_name_repetitions)]
pub struct EulerStepper {}

impl Stepper for EulerStepper {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        // The basic euler method
        // Y_n+1 = Y_n + a dt + \sum_k b_k dW
        // where dW are normalized gaussian random variables,  <dW_k* dW_k'> = dt

        let rng = rand::rng();
        let sqt_dt = dt.sqrt();
        let step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: &rng
                .sample_iter::<Complex<_>, _>(StandardComplexNormal)
                .map(|d| d * sqt_dt)
                .take(system.n_incoherent())
                .collect(),
        };

        (system.get_step(&step, state, t), None)
    }
}

pub struct MilstenStepper {}

impl Stepper for MilstenStepper {
    fn step<T: SDESystem>(
        &self,
        state: &Array1<Complex<f64>>,
        system: &T,
        t: f64,
        dt: f64,
    ) -> (Array1<Complex<f64>>, Option<f64>) {
        // The explicit milsten scheme for commuting noise
        // Y_k(n+1) = Y_k(n) + \underline{a}_k dt + \frac{1}{2} \sum_j (b^j(t, \bar{Y}(n))_k + b^j(t, Y(n))_k)dW^j
        // where dW are normalized gaussian random variables,  <dW_k* dW_k'> = dt

        // Pre-compute the system parts, since we use them twice (for supporting value and actual step)
        let parts = system.get_parts(state, t);

        let rng = rand::rng();
        let sqrt_dt = dt.sqrt();

        let noise = rng
            .sample_iter::<Complex<_>, _>(StandardComplexNormal)
            .map(|d| d * sqrt_dt)
            .take(system.n_incoherent())
            .collect::<Vec<_>>();

        // The non-supported part of the step
        // Y_k(n+1) = Y_k(n) + a dt + \sum_j \frac{1}{2}  b^j(t, Y(n))_k dW^j - sqrt(dt)(b^j(t, Y(n))_k)
        let simple_step = SDEStep {
            coherent: Complex { re: dt, im: 0f64 },
            incoherent: &noise.iter().map(|d| 0.5f64 * (d + sqrt_dt)).collect(),
        };
        let mut out = T::get_step_from_parts(&parts, &simple_step);

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
            incoherent: &noise.iter().map(|d| d + 0.5f64 * sqrt_dt).collect(),
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

        (out, None)
    }
}
