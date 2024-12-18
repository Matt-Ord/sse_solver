use ndarray::{linalg::Dot, Array1, Array2, Array3, Axis};
use num_complex::Complex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    sparse::{BandedArray, FactorizedArray, SplitScatteringArray, Tensor, TransposedBandedArray},
    system::{SDEOperators, SDEStep, SDESystem},
};

pub trait Noise {
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;

    fn get_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Vec<SSEStochasticPart>;

    fn get_incoherent_part(
        &self,
        index: usize,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> SSEStochasticIncoherentPart;

    fn get_incoherent_parts(
        &self,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Vec<SSEStochasticIncoherentPart>;
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct FullNoiseSource<T, U> {
    // Uses the convention taken from https://doi.org/10.1103/PhysRevA.66.012108
    // However we multiply L by a factor of i
    // L -> iL
    // H_int = (Lb^\dagger + bL^\dagger) where Z(t) = b(t)e^(iw_0t) is markovian
    // Note this has no effect in the final SSE.
    // [Z(t), Z(t)^\dagger] = \delta(t-s)
    // Note: we scale the operators such that gamma = 1
    operator: T,
    conjugate_operator: U,
}
#[derive(Clone)]
pub struct SSEParts<'a> {
    state: &'a Array1<Complex<f64>>,
    /// H |\psi>
    hamiltonian: Array1<Complex<f64>>,
    /// Parts from a the stochastic terms
    stochastic: Vec<SSEStochasticPart>,
}
#[derive(Clone)]
pub struct SSEStochasticPart {
    /// <L>
    expectation: Complex<f64>,
    /// L |\psi>
    l_state: Array1<Complex<f64>>,
    /// L^\dagger L |\psi>
    l_dagger_l_state: Array1<Complex<f64>>,
}
#[derive(Clone)]
pub struct SSEIncoherentParts<'a> {
    state: &'a Array1<Complex<f64>>,
    /// Parts from a the stochastic terms
    stochastic: Vec<SSEStochasticIncoherentPart>,
}
#[derive(Clone)]
pub struct SSEStochasticIncoherentPart {
    /// <L>
    expectation: Complex<f64>,
    /// L |\psi>
    l_state: Array1<Complex<f64>>,
}
#[derive(Clone)]
pub struct SSEIncoherentPart<'a> {
    state: &'a Array1<Complex<f64>>,
    /// Parts from a the stochastic terms
    stochastic: SSEStochasticIncoherentPart,
}

impl<'a> From<SSEParts<'a>> for SSEIncoherentParts<'a> {
    fn from(val: SSEParts<'a>) -> Self {
        SSEIncoherentParts {
            state: val.state,
            stochastic: val
                .stochastic
                .into_iter()
                .map(|p| SSEStochasticIncoherentPart {
                    expectation: p.expectation,
                    l_state: p.l_state,
                })
                .collect(),
        }
    }
}

impl<T: Tensor, U: Tensor> FullNoiseSource<T, U> {
    #[inline]
    fn get_part(&self, state: &Array1<Complex<f64>>, t: f64) -> SSEStochasticPart {
        let SSEStochasticIncoherentPart {
            expectation,
            l_state,
        } = self.get_incoherent_part(state, t);

        let l_dagger_l_state = self.conjugate_operator.dot(&l_state);

        SSEStochasticPart {
            expectation,
            l_state,
            l_dagger_l_state,
        }
    }
    #[inline]
    fn get_incoherent_part(
        &self,
        state: &Array1<Complex<f64>>,
        _t: f64,
    ) -> SSEStochasticIncoherentPart {
        let l_state = self.operator.dot(state);

        assert_eq!(l_state.len(), state.len());
        let expectation = state
            .iter()
            .zip(l_state.iter())
            .fold(Complex::default(), |acc, (s, l)| acc + s.conj() * l);

        SSEStochasticIncoherentPart {
            expectation,
            l_state,
        }
    }
}

impl FullNoise<Array2<Complex<f64>>, Array2<Complex<f64>>> {
    #[must_use]
    pub fn from_operators(operators: &Array3<Complex<f64>>) -> Self {
        Self(
            operators
                .axis_iter(Axis(0))
                .map(|o| FullNoiseSource {
                    operator: o.to_owned(),
                    conjugate_operator: o.map(num_complex::Complex::conj).t().to_owned(),
                })
                .collect(),
        )
    }
}

impl FullNoise<BandedArray<Complex<f64>>, TransposedBandedArray<Complex<f64>>> {
    #[must_use]
    pub fn from_banded(operators: &[BandedArray<Complex<f64>>]) -> Self {
        Self(
            operators
                .iter()
                .map(|o| FullNoiseSource {
                    operator: o.clone(),
                    conjugate_operator: o.transpose().conj(),
                })
                .collect(),
        )
    }
}

impl FullNoise<SplitScatteringArray<Complex<f64>>, SplitScatteringArray<Complex<f64>>> {
    #[must_use]
    pub fn from_split(operators: &[SplitScatteringArray<Complex<f64>>]) -> Self {
        Self(
            operators
                .iter()
                .map(|o| FullNoiseSource {
                    operator: o.clone(),
                    conjugate_operator: o.conj_transpose(),
                })
                .collect(),
        )
    }
}

impl FullNoise<FactorizedArray<Complex<f64>>, FactorizedArray<Complex<f64>>> {
    #[must_use]
    pub fn from_bra_ket(
        amplitudes: Array1<Complex<f64>>,
        bra: &Array2<Complex<f64>>,
        ket: &Array2<Complex<f64>>,
    ) -> Self {
        let sources = amplitudes
            .into_iter()
            .zip(bra.axis_iter(Axis(0)).zip(ket.axis_iter(Axis(0))))
            .map(|(a, (b, k))| FactorizedArray::from_bra_ket(a, b.to_owned(), k.to_owned()))
            .map(|operator| FullNoiseSource {
                conjugate_operator: operator.conj().transpose(),
                operator: operator.clone(),
            })
            .collect::<Vec<_>>();
        Self(sources)
    }
}

impl<T: Dot<Array1<Complex<f64>>, Output = Array1<Complex<f64>>>> Tensor for T {}
/// Represents a noise operator in factorized form
/// `S_n = A_n |Ket_n> <Bra_n|`
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FullNoise<T, U>(Vec<FullNoiseSource<T, U>>);

impl<T: Tensor, U: Tensor> Noise for FullNoise<T, U> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn get_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Vec<SSEStochasticPart> {
        self.0.iter().map(|s| s.get_part(state, t)).collect()
    }

    fn get_incoherent_parts(
        &self,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Vec<SSEStochasticIncoherentPart> {
        self.0
            .iter()
            .map(|s| s.get_incoherent_part(state, t))
            .collect()
    }

    fn get_incoherent_part(
        &self,
        index: usize,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> SSEStochasticIncoherentPart {
        self.0[index].get_incoherent_part(state, t)
    }
}

/// Represents The Stochastic Shrodinger SDE System
///
/// ```latex
/// X_t = X_{t0} + \int_{t0}^{t} a(s,X_s) ds + \sum_1^m \int_{t0}^{t} b^j(s, X_s) dW_s^j
/// ```
///
/// The coherent term
/// ```latex
/// a(s,X) =
/// ```
///
/// The incoherent term
/// ```latex
/// b^j(s, X) = (L_j - <L_j>)|X> where |X> is the wavefunction |\psi>
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SSESystem<H, N> {
    pub hamiltonian: H,
    pub noise: N,
}
impl<H: Tensor, N: Noise> SSESystem<H, N> {
    fn coherent(&self, state: &Array1<Complex<f64>>, _t: f64) -> Array1<Complex<f64>> {
        self.hamiltonian.dot(state)
    }
}

impl<H: Tensor, N: Noise> SDESystem for SSESystem<H, N> {
    #[inline]
    fn n_incoherent(&self) -> usize {
        self.noise.len()
    }

    type Parts<'a> = SSEParts<'a>;
    type IncoherentParts<'a> = SSEIncoherentParts<'a>;
    type CoherentPart<'a> = SSEParts<'a>;
    type IncoherentPart<'a> = SSEIncoherentPart<'a>;

    #[inline]
    fn get_parts<'a>(&self, state: &'a Array1<Complex<f64>>, t: f64) -> Self::Parts<'a> {
        SSEParts {
            state,
            hamiltonian: self.coherent(state, t),
            stochastic: self.noise.get_parts(state, t),
        }
    }
    #[inline]
    fn get_incoherent_part<'a>(
        &self,
        index: usize,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentPart<'a> {
        SSEIncoherentPart {
            state,
            stochastic: self.noise.get_incoherent_part(index, state, t),
        }
    }
    #[inline]
    fn get_incoherent_parts<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentParts<'a> {
        SSEIncoherentParts {
            state,
            stochastic: self.noise.get_incoherent_parts(state, t),
        }
    }
    #[inline]
    fn get_coherent_part<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::CoherentPart<'a> {
        self.get_parts(state, t)
    }

    #[inline]
    fn get_step_from_parts(parts: &Self::Parts<'_>, step: &SDEStep) -> Array1<Complex<f64>> {
        let mut diagonal_coherent = Complex::<f64>::default();
        let mut diagonal_stochastic = Complex::default();
        let mut out = Complex {
            re: step.coherent.im,
            im: -step.coherent.re,
        } * &parts.hamiltonian;

        assert!(parts.stochastic.len() == step.incoherent.len());
        for (part, dw) in parts.stochastic.iter().zip(step.incoherent.iter()) {
            // Terms involving the collapse operator contribute to both the coherent and incoherent part
            // (L <L^\dagger> - 1 / 2 <L^\dagger><L> - 1 / 2 L^\dagger L) * coherent_step + (L - <L>) * incoherent_step_i |\psi>

            // - <L> dw - dt / 2 <L^\dagger><L> |\psi>
            diagonal_coherent -= 0.5 * step.coherent * part.expectation.norm_sqr();
            diagonal_stochastic -= dw * part.expectation;

            // + dt L <L^\dagger> + dw L |\psi>
            // TODO: maybe more accurate in other order
            out += &((dw + (part.expectation.conj() * step.coherent)) * &part.l_state);

            // - (dt / 2) L^\dagger L |\psi>
            out -= &((0.5 * step.coherent) * &part.l_dagger_l_state);
        }

        out += &((diagonal_coherent + diagonal_stochastic) * parts.state);
        out
    }
    #[inline]
    fn get_incoherent_steps_from_parts(
        parts: &Self::IncoherentParts<'_>,
        incoherent_step: &[Complex<f64>],
    ) -> Array1<Complex<f64>> {
        let mut out = Array1::zeros([parts.state.len()]);
        let mut diagonal = Complex::default();

        for (part, step) in parts.stochastic.iter().zip(incoherent_step.iter()) {
            // (L - <L>) * incoherent_step |\psi>
            diagonal -= step * part.expectation;

            out += &(*step * &part.l_state);
        }

        out += &(diagonal * parts.state);
        out
    }
    #[inline]
    fn get_incoherent_step_from_part(
        part: &Self::IncoherentPart<'_>,
        incoherent_step: Complex<f64>,
    ) -> Array1<Complex<f64>> {
        // (L - <L>) * incoherent_step |\psi>
        let mut out = incoherent_step * &part.stochastic.l_state;
        out -= &((incoherent_step * part.stochastic.expectation) * part.state);
        out
    }
    #[inline]
    fn get_coherent_step_from_parts(
        parts: &Self::CoherentPart<'_>,
        coherent_step: Complex<f64>,
    ) -> Array1<Complex<f64>> {
        let mut diagonal = Complex::default();

        let mut out = Complex {
            re: coherent_step.im,
            im: -coherent_step.re,
        } * &parts.hamiltonian;

        for part in &parts.stochastic {
            // Terms involving the collapse operator contribute to the coherent part
            // (L <L^\dagger> - 1 / 2 <L^\dagger><L> - 1 / 2 L^\dagger L) * coherent_step

            // - coherent_step * 1 / 2 <L^\dagger><L> |\psi>
            diagonal -= 0.5 * coherent_step * part.expectation.norm_sqr();

            // + coherent_step L <L^\dagger>  |\psi>
            out += &((part.expectation.conj() * coherent_step) * &part.l_state);
            // - (coherent_step / 2) L^\dagger L |\psi>
            out -= &((0.5 * coherent_step) * &part.l_dagger_l_state);
        }

        out += &(diagonal * parts.state);
        out
    }
    #[inline]
    fn get_operators_from_parts(parts: &Self::Parts<'_>) -> SDEOperators {
        SDEOperators {
            coherent: Self::get_coherent_step_from_parts(parts, Complex { re: 1f64, im: 0f64 }),
            incoherent: parts
                .stochastic
                .iter()
                .map(|p| &p.l_state - (p.expectation * parts.state))
                .collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex;

    use crate::solvers::{EulerStepper, FixedStepSolver, Solver, StateMeasurement};
    use crate::tests::{get_initial_state, get_random_system};

    use super::{FullNoise, SSESystem};

    fn compute_outer_product(
        a: &Array1<Complex<f64>>,
        b: &Array1<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let mut result = Array2::zeros((a.len(), b.len()));
        for (i, val_a) in a.iter().enumerate() {
            for (j, val_b) in b.iter().enumerate() {
                result[[i, j]] = val_a * val_b;
            }
        }
        result
    }

    #[test]
    fn test_diagonal_full_equivalent() {
        // TODO: this should pass actually ...
        let n_states = 10;
        let diagonal_system = get_random_system(0, n_states);
        let shape = [diagonal_system.noise.0.len(), n_states, n_states];
        // TODO mul by amplitude
        let full_operators = Array3::from_shape_vec(
            shape,
            diagonal_system
                .noise
                .0
                .iter()
                .flat_map(|s| -> Vec<Complex<f64>> {
                    compute_outer_product(&s.operator.ket, &s.operator.bra)
                        .into_iter()
                        .collect()
                })
                .collect(),
        )
        .unwrap();
        let full_system = SSESystem {
            hamiltonian: diagonal_system.hamiltonian.clone(),
            noise: FullNoise::from_operators(&full_operators),
        };

        let initial_state = get_initial_state(n_states);

        let n_out = 30;
        let dt = 1f64;
        let diagonal_result = FixedStepSolver {
            stepper: EulerStepper {},
            target_dt: 0.1f64,
        }
        .solve(
            &initial_state,
            &diagonal_system,
            &StateMeasurement {},
            &(0..30).map(|d| f64::from(d) * dt).collect::<Vec<_>>(),
        );
        let result_full = FixedStepSolver {
            stepper: EulerStepper {},
            target_dt: 0.1f64,
        }
        .solve(
            &initial_state,
            &full_system,
            &StateMeasurement {},
            &(0..30).map(|d| f64::from(d) * dt).collect::<Vec<_>>(),
        );

        for i in 0..n_out {
            assert_eq!(result_full[i], diagonal_result[i]);
        }
    }
}
