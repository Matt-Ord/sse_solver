use ndarray::{linalg::Dot, Array1, Array2, Array3, Axis};
use num_complex::Complex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    sparse::{BandedArray, FactorizedArray, TransposedBandedArray},
    system::{SDEStep, SDESystem},
};

pub trait Noise {
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;

    fn get_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Vec<SSEStochasticPart>;

    fn get_incoherent_parts(
        &self,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Vec<SSEStochasticIncoherentPart>;
}

struct SSEStepOffDiagonal<'a> {
    amplitude: Complex<f64>,
    direction: &'a Array1<Complex<f64>>,
}
struct SSEStepSummation<'a> {
    diagonal: Complex<f64>,
    off_diagonal: Vec<SSEStepOffDiagonal<'a>>,
}

impl<'a> SSEStepSummation<'a> {
    #[inline]
    fn resolve(self, state: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
        let mut out = self.diagonal * state;
        self.off_diagonal
            .into_iter()
            .for_each(|d| out += &(d.amplitude * d.direction));
        out
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct FullNoiseSource<T: Tensor, U: Tensor> {
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

pub struct SSEParts {
    state: Array1<Complex<f64>>,
    /// H |\psi>
    hamiltonian: Array1<Complex<f64>>,
    /// Parts from a the stochastic terms
    stochastic: Vec<SSEStochasticPart>,
}

pub struct SSEStochasticPart {
    /// <L>
    expectation: Complex<f64>,
    /// L |\psi>
    l_state: Array1<Complex<f64>>,
    /// L^\dagger L |\psi>
    l_dagger_l_state: Array1<Complex<f64>>,
}

pub struct SSEIncoherentParts {
    state: Array1<Complex<f64>>,
    /// Parts from a the stochastic terms
    stochastic: Vec<SSEStochasticIncoherentPart>,
}

pub struct SSEStochasticIncoherentPart {
    /// <L>
    expectation: Complex<f64>,
    /// L |\psi>
    l_state: Array1<Complex<f64>>,
}

impl From<SSEParts> for SSEIncoherentParts {
    fn from(val: SSEParts) -> Self {
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
    fn get_part(&self, state: &Array1<Complex<f64>>, _t: f64) -> SSEStochasticPart {
        let l_state = self.operator.dot(state);
        let l_dagger_l_state = self.conjugate_operator.dot(&l_state);
        let mut expectation = Complex::default();
        // Todo assert etc to improve perf
        for i in 0..state.len() {
            expectation += state[i].conj() * l_state[i];
        }
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
        let mut expectation = Complex::default();
        // Todo assert etc to improve perf
        for i in 0..state.len() {
            expectation += state[i].conj() * l_state[i];
        }
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
                    conjugate_operator: o.map(num_complex::Complex::conj).reversed_axes(),
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

pub trait Tensor: Dot<Array1<Complex<f64>>, Output = Array1<Complex<f64>>> {}

impl<T: Dot<Array1<Complex<f64>>, Output = Array1<Complex<f64>>>> Tensor for T {}
/// Represents a noise operator in factorized form
/// `S_n = A_n |Ket_n> <Bra_n|`
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FullNoise<T: Tensor, U: Tensor>(Vec<FullNoiseSource<T, U>>);

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
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SSESystem<H: Tensor, N: Noise> {
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

    #[inline]
    fn get_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Self::Parts {
        SSEParts {
            state: state.to_owned(),
            hamiltonian: self.coherent(state, t),
            stochastic: self.noise.get_parts(state, t),
        }
    }

    type Parts = SSEParts;
    #[inline]
    fn apply_step_from_parts(out: &mut Array1<Complex<f64>>, parts: &Self::Parts, step: &SDEStep) {
        let mut sse_step = SSEStepSummation {
            diagonal: Complex { re: 0f64, im: 0f64 },
            off_diagonal: Vec::with_capacity(1 + 2 * &parts.stochastic.len()),
        };

        sse_step.off_diagonal.push(SSEStepOffDiagonal {
            amplitude: Complex {
                re: step.coherent.im,
                im: -step.coherent.re,
            },
            direction: &parts.hamiltonian,
        });

        for (part, dw) in parts.stochastic.iter().zip(step.incoherent.iter()) {
            // Terms involving the collapse operator contribute to both the coherent and incoherent part
            // (L <L^\dagger> - 1 / 2 <L^\dagger><L> - 1 / 2 L^\dagger L) * coherent_step + (L - <L>) * incoherent_step_i |\psi>

            // - <L> dw - dt / 2 <L^\dagger><L> |\psi>
            sse_step.diagonal -=
                dw * part.expectation + 0.5 * step.coherent * part.expectation.norm_sqr();

            // + dt L <L^\dagger> + dw L |\psi>
            sse_step.off_diagonal.push(SSEStepOffDiagonal {
                amplitude: (dw + part.expectation.conj() * step.coherent),
                direction: &part.l_state,
            });
            // - (dt / 2) L^\dagger L |\psi>
            sse_step.off_diagonal.push(SSEStepOffDiagonal {
                amplitude: -0.5 * step.coherent,
                direction: &part.l_dagger_l_state,
            });
        }

        *out += &sse_step.resolve(&parts.state);
    }

    type IncoherentParts = SSEIncoherentParts;

    fn get_incoherent_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Self::IncoherentParts {
        SSEIncoherentParts {
            state: state.to_owned(),
            stochastic: self.noise.get_incoherent_parts(state, t),
        }
    }

    fn apply_incoherent_step_from_parts(
        out: &mut Array1<Complex<f64>>,
        parts: &Self::IncoherentParts,
        incoherent_step: &[Complex<f64>],
    ) {
        let mut sse_step = SSEStepSummation {
            diagonal: Complex { re: 0f64, im: 0f64 },
            off_diagonal: Vec::with_capacity(parts.stochastic.len()),
        };

        for (part, incoherent_step) in parts.stochastic.iter().zip(incoherent_step.iter()) {
            // (L - <L>) * incoherent_step |\psi>
            sse_step.diagonal -= incoherent_step * part.expectation;

            sse_step.off_diagonal.push(SSEStepOffDiagonal {
                amplitude: *incoherent_step,
                direction: &part.l_state,
            });
        }

        *out += &sse_step.resolve(&parts.state);
    }
}

#[cfg(test)]
mod test {
    use ndarray::{s, Array1, Array2, Array3};
    use num_complex::Complex;

    use crate::solvers::{EulerSolver, Solver};
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
        let diagonal_result = EulerSolver::solve(&initial_state, &diagonal_system, n_out, 10, dt);
        let result_full = EulerSolver::solve(&initial_state, &full_system, n_out, 10, dt);

        for i in 0..n_out {
            assert_eq!(
                result_full.slice(s![i, ..]),
                diagonal_result.slice(s![i, ..])
            );
        }
    }
}