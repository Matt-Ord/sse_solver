#![warn(clippy::pedantic)]

use ndarray::{linalg::Dot, Array1, Array2, Array3, Axis};
use num_complex::{Complex, Complex64};
use rand::prelude::*;
use rand_distr::{num_traits, StandardNormal};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents an array, stored as a series of (offset) diagonals
/// Each diagonal stores elements M_{i+offset % `N_0`, i}
/// length of diagonals is shape[1], with a total of shape[0] offsets
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BandedArray<T> {
    diagonals: Vec<Vec<T>>,
    offsets: Vec<usize>,
    shape: [usize; 2],
}

impl<T: Copy> BandedArray<T> {
    #[must_use]
    pub fn from_dense(dense: &Array2<T>) -> Self {
        let offsets = (0..dense.shape()[0]).collect::<Vec<_>>();
        let diagonals = offsets
            .iter()
            .map(|o| {
                (0..dense.shape()[1])
                    .map(|i| dense[[(i + o) % dense.shape()[0], i]])
                    .collect::<Vec<_>>()
            })
            .collect();

        BandedArray {
            diagonals,
            offsets,
            shape: [dense.shape()[0], dense.shape()[1]],
        }
    }
    /// # Panics
    ///
    /// Will panic if diagonals are not of length shape[1]
    /// Will panic if len(diagonals) !== len(offsets)
    #[must_use]
    pub fn from_sparse(diagonals: &[Vec<T>], offsets: &[usize], shape: &[usize; 2]) -> Self {
        for d in diagonals {
            assert_eq!(d.len(), shape[1]);
        }
        let diagonals_vec = diagonals.to_vec();
        let offsets_vec = offsets.to_vec();
        assert_eq!(diagonals_vec.len(), offsets_vec.len());
        BandedArray {
            diagonals: diagonals_vec,
            offsets: offsets_vec,
            shape: shape.to_owned(),
        }
    }

    #[must_use]
    pub fn transpose(&self) -> TransposedBandedArray<T> {
        TransposedBandedArray {
            diagonals: self.diagonals.clone(),
            offsets: self.offsets.clone(),
            shape: [self.shape[1], self.shape[0]],
        }
    }
}
impl<T: num_complex::ComplexFloat> TransposedBandedArray<T> {
    fn conj(&self) -> TransposedBandedArray<T> {
        TransposedBandedArray {
            diagonals: self
                .diagonals
                .iter()
                .map(|d| d.iter().map(|i| i.conj()).collect())
                .collect(),
            offsets: self.offsets.clone(),
            shape: self.shape,
        }
    }
}

// impl<
//         T: num_traits::Zero
//             + Clone
//             + Copy
//             + std::ops::AddAssign<<T as std::ops::Mul>::Output>
//             + std::ops::Mul,
//     > Dot<Array1<T>> for BandedArray<T>
// {
//     type Output = Array1<T>;

//     #[inline]
//     fn dot(&self, rhs: &Array1<T>) -> Self::Output {
//         assert!(self.shape[1] == rhs.len());
//         assert!(self.offsets.len() == self.diagonals.len());

//         let mut out = Array1::zeros(self.shape[0]);

//         for (offset, diagonal) in self.offsets.iter().zip(self.diagonals.iter()) {
//             for (i, &rhs_val) in rhs.iter().enumerate() {
//                 let out_idx = (i + offset) % self.shape[0];
//                 out[out_idx] += diagonal[i] * rhs_val;
//             }
//         }

//         out
//     }
// }

impl<
        T: num_traits::Zero
            + Clone
            + Copy
            + std::ops::AddAssign<<T as std::ops::Mul>::Output>
            + std::ops::Mul,
    > Dot<Array1<T>> for BandedArray<T>
{
    type Output = Array1<T>;

    #[inline]
    fn dot(&self, rhs: &Array1<T>) -> Self::Output {
        assert!(self.shape[1] == rhs.len());
        assert!(self.offsets.len() == self.diagonals.len());

        let mut out = Array1::zeros(self.shape[0]);

        for (offset, diagonal) in self.offsets.iter().zip(self.diagonals.iter()) {
            let mut iter_elem = diagonal.iter().zip(rhs.iter());

            // Take the first N_0 - offset
            // These correspond to i=offset..N_0, j=0..N_0-offset
            (*offset..self.shape[0])
                .zip(&mut iter_elem)
                .for_each(|(i, (d, r))| out[i] += *d * *r);

            // In chunks of N_0, starting at N_0-offset
            // These correspond to i=0..N_0 and some j starting at N_0-offset
            iter_elem
                .zip((0..self.shape[0]).cycle())
                .for_each(|((d, r), i)| out[i] += *d * *r);
        }

        out
    }
}

/// Represents an array, stored as a series of (offset) diagonals
/// Each diagonal stores elements M_{i, i+offset % `N_0`}
/// length of diagonals is shape[0], with a total of shape[1] offsets
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransposedBandedArray<T> {
    diagonals: Vec<Vec<T>>,
    offsets: Vec<usize>,
    shape: [usize; 2],
}
// impl<
//         T: num_traits::Zero
//             + Clone
//             + Copy
//             + std::ops::AddAssign<<T as std::ops::Mul>::Output>
//             + std::ops::Mul,
//     > Dot<Array1<T>> for TransposedBandedArray<T>
// {
//     type Output = Array1<T>;

//     #[inline]
//     fn dot(&self, rhs: &Array1<T>) -> Self::Output {
//         assert!(self.shape[1] == rhs.len());
//         assert!(self.offsets.len() == self.diagonals.len());

//         let mut out = Array1::zeros(self.shape[0]);

//         for (offset, diagonal) in self.offsets.iter().zip(self.diagonals.iter()) {
//             for (i, &diag_val) in diagonal.iter().enumerate() {
//                 let rhs_idx = (i + offset) % self.shape[1];
//                 out[i] += diag_val * rhs[rhs_idx];
//             }
//         }

//         out
//     }
// }

impl<
        T: num_traits::Zero
            + Clone
            + Copy
            + std::ops::AddAssign<<T as std::ops::Mul>::Output>
            + std::ops::Mul,
    > Dot<Array1<T>> for TransposedBandedArray<T>
{
    type Output = Array1<T>;

    #[inline]
    fn dot(&self, rhs: &Array1<T>) -> Self::Output {
        assert!(self.shape[1] == rhs.len());
        assert!(self.offsets.len() == self.diagonals.len());

        let mut out = Array1::zeros(self.shape[0]);

        for (offset, diagonal) in self.offsets.iter().zip(self.diagonals.iter()) {
            let mut iter_elem = diagonal.iter().zip(out.iter_mut());

            // Take the first N_1 - offset
            // These correspond to j=offset..N_1, i=0..N_0-offset
            (*offset..self.shape[1])
                .zip(&mut iter_elem)
                .for_each(|(i, (d, o))| *o += *d * rhs[i]);

            // In chunks of N_0, starting at N_0-offset
            // These correspond to i=0..N_0 and some j
            iter_elem
                .zip((0..self.shape[1]).cycle())
                .for_each(|((d, o), i)| *o += *d * rhs[i]);
        }

        out
    }
}
pub trait System {
    fn coherent(&self, state: &Array1<Complex<f64>>, t: f64, dt: f64) -> Array1<Complex<f64>>;

    fn stochastic_euler(
        &self,
        state: &Array1<Complex<f64>>,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>>;
}

pub trait Noise {
    fn euler_step(&self, state: &Array1<Complex<f64>>, dt: f64) -> Array1<Complex<f64>>;
}

pub trait Solver<T: System> {
    fn step(state: &Array1<Complex<f64>>, system: &T, t: f64, dt: f64) -> Array1<Complex<f64>>;

    fn integrate(
        state: &Array1<Complex<f64>>,
        system: &T,
        t_start: f64,
        n_step: usize,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        let mut out = state.clone();
        let mut current_t = t_start.to_owned();
        for _n in 0..n_step {
            out = Self::step(&out, system, current_t, dt);
            current_t += dt;
        }
        out
    }
    #[allow(clippy::cast_precision_loss)]
    fn solve(
        initial_state: &Array1<Complex<f64>>,
        system: &T,
        n: usize,
        step: usize,
        dt: f64,
    ) -> Array2<Complex<f64>> {
        let mut out = Array2::zeros([0, initial_state.len()]);
        let mut current = initial_state.to_owned();
        let mut current_t = 0f64;
        for _n in 1..n {
            out.push_row(current.view()).unwrap();
            current = Self::integrate(&current, system, current_t, step, dt);
            current_t += dt * step as f64;
        }
        out.push_row(current.view()).unwrap();

        out
    }
}

pub struct EulerSolver {}

impl<T: System> Solver<T> for EulerSolver {
    fn step(state: &Array1<Complex<f64>>, system: &T, t: f64, dt: f64) -> Array1<Complex<f64>> {
        let mut out = system.coherent(state, t, dt);

        out += &system.stochastic_euler(state, t, dt);
        out
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct DiagonalNoiseSource {
    amplitude: Complex<f64>,
    // LHS of the factorized operators.
    // This is in 'bra' form, so bra.dot(state) === <bra|state>
    bra: Array1<Complex<f64>>,
    // This is in 'ket' form, so conj(conj_bra).dot(state) === <bra|state>
    conj_bra: Array1<Complex<f64>>,
    // RHS of the factorized operators.
    // This is in 'ket' form, so conj(ket).dot(state) === <ket|state>
    ket: Array1<Complex<f64>>,
    // RHS of the factorized operators.
    // This is in 'bra' form, so conj_ket.dot(state) === <ket|state>
    conj_ket: Array1<Complex<f64>>,
}

struct EulerStep {
    diagonal_amplitude: Complex<f64>,
    off_diagonal: Array1<Complex<f64>>,
}

impl EulerStep {
    fn resolve(self, state: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
        // Also add on initial state ...
        self.off_diagonal + ((self.diagonal_amplitude + 1f64) * state)
    }
}
impl DiagonalNoiseSource {
    #[inline]
    fn apply_bra_to(&self, state: &Array1<Complex<f64>>) -> Complex<f64> {
        self.bra.dot(state)
    }

    #[inline]
    fn apply_ket_to(&self, state: &Array1<Complex<f64>>) -> Complex<f64> {
        self.conj_ket.dot(state)
    }
    #[inline]
    fn accumulate_euler_step(&self, step: &mut EulerStep, state: &Array1<Complex<f64>>, dt: f64) {
        let mut rng = rand::thread_rng();
        let noise: Complex<f64> = rng.sample(StandardComplexNormal);

        let amplitude = &self.amplitude;
        let applied_bra = self.apply_bra_to(state);
        let applied_ket = self.apply_ket_to(state);
        let expectation = (applied_bra * applied_ket.conj() * amplitude).re;

        assert!(self.ket.len() == self.conj_bra.len());
        assert!(step.off_diagonal.len() == self.conj_bra.len());

        let prefactor = applied_bra * dt;

        step.off_diagonal += &((&self.ket * (prefactor * (noise + expectation)))
            - (&self.conj_bra * (prefactor * (0.5 * amplitude.norm_sqr()))));

        // TODO: this may be quicker if we can do unchecked indexing
        // for i in 0..step.off_diagonal.len() {
        //     let k = self.ket[i];
        //     let b = self.conj_bra[i];
        //     step.off_diagonal[i] +=
        //         (k * (noise + expectation) - b * (0.5 * amplitude.norm_sqr())) * prefactor;
        // }

        step.diagonal_amplitude -= (expectation * (0.5 * expectation + noise)) * dt;
    }
}

/// Represents a noise operator in factorized form
/// `S_n = A_n |Ket_n> <Bra_n|`
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiagonalNoise(Vec<DiagonalNoiseSource>);

impl DiagonalNoise {
    #[must_use]
    pub fn amplitudes(&self) -> Array1<Complex<f64>> {
        self.0.iter().map(|s| s.amplitude).collect()
    }
    #[must_use]
    pub fn from_bra_ket(
        amplitudes: Array1<Complex<f64>>,
        bra: &Array2<Complex<f64>>,
        ket: &Array2<Complex<f64>>,
    ) -> Self {
        let mut sources = Vec::with_capacity(amplitudes.len());

        amplitudes
            .into_iter()
            .zip(bra.axis_iter(Axis(0)).zip(ket.axis_iter(Axis(0))))
            .map(|(a, (b, k))| DiagonalNoiseSource {
                amplitude: a,
                conj_bra: b.map(num_complex::Complex::conj),
                conj_ket: k.map(num_complex::Complex::conj),
                bra: b.to_owned(),
                ket: k.to_owned(),
            })
            .for_each(|s| sources.push(s));
        Self(sources)
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

impl<T: Tensor, U: Tensor> FullNoiseSource<T, U> {
    #[inline]
    fn accumulate_euler_step(&self, step: &mut EulerStep, state: &Array1<Complex<f64>>, dt: f64) {
        // Using the conventions from https://doi.org/10.1103/PhysRevA.66.012108
        // with gamma = 1
        // d |\psi> = -i dt H |\psi>
        // + (<L^\dagger>dt + dw) L|\psi>
        // - (dt / 2) L^\dagger L |\psi>
        // - (dt / 2 <L^\dagger><L> + <L> dw) |\psi>
        let mut rng = rand::thread_rng();
        let dw = rng.sample::<Complex<f64>, _>(StandardComplexNormal) * dt.sqrt();

        let l_state = self.operator.dot(state);
        let l_dagger_l_state = self.conjugate_operator.dot(&l_state);

        let mut expectation = Complex::default();
        // Todo assert etc to improve perf
        for i in 0..state.len() {
            expectation += state[i].conj() * l_state[i];
        }

        // (<L^\dagger>dt + dw) L|\psi>
        // - (dt / 2) L^\dagger L |\psi>
        step.off_diagonal +=
            &((&l_state * (dw + dt * expectation.conj())) - (&l_dagger_l_state * (dt * 0.5)));

        // - (dt / 2 <L^\dagger><L> + <L> dw) |\psi>
        step.diagonal_amplitude -= 0.5 * expectation.norm_sqr() * dt + expectation * dw;
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

pub trait Tensor: Dot<Array1<Complex<f64>>, Output = Array1<Complex<f64>>> {}

impl<T: Dot<Array1<Complex<f64>>, Output = Array1<Complex<f64>>>> Tensor for T {}
/// Represents a noise operator in factorized form
/// `S_n = A_n |Ket_n> <Bra_n|`
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FullNoise<T: Tensor, U: Tensor>(Vec<FullNoiseSource<T, U>>);

pub struct StandardComplexNormal;

impl Distribution<Complex<f32>> for StandardComplexNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f32> {
        let re = rng.sample::<f32, StandardNormal>(StandardNormal) / std::f32::consts::SQRT_2;
        let im = rng.sample::<f32, StandardNormal>(StandardNormal) / std::f32::consts::SQRT_2;
        Complex { re, im }
    }
}

impl Distribution<Complex<f64>> for StandardComplexNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        let re = rng.sample::<f64, StandardNormal>(StandardNormal) / std::f64::consts::SQRT_2;
        let im = rng.sample::<f64, StandardNormal>(StandardNormal) / std::f64::consts::SQRT_2;
        Complex { re, im }
    }
}

impl Noise for DiagonalNoise {
    #[inline]
    fn euler_step(&self, state: &Array1<Complex<f64>>, dt: f64) -> Array1<Complex<f64>> {
        let mut step = EulerStep {
            diagonal_amplitude: Complex64::default(),
            off_diagonal: Array1::zeros(state.shape()[0]),
        };

        for source in &self.0 {
            source.accumulate_euler_step(&mut step, state, dt);
        }

        step.resolve(state)
    }
}

impl<T: Tensor, U: Tensor> Noise for FullNoise<T, U> {
    #[inline]
    fn euler_step(&self, state: &Array1<Complex<f64>>, dt: f64) -> Array1<Complex<f64>> {
        let mut step = EulerStep {
            diagonal_amplitude: Complex64::default(),
            off_diagonal: Array1::zeros(state.shape()[0]),
        };

        for source in &self.0 {
            source.accumulate_euler_step(&mut step, state, dt);
        }

        step.resolve(state)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SSESystem<H: Tensor, N: Noise> {
    pub hamiltonian: H,
    pub noise: N,
}

impl<H: Tensor, N: Noise> System for SSESystem<H, N> {
    fn coherent(&self, state: &Array1<Complex<f64>>, _t: f64, dt: f64) -> Array1<Complex<f64>> {
        self.hamiltonian.dot(state) * Complex { re: 0f64, im: -dt }
    }
    #[inline]
    fn stochastic_euler(
        &self,
        state: &Array1<Complex<f64>>,
        _t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>> {
        self.noise.euler_step(state, dt)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::{linalg::Dot, s, Array1, Array2, Array3};
    use num_complex::{Complex, ComplexFloat};
    use rand::Rng;

    use crate::{
        BandedArray, DiagonalNoise, EulerSolver, FullNoise, SSESystem, Solver,
        StandardComplexNormal,
    };

    fn get_random_noise(n_operators: usize, n_states: usize) -> DiagonalNoise {
        let rng = rand::thread_rng();
        // let noise: Complex<f64> = rng.sample(StandardComplexNormal);
        let amplitudes = Array1::from_iter(
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_operators),
        );
        let bra = &Array2::from_shape_vec(
            [n_operators, n_states],
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_operators * n_states)
                .collect(),
        )
        .unwrap();
        let ket = &Array2::from_shape_vec(
            [n_operators, n_states],
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_operators * n_states)
                .collect(),
        )
        .unwrap();
        DiagonalNoise::from_bra_ket(amplitudes, bra, ket)
    }

    fn get_random_system(
        n_operators: usize,
        n_states: usize,
    ) -> SSESystem<Array2<Complex<f64>>, DiagonalNoise> {
        let rng = rand::thread_rng();
        let hamiltonian = Array2::from_shape_vec(
            [n_states, n_states],
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_states * n_states)
                .collect(),
        )
        .unwrap();
        SSESystem {
            noise: get_random_noise(n_operators, n_states),
            hamiltonian,
        }
    }

    fn get_diagonal_system(
        n_operators: usize,
        n_states: usize,
    ) -> SSESystem<Array2<Complex<f64>>, DiagonalNoise> {
        let rng = rand::thread_rng();
        let hamiltonian = Array2::from_diag(&Array1::from_iter(
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_states),
        ));
        SSESystem {
            noise: get_random_noise(n_operators, n_states),
            hamiltonian,
        }
    }

    fn get_initial_state(n_states: usize) -> Array1<Complex<f64>> {
        let mut state = Array1::zeros([n_states]);
        state[0] = Complex { im: 0f64, re: 1f64 };

        state
    }
    #[test]
    fn test_initial_state_is_initial() {
        let n_states = 10;
        let system = get_random_system(10, n_states);
        let initial_state = get_initial_state(n_states);

        let result = EulerSolver::solve(&initial_state, &system, 1, 1, 0.0);
        assert_eq!(result.slice(s![0, ..]), initial_state);
    }
    #[test]
    fn test_zero_timestep() {
        let n_states = 10;
        let system = get_diagonal_system(0, n_states);
        let initial_state = get_initial_state(n_states);

        let n_out = 3;
        let result = EulerSolver::solve(&initial_state, &system, n_out, 10, 0.0);

        for i in 0..n_out {
            assert_eq!(result.slice(s![i, ..]), initial_state);
        }
    }
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
                    compute_outer_product(&s.ket, &s.bra).into_iter().collect()
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
    #[test]
    fn test_banded_dot_product() {
        let rng = rand::thread_rng();
        let shape = [10, 100];

        let full = Array2::from_shape_vec(
            shape,
            rng.clone()
                .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
                .take(shape[0] * shape[1])
                .collect(),
        )
        .unwrap();
        let banded = BandedArray::from_dense(&full);

        let state = Array1::from_iter(
            rng.clone()
                .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
                .take(shape[1]),
        );

        let expected = full.dot(&state);
        let actual = banded.dot(&state);
        for i in 0..shape[0] {
            assert!((expected[i] - actual[i]).abs() < 1e-8);
        }
        assert_eq!(expected.len(), actual.len());
    }

    #[test]
    fn test_banded_transposed_dot_product() {
        let rng = rand::thread_rng();
        let shape = [100, 10];

        let full = Array2::from_shape_vec(
            [shape[1], shape[0]],
            rng.clone()
                .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
                .take(shape[0] * shape[1])
                .collect(),
        )
        .unwrap();
        let banded = BandedArray::from_dense(&full);

        let state = Array1::from_iter(
            rng.clone()
                .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
                .take(shape[1]),
        );

        let expected = full.reversed_axes().dot(&state);
        let actual = banded.transpose().dot(&state);

        for i in 0..shape[0] {
            assert!((expected[i] - actual[i]).abs() < 1e-8);
        }
        assert_eq!(expected.len(), actual.len());
    }
}
