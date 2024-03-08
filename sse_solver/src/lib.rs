#![warn(clippy::pedantic)]

use ndarray::{Array1, Array2, Axis};
use num_complex::{Complex, Complex64};
use rand::prelude::*;
use rand_distr::StandardNormal;

pub trait System {
    fn coherent(&self, state: &Array1<Complex<f64>>, t: f64, dt: f64) -> Array1<Complex<f64>>;

    fn stochastic_euler(
        &self,
        state: &Array1<Complex<f64>>,
        t: f64,
        dt: f64,
    ) -> Array1<Complex<f64>>;
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
            print!("{out}");
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
pub struct DiagonalNoise(Vec<DiagonalNoiseSource>);

impl DiagonalNoise {
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

impl DiagonalNoise {
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

pub struct SSESystem {
    pub hamiltonian: Array2<Complex<f64>>,
    pub noise: DiagonalNoise,
}

impl System for SSESystem {
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

    use ndarray::{s, Array1, Array2};
    use num_complex::Complex;
    use rand::Rng;

    use crate::{DiagonalNoise, EulerSolver, SSESystem, Solver, StandardComplexNormal};

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

    fn get_random_system(n_operators: usize, n_states: usize) -> SSESystem {
        let rng = rand::thread_rng();
        let hamiltonian = Array2::from_shape_vec(
            [n_states, n_states],
            rng.clone()
                .sample_iter(StandardComplexNormal)
                .take(n_operators * n_operators)
                .collect(),
        )
        .unwrap();
        SSESystem {
            noise: get_random_noise(n_operators, n_states),
            hamiltonian,
        }
    }

    fn get_diagonal_system(n_operators: usize, n_states: usize) -> SSESystem {
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
}
