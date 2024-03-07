#![feature(test)]
use ndarray::{Array1, Array2};
use num_complex::Complex;
use sse_solver::{DiagonalNoise, EulerSolver, SSESystem, Solver};
extern crate test;

fn main() {
    // setup_global_subscriber();

    let mut initial_state = Array1::from_elem([200], Complex { im: 0f64, re: 0f64 });
    initial_state[0] = Complex {
        re: 1f64,
        ..Default::default()
    };
    let amplitudes = vec![Complex::default(); 200];
    let hamiltonian = Array2::from_elem(
        [initial_state.len(), initial_state.len()],
        Complex { im: 1f64, re: 1f64 },
    );

    let noise_vectors = Array2::from_elem(
        [amplitudes.len(), initial_state.len()],
        Complex { im: 1f64, re: 1f64 },
    );

    let noise = DiagonalNoise::from_bra_ket(amplitudes.into(), &noise_vectors, &noise_vectors);
    let system = SSESystem { noise, hamiltonian };
    let n = 1000;
    let step = 1000;
    let dt = 0.0001;
    test::black_box(EulerSolver::solve(&initial_state, &system, n, step, dt));
}
