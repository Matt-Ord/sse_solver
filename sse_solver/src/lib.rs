#![warn(clippy::pedantic)]

pub mod distribution;
pub mod solvers;
pub mod sparse;
pub mod system;

#[cfg(test)]
mod tests {
    type DiagonalNoise = FullNoise<FactorizedArray<Complex<f64>>, FactorizedArray<Complex<f64>>>;

    use ndarray::{linalg::Dot, Array1, Array2};
    use num_complex::{Complex, ComplexFloat};
    use rand::Rng;

    use crate::{
        distribution::StandardComplexNormal,
        solvers::{EulerSolver, Solver, StateMeasurement},
        sparse::{BandedArray, FactorizedArray},
        system::sse::{FullNoise, SSESystem},
    };

    fn get_random_noise(
        n_operators: usize,
        n_states: usize,
    ) -> FullNoise<FactorizedArray<Complex<f64>>, FactorizedArray<Complex<f64>>> {
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
        FullNoise::from_bra_ket(amplitudes, bra, ket)
    }

    pub(crate) fn get_random_system(
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

    pub(crate) fn get_initial_state(n_states: usize) -> Array1<Complex<f64>> {
        let mut state = Array1::zeros([n_states]);
        state[0] = Complex { im: 0f64, re: 1f64 };

        state
    }
    #[test]
    fn test_initial_state_is_initial() {
        let n_states = 10;
        let system = get_random_system(10, n_states);
        let initial_state = get_initial_state(n_states);

        let result = EulerSolver {}.solve(&initial_state, &system, &StateMeasurement {}, 1, 1, 0.0);
        assert_eq!(result[0], initial_state);
    }
    #[test]
    fn test_zero_timestep() {
        let n_states = 10;
        let system = get_diagonal_system(0, n_states);
        let initial_state = get_initial_state(n_states);

        let n_out = 3;
        let result = EulerSolver {}.solve(
            &initial_state,
            &system,
            &StateMeasurement {},
            n_out,
            10,
            0.0,
        );

        for res in result {
            assert_eq!(res, initial_state);
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
