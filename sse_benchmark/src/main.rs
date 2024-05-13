#![feature(test)]
#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]
#![feature(slice_flatten)]

use ndarray::{linalg::Dot, Array1, Array2, Array3};
use num_complex::Complex;
use rand::Rng;
use sse_solver::{
    BandedArray, DiagonalNoise, EulerSolver, FullNoise, SSESystem, Solver, StandardComplexNormal,
};
extern crate test;
use std::simd::{prelude::*, StdFloat};

#[allow(dead_code)]
fn euler_solver_benchmark() {
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
#[allow(dead_code)]
fn euler_solver_benchmark_full() {
    let mut initial_state = Array1::from_elem([31], Complex { im: 0f64, re: 0f64 });
    initial_state[0] = Complex {
        re: 1f64,
        ..Default::default()
    };
    let hamiltonian = Array2::from_elem(
        [initial_state.len(), initial_state.len()],
        Complex { im: 1f64, re: 1f64 },
    );

    let noise_vectors = Array3::from_elem(
        [9, initial_state.len(), initial_state.len()],
        Complex { im: 1f64, re: 1f64 },
    );

    let noise = FullNoise::from_operators(&noise_vectors);
    let system = SSESystem { noise, hamiltonian };
    let n = 100;
    let step = 4000;
    let dt = 0.0001;
    test::black_box(EulerSolver::solve(&initial_state, &system, n, step, dt));
}

#[allow(dead_code)]
fn euler_solver_benchmark_sparse() {
    let mut initial_state = Array1::from_elem([93], Complex { im: 0f64, re: 0f64 });
    initial_state[0] = Complex {
        re: 1f64,
        ..Default::default()
    };

    let hamiltonian = BandedArray::<Complex<f64>>::from_sparse(
        &Vec::from_iter((0..3).map(|_| {
            Vec::from_iter(Array1::<Complex<f64>>::from_elem(
                [initial_state.len()],
                Complex { re: 1f64, im: 3f64 },
            ))
        })),
        &[0, 3, 90],
        &[initial_state.len(), initial_state.len()],
    );

    let noise = FullNoise::from_banded(
        &(0..6)
            .map(|_i| {
                BandedArray::from_sparse(
                    &Vec::from_iter((0..2).map(|_| {
                        Vec::from_iter(Array1::<Complex<f64>>::ones([initial_state.len()]))
                    })),
                    &Vec::from_iter(0..2),
                    &[initial_state.len(), initial_state.len()],
                )
            })
            .collect::<Vec<_>>(),
    );
    let system = SSESystem { noise, hamiltonian };

    let n = 800;
    let step = 8000;
    let dt = 1.25e-17;
    test::black_box(EulerSolver::solve(&initial_state, &system, n, step, dt));
}

fn euler_solver_full_matrix_optimal_benchmark_step(
    state: &Array1<Complex<f64>>,
    h: &Array2<Complex<f64>>,
    n_operators: usize,
    out: &mut Array1<Complex<f64>>,
) {
    for _i in 0..(2 * n_operators + 1) {
        test::black_box(h.dot(state));
    }

    for _i in 0..(2 * n_operators + 2) {
        #[allow(clippy::unit_arg)]
        test::black_box(*out += state);
    }
}

#[allow(dead_code)]
fn euler_solver_full_matrix_optimal_benchmark() {
    let n_states = 31;
    let n_operators = 9;

    let mut initial_state = Array1::from_elem([n_states], Complex { im: 0f64, re: 0f64 });
    initial_state[0] = Complex {
        re: 1f64,
        ..Default::default()
    };

    let mut out = initial_state.clone();

    let hamiltonian = Array2::from_elem(
        [initial_state.len(), initial_state.len()],
        Complex { im: 1f64, re: 1f64 },
    );

    for _n in 0..100 {
        for _m in 0..4000 {
            #[allow(clippy::unit_arg)]
            test::black_box(euler_solver_full_matrix_optimal_benchmark_step(
                &initial_state,
                &hamiltonian,
                n_operators,
                &mut out,
            ));
        }
    }
}

#[allow(dead_code)]
fn complex_array_dot_product() {
    let n = 200;
    let v1 = Array1::from_elem([n], Complex { im: 1f64, re: 1f64 });
    let v2 = Array1::from_elem([n], Complex { im: 1f64, re: 1f64 });
    for n in 0..1000 {
        for _m in 0..1000 {
            for _source in 0..n {
                test::black_box(&v1.dot(&v2));
            }
        }
    }
}

#[allow(dead_code)]
fn complex_array_2d_dot_product() {
    let n = 200;
    let v1 = Array2::from_elem([n, n], Complex { im: 1f64, re: 1f64 });
    let v2 = Array1::from_elem([n], Complex { im: 1f64, re: 1f64 });
    for _n in 0..1000 {
        for _m in 0..1000 {
            test::black_box(&v1.dot(&v2));
        }
    }
}

#[allow(dead_code)]
fn float_array_dot_product() {
    let n = 200;
    let v1 = Array1::from_elem([n], 1f64);
    let v2 = Array1::from_elem([n], 1f64);
    for n in 0..1000 {
        for _m in 0..1000 {
            for _source in 0..n {
                for _i in 0..4 {
                    test::black_box(&v1.dot(&v2));
                }
            }
        }
    }
}
#[allow(dead_code)]
fn float_array_dot_product_vector() {
    let n = 200;
    let v1 = vec![1f64; n];
    let v2 = vec![1f64; n];
    for n in 0..1000 {
        for _m in 0..1000 {
            for _source in 0..n {
                for _i in 0..4 {
                    test::black_box({
                        assert!(v1.len() == v2.len());
                        v1.iter().zip(v2.iter()).map(|(i, j)| i * j).sum::<f64>()
                    });
                }
            }
        }
    }
}
#[inline]
pub fn dot_prod_simd(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len());
    a.array_chunks::<8>()
        .map(|&a| f64x8::from_array(a))
        .zip(b.array_chunks::<8>().map(|&b| f64x8::from_array(b)))
        .fold(f64x8::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

#[inline]
pub fn dot_prod_simd_complex(a: &Complex<Vec<f64>>, b: &Complex<Vec<f64>>) -> Complex<f64> {
    let re_re = dot_prod_simd(&a.re, &b.re);
    let re_im = dot_prod_simd(&a.re, &b.im);
    let im_re = dot_prod_simd(&a.im, &b.re);
    let im_im = dot_prod_simd(&a.im, &b.im);

    Complex {
        re: re_re - im_im,
        im: re_im + im_re,
    }
}
#[inline]
pub fn dot_prod_simd_complex_2(a: &Complex<&Vec<f64>>, b: &Complex<Vec<f64>>) -> Complex<f64> {
    let re_re = dot_prod_simd(a.re, &b.re);
    let re_im = dot_prod_simd(a.re, &b.im);
    let im_re = dot_prod_simd(a.im, &b.re);
    let im_im = dot_prod_simd(a.im, &b.im);

    Complex {
        re: re_re - im_im,
        im: re_im + im_re,
    }
}

#[inline]
pub fn dot_prod_2d_simd_complex(
    a: &Complex<Vec<Vec<f64>>>,
    b: &Complex<Vec<f64>>,
) -> Vec<Complex<f64>> {
    a.re.iter()
        .zip(a.im.iter())
        .map(|(re, im)| dot_prod_simd_complex_2(&Complex { re, im }, b))
        .collect()
}

#[allow(dead_code)]
fn float_array_dot_product_vector_simd() {
    let n = 200;
    let v1 = Complex {
        re: vec![1f64; n],
        im: vec![1f64; n],
    };
    let v2 = Complex {
        re: vec![1f64; n],
        im: vec![1f64; n],
    };
    for n in 0..1000 {
        for _m in 0..1000 {
            for _source in 0..n {
                test::black_box(dot_prod_simd_complex(&v1, &v2));
            }
        }
    }
}

#[allow(dead_code)]
fn float_array_2d_dot_product_vector_simd() {
    let n = 200;
    let v1 = Complex {
        re: vec![vec![1f64; n]; n],
        im: vec![vec![1f64; n]; n],
    };

    let v2 = Complex {
        re: vec![1f64; n],
        im: vec![1f64; n],
    };
    for _n in 0..1000 {
        for _m in 0..1000 {
            test::black_box(dot_prod_2d_simd_complex(&v1, &v2));
        }
    }
}

#[allow(dead_code)]
fn array_complex_dot_product() {
    let n = 200;
    let v1 = Complex {
        re: Array1::from_elem([n], 1f64),
        im: Array1::from_elem([n], 1f64),
    };
    let v2 = Complex {
        re: Array1::from_elem([n], 1f64),
        im: Array1::from_elem([n], 1f64),
    };
    for n in 0..1000 {
        for _m in 0..1000 {
            for _source in 0..n {
                test::black_box({
                    let re_re = v1.re.dot(&v2.re);
                    let re_im = v1.re.dot(&v2.im);
                    let im_re = v1.im.dot(&v2.re);
                    let im_im = v1.im.dot(&v2.im);

                    Complex {
                        re: re_re - im_im,
                        im: re_im + im_re,
                    }
                });
            }
        }
    }
}

// impl<
//         T: num_traits::Zero
//             + Clone
//             + AddAssign
//             + Mul
//             + Copy
//             + std::ops::AddAssign<<T as std::ops::Mul>::Output>,
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
//                 let out_index = (i + offset) % self.shape[0];
//                 out[out_index] += diagonal[i] * rhs_val;
//             }
//         }

//         out
//     }
// }

// impl<
//         T: num_traits::Zero
//             + Clone
//             + AddAssign
//             + Mul
//             + Copy
//             + std::ops::AddAssign<<T as std::ops::Mul>::Output>,
//     > Dot<Array1<T>> for BandedArray<T>
// {
//     type Output = Array1<T>;

//     #[inline]
//     fn dot(&self, rhs: &Array1<T>) -> Self::Output {
//         assert!(self.shape[1] == rhs.len());
//         assert!(self.offsets.len() == self.diagonals.len());

//         let mut out = Array1::zeros(self.shape[0]);

//         for (offset, diagonal) in self.offsets.iter().zip(self.diagonals.iter()) {
//             // let n = (self.shape[1] + offset).div_floor(self.shape[0]) - 1;
//             let mut iter_elem = diagonal.iter().zip(rhs.iter());

//             // Take the first N_0 - offset
//             // These correspond to i=offset..N_0, j=0..N_0-offset
//             (*offset..self.shape[0])
//                 .zip(&mut iter_elem)
//                 .for_each(|(i, (d, r))| out[i] += *d * *r);

//             // In chunks of N_0, starting at N_0-offset
//             // These correspond to i=0..N_0 and some j
//             iter_elem
//                 .zip((0..self.shape[0]).cycle())
//                 .for_each(|((d, r), i)| out[i] += *d * *r);
//         }

//         out
//     }
// }
#[allow(dead_code)]
fn array_dot_benchmark_full() {
    let shape = [100, 100];

    let a = Array2::<Complex<f64>>::ones([shape[0], shape[1]]);
    let b = Array1::<Complex<f64>>::ones([shape[1]]);
    let mut out = Array1::<Complex<f64>>::ones([shape[0]]);
    for _i in 0..10000000 {
        #[allow(clippy::unit_arg)]
        test::black_box(out += &a.dot(&b));
    }
}
#[allow(dead_code)]
fn array_dot_benchmark_array_2() {
    let n_bands = 3;
    let shape = [100, 100];

    let a = Array2::<Complex<f64>>::ones([n_bands, shape[1]]);
    let b = Array1::<Complex<f64>>::ones([shape[1]]);
    let mut out = Array1::<Complex<f64>>::ones([shape[0]]);
    for _i in 0..10000000 {
        #[allow(clippy::unit_arg)]
        test::black_box(out += a.dot(&b)[0]);
    }
}

#[allow(dead_code)]
fn banded_array_dot_benchmark_approx() {
    let n_bands = 3;
    let shape = [100, 100];

    let a = Array1::<Complex<f64>>::ones([shape[1]]);
    let b = Array1::<Complex<f64>>::ones([shape[1]]);
    let mut out = Array1::<Complex<f64>>::ones([shape[1]]);
    for _i in 0..100000000 {
        for _j in 0..n_bands {
            #[allow(clippy::unit_arg)]
            test::black_box(out += a.dot(&b));
        }
    }
}

#[allow(dead_code)]
fn banded_array_dot_benchmark() {
    let n_bands = 3;
    let shape = [100, 100];

    let a = BandedArray::<Complex<f64>>::from_sparse(
        &Vec::from_iter(
            (0..n_bands).map(|_| Vec::from_iter(Array1::<Complex<f64>>::ones([shape[1]]))),
        ),
        &Vec::from_iter((shape[0] - n_bands)..shape[0]),
        &shape,
    );
    let b = Array1::<Complex<f64>>::ones([shape[1]]);
    for _i in 0..100000000 {
        #[allow(clippy::unit_arg)]
        test::black_box(&a.dot(&b));
    }
}

#[allow(dead_code)]
fn banded_array_transposed_dot_benchmark() {
    let n_bands = 3;
    let shape = [100, 100];

    let a = BandedArray::<Complex<f64>>::from_sparse(
        &Vec::from_iter(
            (0..n_bands).map(|_| Vec::from_iter(Array1::<Complex<f64>>::ones([shape[1]]))),
        ),
        &Vec::from_iter(0..n_bands),
        &shape,
    )
    .transpose();
    let b = Array1::<Complex<f64>>::ones([shape[1]]);
    for _i in 0..100000000 {
        #[allow(clippy::unit_arg)]
        test::black_box(&a.dot(&b));
    }
}

#[allow(dead_code)]
#[inline(never)]
fn mul_bench(
    lhs: Array1<Complex<f64>>,
    rhs: Array1<Complex<f64>>,
    n: usize,
) -> Array1<Complex<f64>> {
    let mut out = Array1::zeros(lhs.shape()[0]);
    for _i in 0..n {
        #[allow(clippy::unit_arg)]
        test::black_box(out += &(&lhs + &rhs));
    }
    out
}

#[allow(dead_code)]
fn mul_rand_array() {
    let rng = rand::thread_rng();

    let lhs: Array1<Complex<f64>> = rng
        .clone()
        .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
        .take(10000)
        .map(|d| d * Complex { re: 1f64, im: 1f64 })
        .collect();

    let rhs: Array1<Complex<f64>> = rng
        .clone()
        .sample_iter::<Complex<f64>, _>(StandardComplexNormal)
        .take(10000)
        .map(|d| d * Complex { re: 1f64, im: 1f64 })
        .collect();

    test::black_box(mul_bench(lhs, rhs, test::black_box(500000)));
}
fn main() {
    euler_solver_benchmark_sparse()
}
