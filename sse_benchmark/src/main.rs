#![feature(test)]
#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]
#![feature(slice_flatten)]
use ndarray::{Array1, Array2};
use num_complex::Complex;
use sse_solver::{DiagonalNoise, EulerSolver, SSESystem, Solver};
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

fn main() {
    euler_solver_benchmark()
}
