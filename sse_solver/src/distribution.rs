use ndarray::Array2;
use num_complex::Complex;
use rand::Rng;
use rand_distr::{weighted::WeightedIndex, Distribution, StandardNormal};

/// The Standard Normal distribution for a complex number
/// ``<dWi dWj*> = delta_ij``
pub struct StandardComplexNormal;

impl Distribution<Complex<f32>> for StandardComplexNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f32> {
        let re = rng.sample::<f32, _>(StandardNormal) / std::f32::consts::SQRT_2;
        let im = rng.sample::<f32, _>(StandardNormal) / std::f32::consts::SQRT_2;
        Complex { re, im }
    }
}

impl Distribution<Complex<f64>> for StandardComplexNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        let re = rng.sample::<f64, _>(StandardNormal) / std::f64::consts::SQRT_2;
        let im = rng.sample::<f64, _>(StandardNormal) / std::f64::consts::SQRT_2;
        Complex { re, im }
    }
}

/// The V distribution for n incoherent operators, according to eqn 14.2.8 - 14.2.10
/// in 10.1007/978-3-662-12616-5
/// P(V_{ij} = \pm dt) = 0.5
/// V_{k,k} = -dt
/// V_{i,j} = -V_{j,i}
pub struct V {
    pub n: usize,
    pub dt: f64,
}

impl Distribution<Array2<f64>> for V {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array2<f64> {
        let mut out = Array2::zeros([self.n, self.n]);

        let options = [self.dt, -self.dt];

        for i in 0..self.n {
            for j in 0..i {
                let choice = rng.random::<bool>();
                out[[i, j]] = if choice { options[0] } else { options[1] };
                out[[j, i]] = if choice { options[1] } else { options[0] };
            }
            out[[i, i]] = options[1];
        }

        out
    }
}

/// A general distribution with N fixed values
/// sampled by a given weight
pub struct NPoint<const N: usize, T> {
    pub weights: WeightedIndex<i32>,
    pub values: [T; N],
}

impl<const N: usize, T: Copy> Distribution<T> for NPoint<N, T> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        self.values[self.weights.sample(rng)]
    }
}

/// The W distribution, according to eqn 14.2.4
/// in 10.1007/978-3-662-12616-5
/// P(W = \pm \sqrt{3dt}) = 1/6
/// P(W = 0) = 1/3
pub struct ThreePointW(NPoint<3, f64>);

impl Distribution<f64> for ThreePointW {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.0.sample(rng)
    }
}

impl ThreePointW {
    #[must_use]
    pub fn new(dt: f64) -> Self {
        let w_plus = (3.0 * dt).sqrt();
        Self(NPoint {
            // Safety: [1, 1, 4] are valid weights
            weights: unsafe { WeightedIndex::new([1, 1, 4]).unwrap_unchecked() },
            values: [w_plus, -w_plus, 0.0],
        })
    }
}

/// The Complex W distribution, as an extension of eqn 14.2.4
/// in 10.1007/978-3-662-12616-5
///
/// we extend this
/// P(W = \pm \sqrt{3dt}) = 1/6
/// P(W = 0) = 1/3
pub struct NinePointComplexW(NPoint<9, Complex<f64>>);

impl Distribution<Complex<f64>> for NinePointComplexW {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        self.0.sample(rng)
    }
}

impl NinePointComplexW {
    #[must_use]
    pub fn new(dt: f64) -> Self {
        let w_plus = (3.0 * dt).sqrt() / std::f64::consts::SQRT_2;

        Self(NPoint {
            // Safety: [1, 1, 4, 1, 1, 4, 4, 4, 16] are valid weights
            weights: unsafe { WeightedIndex::new([1, 1, 4, 1, 1, 4, 4, 4, 16]).unwrap_unchecked() },
            values: [
                Complex {
                    re: w_plus,
                    im: w_plus,
                },
                Complex {
                    re: -w_plus,
                    im: w_plus,
                },
                Complex {
                    re: 0f64,
                    im: w_plus,
                },
                Complex {
                    re: w_plus,
                    im: -w_plus,
                },
                Complex {
                    re: -w_plus,
                    im: -w_plus,
                },
                Complex {
                    re: 0f64,
                    im: -w_plus,
                },
                Complex {
                    re: w_plus,
                    im: 0f64,
                },
                Complex {
                    re: -w_plus,
                    im: 0f64,
                },
                Complex { re: 0f64, im: 0f64 },
            ],
        })
    }
}

/// The two point W distribution
///
/// This is the I tilde distribution as defined in 5.7 of
/// in <https://www.jstor.org/stable/pdf/27862707.pdf>
///
/// P(W = \pm \sqrt{dt}) = 1/2
pub struct TwoPointW(NPoint<2, f64>);

impl Distribution<f64> for TwoPointW {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.0.sample(rng)
    }
}

impl TwoPointW {
    #[must_use]
    pub fn new(dt: f64) -> Self {
        let w_plus = dt.sqrt();
        Self(NPoint {
            // Safety: [1, 1] are valid weights
            weights: unsafe { WeightedIndex::new([1, 1]).unwrap_unchecked() },
            values: [w_plus, -w_plus],
        })
    }
}

/// The four point W distribution
///
/// This is the generalization of I tilde distribution as defined in 5.7 of
/// in <https://www.jstor.org/stable/pdf/27862707.pdf> to complex noise
///
/// P(W = \pm \sqrt{dt}) = 1/2
pub struct FourPointComplexW(NPoint<4, Complex<f64>>);

impl Distribution<Complex<f64>> for FourPointComplexW {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        self.0.sample(rng)
    }
}

impl FourPointComplexW {
    #[must_use]
    pub fn new(dt: f64) -> Self {
        let w_plus = dt.sqrt() / std::f64::consts::SQRT_2;
        Self(NPoint {
            // Safety: [1, 1, 1, 1] are valid weights
            weights: unsafe { WeightedIndex::new([1, 1, 1, 1]).unwrap_unchecked() },
            values: [
                Complex {
                    re: w_plus,
                    im: w_plus,
                },
                Complex {
                    re: -w_plus,
                    im: w_plus,
                },
                Complex {
                    re: w_plus,
                    im: -w_plus,
                },
                Complex {
                    re: -w_plus,
                    im: -w_plus,
                },
            ],
        })
    }
}
