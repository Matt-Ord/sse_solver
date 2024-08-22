use ndarray::Array2;
use num_complex::Complex;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, WeightedIndex};

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
                let choice = rng.gen::<bool>();
                out[[i, j]] = if choice { options[0] } else { options[1] };
                out[[j, i]] = if choice { options[1] } else { options[0] };
            }
            out[[i, i]] = options[1];
        }

        out
    }
}

/// The W distribution, according to eqn 14.2.4
/// in 10.1007/978-3-662-12616-5
/// P(W = \pm \sqrt{3dt}) = 1/6
/// P(W = 0) = 1/3
pub struct W {
    weights: WeightedIndex<i32>,
    values: [f64; 3],
}

impl W {
    #[must_use]
    pub fn new(dt: f64) -> Self {
        let w_plus = (3.0 * dt).sqrt();
        W {
            // Safety: [1, 1, 4] are valid weights
            weights: unsafe { WeightedIndex::new([1, 1, 4]).unwrap_unchecked() },
            values: [w_plus, -w_plus, 0.0],
        }
    }
}

impl Distribution<f64> for W {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.values[self.weights.sample(rng)]
    }
}

impl Distribution<Complex<f64>> for W {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<f64> {
        Complex {
            re: rng.sample::<f64, _>(self) / std::f64::consts::SQRT_2,
            im: rng.sample::<f64, _>(self) / std::f64::consts::SQRT_2,
        }
    }
}
