use ndarray::Array2;
use num_complex::Complex;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

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

// The V distribution for n incoherent operators, according to eqn 14.2.8 - 14.2.10
// in TODO paper
pub struct VMatrix {
    pub n: usize,
    pub dt: f64,
}

impl Distribution<Array2<f64>> for VMatrix {
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
