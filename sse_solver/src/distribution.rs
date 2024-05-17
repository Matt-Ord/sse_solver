use num_complex::Complex;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

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
