use std::sync::Arc;

use ndarray::{linalg::Dot, Array1, Array2};
use num_complex::Complex;
use rand_distr::num_traits;

use rustfft::{Fft, FftNum, FftPlanner};
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

impl<T: num_complex::ComplexFloat> TransposedBandedArray<T> {
    #[must_use]
    pub fn conj(&self) -> TransposedBandedArray<T> {
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

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FactorizedArray<T> {
    amplitude: T,
    // LHS of the factorized operators.
    // This is in 'bra' form, so bra.dot(state) === <bra|state>
    pub(crate) bra: Array1<T>,
    // RHS of the factorized operators.
    // This is in 'ket' form, so conj(ket).dot(state) === <ket|state>
    pub(crate) ket: Array1<T>,
}

impl Dot<Array1<Complex<f64>>> for FactorizedArray<Complex<f64>> {
    type Output = Array1<Complex<f64>>;

    #[inline]
    fn dot(&self, rhs: &Array1<Complex<f64>>) -> Self::Output {
        let applied_bra = self.bra.dot(rhs);

        &self.ket * (self.amplitude * applied_bra)
    }
}

impl<T: num_complex::ComplexFloat> FactorizedArray<T> {
    #[must_use]
    pub fn conj(&self) -> FactorizedArray<T> {
        FactorizedArray {
            amplitude: self.amplitude.conj(),
            bra: self.bra.clone(),
            ket: self.ket.clone(),
        }
    }
}

impl<T: Clone> FactorizedArray<T> {
    #[must_use]
    pub fn transpose(&self) -> FactorizedArray<T> {
        FactorizedArray {
            amplitude: self.amplitude.clone(),
            ket: self.bra.clone(),
            bra: self.ket.clone(),
        }
    }
}
impl<T> FactorizedArray<T> {
    pub fn from_bra_ket(amplitude: T, bra: Array1<T>, ket: Array1<T>) -> FactorizedArray<T> {
        Self {
            amplitude,
            bra,
            ket,
        }
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Represents a scattering operator of the form
/// ```A(x) + B(x)C(p)D(x)```
/// This allows us to use the split operator method to improve performance
pub struct SplitScatteringArray<T> {
    a: Option<Array1<T>>,
    b: Option<Array1<T>>,
    c: Array1<T>,
    d: Option<Array1<T>>,
    n_states: usize,
}

impl<T: Clone + num_complex::ComplexFloat> SplitScatteringArray<T> {
    #[must_use]
    pub fn conj_transpose(&self) -> SplitScatteringArray<T> {
        SplitScatteringArray {
            a: self.a.as_ref().map(|d| d.map(|i| i.conj())),
            b: self.d.as_ref().map(|d| d.map(|i| i.conj())),
            c: self.c.map(|i| i.conj()),
            d: self.b.as_ref().map(|d| d.map(|i| i.conj())),
            n_states: self.n_states,
        }
    }
}

impl<T> SplitScatteringArray<T> {
    #[must_use]
    pub fn try_from_parts(
        a: Option<Array1<T>>,
        b: Option<Array1<T>>,
        c: Array1<T>,
        d: Option<Array1<T>>,
    ) -> Option<SplitScatteringArray<T>> {
        if [a.as_ref(), b.as_ref(), d.as_ref()]
            .iter()
            .any(|x| x.is_some_and(|i| i.len() != c.len()))
        {
            return None;
        }

        Some(SplitScatteringArray {
            n_states: c.len(),
            a,
            b,
            c,
            d,
        })
    }
    #[must_use]
    /// Build a split scattering matrix from parts
    ///
    /// # Panics
    ///
    /// Will panic if the length of parts are not equal
    pub fn from_parts(
        a: Option<Array1<T>>,
        b: Option<Array1<T>>,
        c: Array1<T>,
        d: Option<Array1<T>>,
    ) -> SplitScatteringArray<T> {
        SplitScatteringArray::try_from_parts(a, b, c, d).expect("Parts must have the same length")
    }
}

pub struct PlannedSplitScatteringArray<T> {
    inner: SplitScatteringArray<Complex<T>>,
    pub forward_plan: Arc<dyn Fft<T>>,
    pub inverse_plan: Arc<dyn Fft<T>>,
}

impl Dot<Array1<Complex<f64>>> for SplitScatteringArray<Complex<f64>> {
    type Output = Array1<Complex<f64>>;

    #[inline]
    fn dot(&self, rhs: &Array1<Complex<f64>>) -> Self::Output {
        PlannedSplitScatteringArray::from(self.to_owned()).dot(rhs)
    }
}

impl Dot<Array1<Complex<f64>>> for PlannedSplitScatteringArray<f64> {
    type Output = Array1<Complex<f64>>;

    #[inline]
    fn dot(&self, rhs: &Array1<Complex<f64>>) -> Self::Output {
        assert!(self.inner.n_states == rhs.len());

        let mut b_vec = if let Some(inner) = &self.inner.b {
            inner * rhs
        } else {
            rhs.to_owned()
        };

        self.forward_plan.process(b_vec.as_slice_mut().unwrap());

        let mut cb_vec = &self.inner.c * b_vec;

        self.inverse_plan.process(cb_vec.as_slice_mut().unwrap());

        if let Some(inner) = &self.inner.a {
            cb_vec + inner * rhs
        } else {
            cb_vec
        }
    }
}

impl<T: FftNum> From<SplitScatteringArray<Complex<T>>> for PlannedSplitScatteringArray<T> {
    fn from(value: SplitScatteringArray<Complex<T>>) -> Self {
        let mut planner = FftPlanner::new();
        let forward_plan = planner.plan_fft_forward(value.n_states);
        let inverse_plan = planner.plan_fft_inverse(value.n_states);

        Self {
            inner: value,
            forward_plan,
            inverse_plan,
        }
    }
}
