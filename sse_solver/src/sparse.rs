use ndarray::{linalg::Dot, Array1, Array2};
use num_complex::Complex;
use rand_distr::num_traits;

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
