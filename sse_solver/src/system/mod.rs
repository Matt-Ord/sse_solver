use ndarray::Array1;
use num_complex::Complex;
pub mod sse;

pub struct SDEStep<'a> {
    pub coherent: Complex<f64>,
    pub incoherent: &'a Vec<Complex<f64>>,
}

pub struct SDEOperators {
    pub coherent: Array1<Complex<f64>>,
    pub incoherent: Vec<Array1<Complex<f64>>>,
}

/// Represents a SDE System, seperated into a 'coherent' term and a series of 'incoherent' terms
///
/// ```latex
/// X_t = X_{t0} + \int_{t0}^{t} a(s,X) ds + \sum_1^m \int_{t0}^{t} b^j(s, X_s) dW_s^j
/// ```
///
/// Where a(t,X) is the coherent part at time T, and `B(s,X_s)` is the incoherent part at time T
#[allow(clippy::module_name_repetitions)]
pub trait SDESystem {
    /// Type used to store a cache of 'Parts' required to calculate a SDE step.
    type Parts<'a>: Into<Self::IncoherentParts<'a>> + Into<Self::CoherentPart<'a>>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_parts<'a>(&self, state: &'a Array1<Complex<f64>>, t: f64) -> Self::Parts<'a>;

    /// Get the resulting state after the given 'step' has been performed
    /// `coherent_step * a(s,X_s) + \sum_j steps[j] * b^j(s, X_s)`
    #[inline]
    fn get_step(
        &self,
        step: &SDEStep,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_parts(state, t);
        Self::get_step_from_parts(&parts, step)
    }

    /// Get the resulting state after the given 'step' has been performed
    /// `coherent_step * a(s,X_s) + \sum_j steps[j] * b^j(s, X_s)`
    fn get_step_from_parts(parts: &Self::Parts<'_>, step: &SDEStep) -> Array1<Complex<f64>>;

    /// Type used to store a cache of 'Parts'.
    /// Required to calculate a SDE step involving all incoherent terms `^j(s, X_s)`.
    type IncoherentParts<'a>;

    /// Type used to store a cache of a singel part.
    /// Required to calculate a SDE step involving only a single incoherent term `b^j(s, X_s)`.
    type IncoherentPart<'a>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_incoherent_parts<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentParts<'a>;

    /// Get the resulting state after the given 'step' has been performed
    /// `\sum_j steps[j] * b^j(s, X_s)`
    #[inline]
    fn get_incoherent_steps(
        &self,
        steps: &[Complex<f64>],
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_incoherent_parts(state, t);
        Self::get_incoherent_steps_from_parts(&parts, steps)
    }

    /// Get the incoherent steps `\sum_j steps[j] * b^j(s, X_s)`
    fn get_incoherent_steps_from_parts(
        parts: &Self::IncoherentParts<'_>,
        steps: &[Complex<f64>],
    ) -> Array1<Complex<f64>>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_incoherent_part<'a>(
        &self,
        idx: usize,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentPart<'a>;

    /// Get a single incoherent step `step * b^j(s, X_s)`
    #[inline]
    fn get_incoherent_step(
        &self,
        idx: usize,
        step: Complex<f64>,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_incoherent_part(idx, state, t);
        Self::get_incoherent_step_from_part(&parts, step)
    }

    /// Get the resulting state after the given 'step' has been performed
    /// Involving only incoherent terms
    /// `step * b^j(s, X_s)`
    fn get_incoherent_step_from_part(
        part: &Self::IncoherentPart<'_>,
        step: Complex<f64>,
    ) -> Array1<Complex<f64>>;

    /// Type used to store a cache of 'Parts' required to calculate a SDE step involving only the incoherent term `a(s,X_s)`.
    type CoherentPart<'a>;

    /// Get the parts used to calculate an SDE step `a(s,X_s)`.
    fn get_coherent_part<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::CoherentPart<'a>;

    /// Get the coherent step `a(s,X_s)` from parts
    fn get_coherent_step_from_parts(
        parts: &Self::CoherentPart<'_>,
        step: Complex<f64>,
    ) -> Array1<Complex<f64>>;

    /// Get the coherent step, `step * a(s,X_s)`
    #[inline]
    fn get_coherent_step(
        &self,
        step: Complex<f64>,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_coherent_part(state, t);
        Self::get_coherent_step_from_parts(&parts, step)
    }

    /// The total number of incoherent terms
    fn n_incoherent(&self) -> usize;

    fn operators_from_parts(parts: &Self::Parts<'_>) -> SDEOperators;
}
