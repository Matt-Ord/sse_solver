use ndarray::Array1;
use num_complex::Complex;

pub struct SDEStep {
    pub coherent: Complex<f64>,
    pub incoherent: Vec<Complex<f64>>,
}

pub struct SDEOperators {
    pub coherent: Array1<Complex<f64>>,
    pub incoherent: Vec<Array1<Complex<f64>>>,
}

/// Represents a SDE System, seperated into a 'coherent' term and a series of 'incoherent' terms
#[allow(clippy::module_name_repetitions)]
pub trait SDESystem {
    /// Type used to store a cache of 'Parts' required to calculate a SDE step.
    type Parts<'a>: Into<Self::IncoherentParts<'a>> + Into<Self::CoherentParts<'a>>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_parts<'a>(&self, state: &'a Array1<Complex<f64>>, t: f64) -> Self::Parts<'a>;

    /// Get the resulting state after the given 'step' has been performed
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
    fn get_step_from_parts(parts: &Self::Parts<'_>, step: &SDEStep) -> Array1<Complex<f64>>;

    /// Type used to store a cache of 'Parts' required to calculate a SDE step involving only the incoherent term.
    type IncoherentParts<'a>;

    /// Type used to store a cache of a singel part. required to calculate a SDE step involving only a single incoherent term.
    type IncoherentPart<'a>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_incoherent_parts<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentParts<'a>;

    /// Get the resulting state after the given 'step' has been performed
    #[inline]
    fn get_incoherent_steps(
        &self,
        incoherent_step: &[Complex<f64>],
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_incoherent_parts(state, t);
        Self::get_incoherent_steps_from_parts(&parts, incoherent_step)
    }

    /// Get the resulting state after the given 'step' has been performed
    /// Involving only incoherent terms
    fn get_incoherent_steps_from_parts(
        parts: &Self::IncoherentParts<'_>,
        incoherent_step: &[Complex<f64>],
    ) -> Array1<Complex<f64>>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_incoherent_part<'a>(
        &self,
        idx: usize,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentPart<'a>;

    /// Get the resulting state after the given 'step' has been performed
    #[inline]
    fn get_incoherent_step(
        &self,
        idx: usize,
        incoherent_step: Complex<f64>,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_incoherent_part(idx, state, t);
        Self::get_incoherent_step_from_part(&parts, incoherent_step)
    }

    /// Get the resulting state after the given 'step' has been performed
    /// Involving only incoherent terms
    fn get_incoherent_step_from_part(
        part: &Self::IncoherentPart<'_>,
        incoherent_step: Complex<f64>,
    ) -> Array1<Complex<f64>>;

    /// Type used to store a cache of 'Parts' required to calculate a SDE step involving only the incoherent term.
    type CoherentParts<'a>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_coherent_parts<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::CoherentParts<'a>;

    /// Get the resulting state after the given 'step' has been performed
    #[inline]
    fn get_coherent_step(
        &self,
        coherent_step: Complex<f64>,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) -> Array1<Complex<f64>> {
        let parts = self.get_coherent_parts(state, t);
        Self::get_coherent_step_from_parts(&parts, coherent_step)
    }

    /// Get the resulting state after the given 'step' has been performed
    /// Involving only coherent terms
    fn get_coherent_step_from_parts(
        parts: &Self::CoherentParts<'_>,
        coherent_step: Complex<f64>,
    ) -> Array1<Complex<f64>>;

    /// The total number of incoherent terms
    fn n_incoherent(&self) -> usize;

    fn operators_from_parts(&self, parts: &Self::Parts<'_>) -> SDEOperators;
}
