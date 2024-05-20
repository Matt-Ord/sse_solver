use ndarray::Array1;
use num_complex::Complex;

pub struct SDEStep {
    pub coherent: Complex<f64>,
    pub incoherent: Vec<Complex<f64>>,
}

/// Represents a SDE System, seperated into a 'coherent' term and a series of 'incoherent' terms
#[allow(clippy::module_name_repetitions)]
pub trait SDESystem {
    /// Type used to store a cache of 'Parts' required to calculate a SDE step.
    type Parts: Into<Self::IncoherentParts>;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Self::Parts;

    /// Get the resulting state after the given 'step' has been performed
    #[inline]
    fn apply_step(
        &self,
        out: &mut Array1<Complex<f64>>,
        step: &SDEStep,
        state: &Array1<Complex<f64>>,
        t: f64,
    ) {
        let parts = self.get_parts(state, t);
        Self::apply_step_from_parts(out, &parts, step);
    }

    /// Get the resulting state after the given 'step' has been performed
    fn apply_step_from_parts(out: &mut Array1<Complex<f64>>, parts: &Self::Parts, step: &SDEStep);

    /// Type used to store a cache of 'Parts' required to calculate a SDE step involving only the incoherent term.
    type IncoherentParts;

    /// Get the parts used to calculate an SDE step.
    /// This is useful if multiple separate steps are required, ie for supporting value calculations
    fn get_incoherent_parts(&self, state: &Array1<Complex<f64>>, t: f64) -> Self::IncoherentParts;

    /// Get the resulting state after the given 'step' has been performed
    #[inline]
    fn apply_incoherent_step(
        &self,
        out: &mut Array1<Complex<f64>>,
        incoherent_step: &[Complex<f64>],
        state: &Array1<Complex<f64>>,
        t: f64,
    ) {
        let parts = self.get_incoherent_parts(state, t);
        Self::apply_incoherent_step_from_parts(out, &parts, incoherent_step);
    }

    /// Get the resulting state after the given 'step' has been performed
    /// Involving only incoherent terms
    fn apply_incoherent_step_from_parts(
        out: &mut Array1<Complex<f64>>,
        parts: &Self::IncoherentParts,
        incoherent_step: &[Complex<f64>],
    );

    /// The total number of incoherent terms
    fn n_incoherent(&self) -> usize;
}
