use ndarray::Array1;
use num_complex::Complex;

use crate::system::{SDEOperators, SDESystem};

pub type SimpleStochasticFn =
    dyn Fn(f64, &Array1<Complex<f64>>) -> Array1<Complex<f64>> + Send + Sync;
/// A Simple Stochastic System where the coherent and incoherent parts are given by functions
pub struct SimpleStochasticSDESystem {
    pub coherent: Box<SimpleStochasticFn>,
    pub incoherent: Vec<Box<SimpleStochasticFn>>,
}

pub struct SimpleStochasticParts {
    coherent: Array1<Complex<f64>>,
    incoherent: Vec<Array1<Complex<f64>>>,
}

pub struct SimpleStochasticIncoherentParts {
    incoherent: Vec<Array1<Complex<f64>>>,
    size: usize,
}

impl From<SimpleStochasticParts> for SimpleStochasticIncoherentParts {
    fn from(val: SimpleStochasticParts) -> Self {
        SimpleStochasticIncoherentParts {
            incoherent: val.incoherent,
            size: val.coherent.len(),
        }
    }
}
pub struct SimpleStochasticIncoherentPart {
    incoherent: Array1<Complex<f64>>,
}

pub struct SimpleStochasticCoherentPart {
    coherent: Array1<Complex<f64>>,
}

impl From<SimpleStochasticParts> for SimpleStochasticCoherentPart {
    fn from(val: SimpleStochasticParts) -> Self {
        SimpleStochasticCoherentPart {
            coherent: val.coherent,
        }
    }
}

impl SDESystem for SimpleStochasticSDESystem {
    type Parts<'a> = SimpleStochasticParts;

    fn get_parts<'a>(&self, state: &'a Array1<Complex<f64>>, t: f64) -> Self::Parts<'a> {
        SimpleStochasticParts {
            coherent: (self.coherent)(t, state),
            incoherent: self
                .incoherent
                .iter()
                .map(|f| f(t, state))
                .collect::<Vec<_>>(),
        }
    }

    fn get_step_from_parts(parts: &Self::Parts<'_>, step: &super::SDEStep) -> Array1<Complex<f64>> {
        let mut result = &parts.coherent * step.coherent;
        for (i, s) in step.incoherent.iter().enumerate() {
            result = result + &parts.incoherent[i] * *s;
        }
        result
    }

    type IncoherentParts<'a> = SimpleStochasticIncoherentParts;

    type IncoherentPart<'a> = SimpleStochasticIncoherentPart;

    fn get_incoherent_parts<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentParts<'a> {
        SimpleStochasticIncoherentParts {
            size: state.len(),
            incoherent: self
                .incoherent
                .iter()
                .map(|f| f(t, state))
                .collect::<Vec<_>>(),
        }
    }

    fn get_incoherent_steps_from_parts(
        parts: &Self::IncoherentParts<'_>,
        steps: &[f64],
    ) -> Array1<Complex<f64>> {
        let mut result = Array1::from_elem(parts.size, Complex::<f64>::new(0.0, 0.0));
        for (i, step) in steps.iter().enumerate() {
            result = result + &parts.incoherent[i] * *step;
        }
        result
    }

    fn get_incoherent_part<'a>(
        &self,
        idx: usize,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::IncoherentPart<'a> {
        SimpleStochasticIncoherentPart {
            incoherent: (self.incoherent[idx])(t, state),
        }
    }

    fn get_incoherent_step_from_part(
        part: &Self::IncoherentPart<'_>,
        step: f64,
    ) -> Array1<Complex<f64>> {
        &part.incoherent * step
    }

    type CoherentPart<'a> = SimpleStochasticCoherentPart;

    fn get_coherent_part<'a>(
        &self,
        state: &'a Array1<Complex<f64>>,
        t: f64,
    ) -> Self::CoherentPart<'a> {
        SimpleStochasticCoherentPart {
            coherent: (self.coherent)(t, state),
        }
    }

    fn get_coherent_step_from_parts(
        parts: &Self::CoherentPart<'_>,
        step: f64,
    ) -> Array1<Complex<f64>> {
        &parts.coherent * step
    }

    fn n_incoherent(&self) -> usize {
        self.incoherent.len()
    }

    fn get_operators_from_parts(parts: &Self::Parts<'_>) -> super::SDEOperators {
        super::SDEOperators {
            coherent: parts.coherent.clone(),
            incoherent: parts.incoherent.clone(),
        }
    }
    // TODO: we should make an abstraction for an SDE system who's parts are
    // defined by its 'operators' only.
    fn get_operators(&self, state: &Array1<Complex<f64>>, t: f64) -> SDEOperators {
        let parts = self.get_parts(state, t);
        SDEOperators {
            coherent: parts.coherent,
            incoherent: parts.incoherent,
        }
    }
}
