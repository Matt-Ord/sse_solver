use ndarray::{Array1, ArrayView1, ArrayViewMut1, array, s};
use ndarray_linalg::Scalar;
use num_complex::Complex;

use crate::system::simple_stochastic::{SimpleStochasticFn, SimpleStochasticSDESystem};

pub trait LangevinParameters {
    fn dimensionless_mass(&self) -> f64;
    fn dimensionless_lambda(&self) -> f64;
    fn kbt_div_hbar(&self) -> f64;
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64;
    fn get_potential_coefficient(&self, idx: u32, alpha: Complex<f64>, ratio: Complex<f64>) -> f64;
}

#[derive(Clone)]
pub struct HarmonicLangevinParameters {
    pub dimensionless_mass: f64,
    /// The dimensionless frequency
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Omega}{K_b T}$
    /// ```
    pub dimensionless_omega: f64,
    /// The dimensionless friction coefficient
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Lambda}{K_b T}$
    /// ```
    pub dimensionless_lambda: f64,
    pub kbt_div_hbar: f64,
}

impl LangevinParameters for HarmonicLangevinParameters {
    fn dimensionless_mass(&self) -> f64 {
        self.dimensionless_mass
    }
    fn dimensionless_lambda(&self) -> f64 {
        self.dimensionless_lambda
    }
    fn kbt_div_hbar(&self) -> f64 {
        self.kbt_div_hbar
    }
    #[inline]
    fn get_potential_coefficient(
        &self,
        idx: u32,
        alpha: Complex<f64>,
        _ratio: Complex<f64>,
    ) -> f64 {
        if idx == 1 {
            return self.dimensionless_omega.square() * alpha.re / (2.0 * self.dimensionless_mass);
        }
        if idx == 2 {
            return self.dimensionless_omega.square() / (4.0 * self.dimensionless_mass);
        }
        0.0
    }
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64 {
        self.get_potential_coefficient(1, alpha, Complex { re: 1.0, im: 0.0 })
    }
}

#[derive(Clone)]
pub struct DoubleHarmonicLangevinParameters {
    pub dimensionless_mass: f64,
    /// The dimensionless frequency of the barrier
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Omega_b}{K_b T}$
    /// ```
    pub dimensionless_omega_barrier: f64,
    /// The dimensionless friction coefficient
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Lambda}{K_b T}$
    /// ```
    pub dimensionless_lambda: f64,
    pub kbt_div_hbar: f64,
    pub left_distance_div_lengthscale: f64,
    pub right_distance_div_lengthscale: f64,
}

impl DoubleHarmonicLangevinParameters {
    fn c2(&self) -> f64 {
        -6.0 * self.right_distance_div_lengthscale * self.left_distance_div_lengthscale
    }
    fn c3(&self) -> f64 {
        4.0 * (self.left_distance_div_lengthscale - self.right_distance_div_lengthscale)
    }
    #[allow(clippy::unused_self)]
    fn c4(&self) -> f64 {
        3.0
    }
    fn prefactor(&self) -> f64 {
        let l_times_r = self.left_distance_div_lengthscale * self.right_distance_div_lengthscale;
        self.dimensionless_omega_barrier.square() / (12.0 * self.dimensionless_mass * l_times_r)
    }
}

impl LangevinParameters for DoubleHarmonicLangevinParameters {
    fn dimensionless_mass(&self) -> f64 {
        self.dimensionless_mass
    }
    fn dimensionless_lambda(&self) -> f64 {
        self.dimensionless_lambda
    }
    fn kbt_div_hbar(&self) -> f64 {
        self.kbt_div_hbar
    }
    #[inline]
    fn get_potential_coefficient(&self, idx: u32, alpha: Complex<f64>, ratio: Complex<f64>) -> f64 {
        let prefactor = self.prefactor();
        let width_factor = self.dimensionless_mass / ratio.re;
        let displacement_div_l = 2.0.sqrt() * alpha.re;

        match idx {
            1 => {
                let c4 = self.c4();
                let c3 = self.c3();
                let c2 = self.c2();

                (2.0.sqrt() * prefactor * 0.125)
                    * (4.0 * c2 * displacement_div_l
                        + 3.0 * c3 * (2.0 * displacement_div_l.square() + width_factor)
                        + (4.0 * c4 * displacement_div_l)
                            * (2.0 * displacement_div_l.square() + 3.0 * width_factor))
            }
            2 => {
                let c4 = self.c4();
                let c3 = self.c3();
                let c2 = self.c2();
                prefactor
                    * (c2
                        + 3.0 * c3 * displacement_div_l
                        + 3.0 * c4 * (2.0 * displacement_div_l.square() + width_factor))
            }
            3 => {
                let c4 = self.c4();
                let c3 = self.c3();
                (2.0.sqrt() * prefactor) * (1.5 * c3 + 6.0 * c4 * displacement_div_l)
            }
            4 => {
                let c4 = self.c4();
                prefactor * c4 * 6.0
            }
            _ => 0.0,
        }
    }
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64 {
        let prefactor = self.prefactor();
        let displacement_div_l = 2.0.sqrt() * alpha.re;
        let c4 = self.c4();
        let c3 = self.c3();
        let c2 = self.c2();

        (2.0.sqrt() * prefactor * 0.125)
            * (4.0 * c2 * displacement_div_l
                + 3.0 * c3 * (2.0 * displacement_div_l.square())
                + (4.0 * c4 * displacement_div_l) * (2.0 * displacement_div_l.square()))
    }
}

#[derive(Clone)]
pub struct PeriodicLangevinParameters {
    pub dimensionless_mass: f64,
    /// The dimensionless potential, given as rfft coefficients
    /// ```latex
    /// $\displaystyle V(x) = K_b T (c_0 + \sum_{n=1}^{N} c_n 2 \cos{\left(n x\right)})$
    /// ```
    pub dimensionless_potential: Vec<Complex<f64>>,
    pub dk_times_lengthscale: f64,
    /// The dimensionless friction coefficient
    /// ```latex
    /// $\displaystyle \lambda = \frac{\hbar\Lambda}{K_b T}$
    /// ```
    pub dimensionless_lambda: f64,
    /// The Time scale factor
    /// ```latex
    /// $\displaystyle \frac{K_b T}{\hbar}$
    /// ```
    pub kbt_div_hbar: f64,
}

impl LangevinParameters for PeriodicLangevinParameters {
    fn dimensionless_mass(&self) -> f64 {
        self.dimensionless_mass
    }
    fn dimensionless_lambda(&self) -> f64 {
        self.dimensionless_lambda
    }
    fn kbt_div_hbar(&self) -> f64 {
        self.kbt_div_hbar
    }
    /// Calculate the coefficient
    /// ```latex
    /// \begin{align}
    //     C_N
    //     &= \sum_{i=-\infty}^\infty e^{ik n_i x_0}V_i^F(ik n_i)^{N}e^{-(k n_i|\mu+\nu|)^2/4}
    //     \\
    //     &= \sum_{i=1}^\infty (e^{ik n_i x_0}V_i^F(ik n_i)^{N} + e^{-ik n_i x_0}{V_i^F}^*(-ik n_i)^{N}) e^{-(k n_i|\mu+\nu|)^2/4}
    // \end{align}
    /// ```
    fn get_potential_coefficient(&self, idx: u32, alpha: Complex<f64>, ratio: Complex<f64>) -> f64 {
        let k = self.dk_times_lengthscale;
        let x_0 = 2.0.sqrt() * alpha.re;
        let mut out = 0.0;
        for (n_i, v_k) in self.dimensionless_potential.iter().enumerate().skip(1) {
            #[allow(clippy::cast_precision_loss)]
            let phase = k * n_i as f64;

            let prefactor =
                2.0 * (-0.25 * self.dimensionless_mass * phase.square() / ratio.re).exp();

            let inner = (Complex::i() * phase * x_0).exp() * v_k * (Complex::i() * phase).powu(idx);

            out += prefactor * inner.re;
        }
        out
    }
    fn get_classical_potential_coefficient(&self, alpha: Complex<f64>) -> f64 {
        let k = self.dk_times_lengthscale;
        let x_0 = 2.0.sqrt() * alpha.re;
        let mut out = 0.0;
        for (n_i, v_k) in self.dimensionless_potential.iter().enumerate().skip(1) {
            #[allow(clippy::cast_precision_loss)]
            let phase = k * n_i as f64;
            // Classically, the wavefunction has a width much less than the potential period
            // So the exponential prefactor is approximately 1
            let prefactor = 2.0;

            let inner = (Complex::i() * phase * x_0).exp() * v_k * (Complex::i() * phase).powu(1);

            out += prefactor * inner.re;
        }
        out
    }
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in a harmonic potential with Langevin dynamics.
///
/// We simulate alpha, where alpha is the coherent state parameter.
/// The SDE is given by:
/// ```latex
/// $d\alpha = \displaystyle \frac{K_{B} T \left(4 M^{2} \operatorname{im}{\left(\alpha\right)} - 2 i M \Lambda \operatorname{im}{\left(\alpha\right)} - i \Omega^{2} \operatorname{re}{\left(\alpha\right)}\right)}{2 M \text{hbar}} + \frac{i \sqrt{K_{B} T} \sqrt{\Lambda} \xi(t)}{\sqrt{M} \sqrt{\text{hbar}}}$
/// ```
#[must_use]
pub fn get_langevin_system<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> SimpleStochasticSDESystem {
    let alpha_im_factor = Complex {
        re: 2.0 * params.kbt_div_hbar() * params.dimensionless_mass().square(),
        im: -params.dimensionless_lambda() * params.kbt_div_hbar(),
    };
    let force_prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda()
        / (2.0 * params.dimensionless_mass()))
    .sqrt();
    let params_coherent = params.clone();
    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];

            let c1 = params_coherent.get_classical_potential_coefficient(alpha);
            let potential_factor = Complex { re: 0.0, im: -c1 } * params_coherent.kbt_div_hbar();

            array![alpha_im_factor * alpha.im + potential_factor]
        }),
        incoherent: vec![Box::new(move |_t, _state| {
            array![Complex {
                re: 0.0,
                im: force_prefactor
            }]
        })],
    }
}

fn get_quantum_re_force_prefactor<T: LangevinParameters>(
    params: &T,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda() / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass().sqrt();
    Complex {
        re: 0.0,
        // re: prefactor * sqrt_m * ((2.0 / ratio.re) - 1.0),
        im: -2.0 * ratio.im * prefactor / (sqrt_m * ratio.re),
    }
}

fn get_quantum_im_force_prefactor<T: LangevinParameters>(
    params: &T,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda() / 8.0).sqrt();
    let sqrt_m = params.dimensionless_mass().sqrt();
    Complex {
        re: 0.0,
        // re: sqrt_m * (ratio.im / ratio.re) * prefactor,
        im: prefactor * (2.0 - ratio.re - ratio.im * (ratio.im / ratio.re)) / (sqrt_m),
    }
}

/// Apply the lowering operator to the state
fn get_lowered_state(psi: &ArrayView1<Complex<f64>>) -> Array1<Complex<f64>> {
    let ns = psi.len();
    let mut out = Array1::zeros(ns);
    for n in 1..ns {
        #[allow(clippy::cast_precision_loss)]
        let factor = (n as f64).sqrt();
        out[n - 1] = psi[n] * factor;
    }
    out
}

fn build_quantum_incoherent_terms<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> Vec<Box<SimpleStochasticFn>> {
    let params0 = params.clone();
    let params1 = params.clone();
    let random_scatter_prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda()
        / (8.0 * params.dimensionless_mass()))
    .sqrt();
    vec![
        Box::new(move |_t, state| {
            let ratio = state[1];
            let re_prefactor = get_quantum_re_force_prefactor(&params0, ratio);

            let mut out = Array1::zeros(state.len());
            out[0] = re_prefactor;

            let occupation = state.slice(s![2..]);
            let a_operator = get_lowered_state(&occupation);
            let mu_plus_nu = get_mu_plus_nu(ratio, params0.dimensionless_mass());

            let re_scatter_prefactor =
                random_scatter_prefactor * mu_plus_nu.conj() * (2.0 + ratio.conj());

            out.slice_mut(s![2..])
                .assign(&(a_operator * re_scatter_prefactor));
            out
        }),
        Box::new(move |_t, state| {
            let ratio = state[1];
            let im_prefactor = get_quantum_im_force_prefactor(&params1, ratio);

            let mut out = Array1::zeros(state.len());
            out[0] = im_prefactor;

            let occupation = state.slice(s![2..]);
            let a_operator = get_lowered_state(&occupation);
            let mu_plus_nu = get_mu_plus_nu(ratio, params1.dimensionless_mass());

            let im_scatter_prefactor =
                random_scatter_prefactor * mu_plus_nu.conj() * (2.0 - ratio.conj());

            out.slice_mut(s![2..])
                .assign(&(a_operator * im_scatter_prefactor));
            out
        }),
    ]
}

fn get_ratio_derivative<T: LangevinParameters>(
    params: &T,
    alpha: Complex<f64>,
    ratio: Complex<f64>,
) -> Complex<f64> {
    let r2 = Complex {
        re: -0.125 * params.dimensionless_lambda(),
        im: -2.0,
    };
    let r1 = -params.dimensionless_lambda();

    let c2 = params.get_potential_coefficient(2, alpha, ratio);
    let r0 = Complex::new(
        params.dimensionless_lambda(),
        2.0 * c2 * params.dimensionless_mass(),
    );

    params.kbt_div_hbar() * ((ratio * ratio) * r2 + ratio * r1 + r0)
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in the most stable squeezed state
/// in a harmonic potential with Calderia-Leggett quantum Langevin dynamics.
#[must_use]
pub fn get_stable_quantum_langevin_system<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
) -> SimpleStochasticSDESystem {
    let alpha_im_factor = Complex {
        re: 2.0 * params.kbt_div_hbar() * params.dimensionless_mass().square(),
        im: -params.dimensionless_lambda() * params.kbt_div_hbar(),
    };

    let params_coherent = params.clone();
    let params_incoherent = params.clone();

    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];
            let ratio = state[1];

            let mut out = Array1::zeros(state.len());

            let c1 = params_coherent.get_potential_coefficient(1, alpha, ratio);
            let potential_factor = Complex { re: 0.0, im: -c1 } * params_coherent.kbt_div_hbar();
            out[0] = alpha_im_factor * alpha.im + potential_factor;
            out[1] = get_ratio_derivative(&params_coherent, alpha, ratio);

            out
        }),
        incoherent: build_quantum_incoherent_terms(&params_incoherent),
    }
}

struct OperatorCache {
    // A cache of pre-computed factors sqrt(n!) for n=0..size-1
    sqrt_factors: Vec<f64>,
    // A cache of pre-computed factors 1/n! for n=0..size-1
    inv_factors: Vec<f64>,
}

impl OperatorCache {
    fn build(size: usize) -> Self {
        // Initialize the vectors with capacity for efficiency
        let mut sqrt_factors = Vec::with_capacity(size);
        let mut inv_factors = Vec::with_capacity(size);

        // n=0 case: 0! = 1.0. sqrt(0!) = 1.0, 1/0! = 1.0
        if size > 0 {
            sqrt_factors.push(1.0);
            inv_factors.push(1.0);
        }

        // Iterate from n=1 up to size-1 (since n=0 is handled above)
        for n in 1..size {
            // Calculate sqrt(n!): sqrt(n!) = sqrt((n-1)!) * sqrt(n)
            #[allow(clippy::cast_precision_loss)]
            let next_sqrt = sqrt_factors[n - 1] * (n as f64).sqrt();
            sqrt_factors.push(next_sqrt);

            // Calculate 1/n!: 1/n! = (1/(n-1)!) / n
            #[allow(clippy::cast_precision_loss)]
            let next_inv = inv_factors[n - 1] / (n as f64);
            inv_factors.push(next_inv);
        }

        Self {
            sqrt_factors,
            inv_factors,
        }
    }
}

fn get_mu_plus_nu(ratio: Complex<f64>, dimensionless_m: f64) -> Complex<f64> {
    let m_plus_r = dimensionless_m + ratio;
    let mu_plus_mu_sq = 2.0 * dimensionless_m * m_plus_r.conj() / (ratio.re * m_plus_r);
    mu_plus_mu_sq.sqrt()
}

fn add_c2_scattering<T: LangevinParameters>(
    psi: &ArrayView1<Complex<f64>>,
    alpha: Complex<f64>,
    ratio: Complex<f64>,
    params: &T,
    psi_out: &mut ArrayViewMut1<Complex<f64>>,
) {
    let ns = psi.len();

    let mu_plus_nu = get_mu_plus_nu(ratio, params.dimensionless_mass());

    // Add the effect of scattering at i=2.
    // We only include the effect when n=m, as other terms are eliminated
    let c2_val = params.get_potential_coefficient(2, alpha, ratio);
    let factor = Complex::new(0.0, -c2_val * params.kbt_div_hbar());
    for n in 1..ns {
        #[allow(clippy::cast_precision_loss)]
        let n_f64 = n as f64;
        psi_out[n] += factor * mu_plus_nu.norm_sqr() * n_f64 * psi[n];
    }
}

fn add_potential_scattering<T: LangevinParameters>(
    psi: &ArrayView1<Complex<f64>>,
    alpha: Complex<f64>,
    ratio: Complex<f64>,
    params: &T,
    cache: &OperatorCache,
    psi_out: &mut ArrayViewMut1<Complex<f64>>,
) {
    let ns = psi.len();

    let mu_plus_nu = get_mu_plus_nu(ratio, params.dimensionless_mass());

    let mut alpha_dist = Vec::with_capacity(ns);
    alpha_dist.push(Complex::new(1.0, 0.0));
    for i in 1..ns {
        #[allow(clippy::cast_precision_loss)]
        let next_alpha = unsafe { alpha_dist.last().unwrap_unchecked() } * mu_plus_nu / (i as f64);
        alpha_dist.push(next_alpha);
    }

    add_c2_scattering(psi, alpha, ratio, params, psi_out);
    // Add to psi_out the remaining effect of scattering.
    // We ignore C_1, C_2 as they are modified by the squeezing transformation.
    // The sum we are trying to compute is:
    // psi_out[m] += (-i/ hbar) \sum_n \sum_k^{min(m,n)} C_{L} (alpha^p / p!) (beta^q / q!) (sqrt(m!n!)/k!) psi[n]
    // where L = m+n-2K, p=m-k, q=n-k.
    // However we re-arrange these sums to first loop over L, then p, then k.
    // This allows us to only compute C_L once per L, rather than for every (m,n) pair.
    for l in 3..(2 * ns - 1) {
        #[allow(clippy::cast_possible_truncation)]
        let c_val = params.get_potential_coefficient(l as u32, alpha, ratio);

        if c_val < 1e-12 {
            continue;
        }

        // Determine valid range for 'p' such that p + q = l
        // Constraints: 0 <= p < ns  AND  0 <= q < ns
        // Since q = l - p, implies: p > l - ns
        // 0 <= p < ns and l - ns < p =< l
        let p_start = 0.max(l - ns + 1);
        let p_end = l.min(ns - 1);
        for p in p_start..=p_end {
            let q = l - p;

            // Pre-calculate the static part of the term
            // Term = (-i/hbar) * C_L * (alpha^p / p!) * (beta^q / q!)
            let mut factor_static = c_val * alpha_dist[p] * alpha_dist[q].conj();
            factor_static *= Complex::new(0.0, -params.kbt_div_hbar());
            if factor_static.norm_sqr() < 1e-24 {
                continue;
            }

            // 4. Inner Loop 'k': The "Shift"
            // We apply this term to the vectors.
            // Output index m = p + k
            // Input index  n = q + k
            // Limit: m < ns AND n < ns => k < ns - max(p, q)
            let max_pq = std::cmp::max(p, q);
            let k_limit = ns - 1 - max_pq;

            for k in 0..=k_limit {
                let m = p + k;
                let n = q + k;

                // weight = (sqrt(m!n!) / k!)
                let weight = cache.sqrt_factors[m] * cache.sqrt_factors[n] * cache.inv_factors[k];
                psi_out[m] += factor_static * weight * psi[n];
            }
        }
    }
}

/// Create a `SimpleStochasticSDESystem` representing a particle
/// in the most stable squeezed state
/// in a harmonic potential with Calderia-Leggett quantum Langevin dynamics.
#[must_use]
pub fn get_quantum_langevin_system<T: LangevinParameters + Clone + Send + Sync + 'static>(
    params: &T,
    size: usize,
) -> SimpleStochasticSDESystem {
    let alpha_im_factor = Complex {
        re: 2.0 * params.kbt_div_hbar() * params.dimensionless_mass().square(),
        im: -params.dimensionless_lambda() * params.kbt_div_hbar(),
    };

    let params_coherent = params.clone();
    let params_incoherent = params.clone();
    let operator_cache = OperatorCache::build(size);
    let coherent_scatter_prefactor = (params.kbt_div_hbar() * params.dimensionless_lambda())
        / (4.0 * params.dimensionless_mass());

    SimpleStochasticSDESystem {
        coherent: Box::new(move |_t, state| {
            let alpha = state[0];
            let ratio = state[1];
            let occupation = state.slice(s![2..]);

            let mut out = Array1::zeros(state.len());

            let c1 = params_coherent.get_potential_coefficient(1, alpha, ratio);
            let potential_factor = Complex { re: 0.0, im: -c1 } * params_coherent.kbt_div_hbar();

            out[0] = alpha_im_factor * alpha.im + potential_factor;

            // We also have an additional contribution to d alpha/dt if the state
            // is not in the ground state
            let a_state = get_lowered_state(&occupation);
            let expect_a = a_state
                .iter()
                .zip(occupation.iter())
                .map(|(a_val, psi_val)| a_val.conj() * psi_val)
                .sum::<Complex<f64>>();
            let ratio_norm_sqr = ratio.norm_sqr();
            let mu_plus_nu = get_mu_plus_nu(ratio, params_coherent.dimensionless_mass());
            let scaled_expect_a = expect_a * mu_plus_nu;

            let expect_l =
                scaled_expect_a * (ratio.conj() + 2.0) - (ratio - 2.0) * scaled_expect_a.conj();
            let inner = expect_l
                * ((params_coherent.dimensionless_mass() - ratio) * (2.0 - ratio.conj()))
                + expect_l.conj()
                    * ((params_coherent.dimensionless_mass() + ratio.conj()) * (2.0 - ratio));
            out[0] += coherent_scatter_prefactor * inner / (2.0 * ratio.re);

            out[1] = get_ratio_derivative(&params_coherent, alpha, ratio);
            let mut psi_out = out.slice_mut(s![2..]);

            // Add the effect of scattering from all states C_i with i > 2
            add_potential_scattering(
                &occupation,
                alpha,
                ratio,
                &params_coherent,
                &operator_cache,
                &mut psi_out,
            );

            // We have a contribution from single ladder operators (ie C_01) as well
            let mu_plus_nu_conj = mu_plus_nu.conj();

            let e_1_0 = expect_a.conj() / mu_plus_nu;
            let prefactor = coherent_scatter_prefactor
                * mu_plus_nu_conj
                * (e_1_0 * (4.0 + ratio_norm_sqr) + e_1_0.conj() * (4.0 - ratio_norm_sqr));

            psi_out += &(&a_state * prefactor);

            // And another from the operator C_02
            let a_a_state = get_lowered_state(&a_state.view());
            let prefactor = 0.5
                * coherent_scatter_prefactor
                * (ratio_norm_sqr - 4.0)
                * (mu_plus_nu_conj * mu_plus_nu_conj);
            psi_out += &(&a_a_state * prefactor);
            out
        }),
        incoherent: build_quantum_incoherent_terms(&params_incoherent),
    }
}
