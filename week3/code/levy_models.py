"""
levy_models.py — Variance-Gamma and Normal Inverse Gaussian MLE
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Provides the two Lévy process models for Week 3 onward.  All downstream
scripts import from this module rather than re-implementing the mathematics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
fit_vg(r, n_starts)  → dict
    MLE for VG(μ, σ, θ, ν).  Optimises over L-BFGS-B from n_starts
    starting points; returns the global best.

fit_nig(r, n_starts)  → dict
    MLE for NIG(μ, α, β, δ).  Uses the ξ = β/α reparametrisation
    for box-constrained optimisation; converts to natural parameters
    on output.  SEs computed via Jacobian transformation.

simulate_vg(vg, n, seed)  → ndarray
    Draw n samples from VG(μ,σ,θ,ν) via Gamma–Normal mixture.

simulate_nig(nig, n, seed)  → ndarray
    Draw n samples from NIG(μ,α,β,δ) via Inverse-Gaussian–Normal
    variance-mean mixture.

var_es_mc(samples, alphas)  → dict
    VaR and ES at each confidence level from a pre-generated sample.
    Output dict keyed by confidence level, each value {"VaR": …, "ES": …}.

Internal helpers (prefixed _; not called directly):
    _logpdf_vg(x, mu, sigma, theta, nu)
    _logpdf_nig(x, mu, alpha, beta, delta)
    _neg_loglik_vg(params, r)
    _neg_loglik_nig_xi(params, r)
    _numerical_hessian(f, x)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER CONVENTIONS
─────────────────────
VG  (μ, σ, θ, ν)
  μ   location shift (ℝ)
  σ   scale (> 0)
  θ   asymmetry / Brownian drift (ℝ; negative → left-skewed)
  ν   variance rate of Gamma time-change (> 0)
  Mean     = μ + θ
  Variance = σ² + θ²ν

NIG  (μ, α, β, δ)
  μ   location shift (ℝ)
  α   tail-heaviness (> 0; larger α → lighter tails)
  β   asymmetry (−α < β < α; negative → left-skewed)
  δ   scale (> 0)
  γ   = √(α²−β²)   [derived]
  Mean     = μ + δβ/γ
  Variance = δα²/γ³

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATHEMATICAL NOTES
──────────────────
VG PDF (Madan, Carr, Chang 1998, eq. 2):

  f(x) = 2·exp(θy/σ²) · (|y|/ω)^(1/ν−½) · K_{1/ν−½}(ω|y|/σ²)
         ─────────────────────────────────────────────────────────
         σ√(2π) · ν^(1/ν) · Γ(1/ν)

  where y = x − μ,  ω = √(2σ²/ν + θ²)

  Derivation: integrate out the Gamma time-change G ~ Gamma(1/ν, ν)
  from the conditional normal N(μ + θG, σ²G) using the integral
  ∫₀^∞ g^(p−1) exp(−qg − r/g) dg = 2(r/q)^(p/2) K_p(2√(qr)).

NIG PDF (Barndorff-Nielsen 1997):

  f(x) = αδ · exp(δγ + β(x−μ)) · K₁(αq) / (π · q)

  where γ = √(α²−β²),  q = √(δ²+(x−μ)²)

Bessel numerics: K_ν(z) = kve(ν,z)·exp(−z), so
  log K_ν(z) = log kve(ν,z) − z.
Using kve avoids underflow for large z (deep tail observations).
kve handles negative orders via the reflection K_{−ν} = K_ν.

NIG simulation uses the variance-mean mixture:
  X | V ~ N(μ + βV, V),   V ~ IG(δ/γ, δ²)
  where scipy invgauss(mu_s, scale=s) has mean = mu_s·s, var = mu_s³·s².
  Setting mu_s = 1/(δγ), s = δ² gives E[V]=δ/γ, Var[V]=δ/γ³ as required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install dependencies
────────────────────
    pip install numpy pandas scipy yfinance matplotlib

REFERENCES
──────────
Madan, D. B., Carr, P. P. and Chang, E. C. (1998). The Variance Gamma
Process and Option Pricing. European Finance Review, 2(1), 79–105.
"""

import getpass
import os
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import kve, gammaln

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ════════════════════════════════════════════════════════════════════════════

def default_save_dir(script_file):
    """
    Return the default figure output directory.

    For user 'darra': figures go into a 'figures' subfolder alongside the
    calling script (i.e. inside the Week3 folder).
    For all other users: figures go into their system Downloads folder so
    nothing is written to an unfamiliar project path.

    Parameters
    ----------
    script_file : str   pass __file__ from the calling script
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    if getpass.getuser() == "darra":
        return os.path.join(script_dir, "figures")
    return os.path.join(os.path.expanduser("~"), "Downloads")


# ════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS  (mirrored from week2_gaussian_student_mle.py)
# ════════════════════════════════════════════════════════════════════════════

SHOCK_PERIODS = {
    "Dot-com crash":  ("2000-03-01", "2002-10-31"),
    "GFC":            ("2007-10-01", "2009-03-31"),
    "COVID-19":       ("2020-02-01", "2020-06-30"),
    "Fed rate hikes": ("2022-01-01", "2023-12-31"),
}

SHOCK_COLOURS = {
    "Dot-com crash":  "#fff3cd",
    "GFC":            "#f8d7da",
    "COVID-19":       "#d1ecf1",
    "Fed rate hikes": "#d4edda",
}

# Consistent colour scheme for the models across all Week 3 figures
MODEL_COLOURS = {
    "Gaussian":  "steelblue",
    "Laplace":   "purple",
    "Student-t": "crimson",
    "VG":        "forestgreen",
    "NIG":       "darkorange",
}


# ════════════════════════════════════════════════════════════════════════════
# SHARED NUMERICAL UTILITY
# ════════════════════════════════════════════════════════════════════════════

def _numerical_hessian(f, x):
    """
    4-point central-difference Hessian with parameter-scaled step sizes.

    h_i = max(1e-6, 1e-4·|x_i|) so steps scale with each parameter's
    magnitude.  Identical to the implementation in week2_gaussian_student_mle.py.
    """
    n = len(x)
    h = np.maximum(1e-6, 1e-4 * np.abs(x))
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


# ════════════════════════════════════════════════════════════════════════════
# VARIANCE-GAMMA — LOG-PDF
# ════════════════════════════════════════════════════════════════════════════

def _logpdf_vg(x, mu, sigma, theta, nu):
    """
    Vectorised log-PDF of VG(μ, σ, θ, ν).

    Uses scipy.special.kve for the scaled Bessel function to avoid
    underflow in the tails: log K_ν(z) = log kve(ν,z) − z.
    The guard abs_y ≥ 1e-14 prevents log(0) when a return equals μ exactly.
    """
    y     = x - mu
    abs_y = np.maximum(np.abs(y), 1e-14)
    omega = np.sqrt(2.0 * sigma**2 / nu + theta**2)
    order = 1.0 / nu - 0.5
    z     = omega * abs_y / sigma**2

    log_kv = np.log(kve(order, z)) - z

    return (
        np.log(2.0)
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - (1.0 / nu) * np.log(nu)
        - gammaln(1.0 / nu)
        + theta * y / sigma**2
        + order * np.log(abs_y / omega)
        + log_kv
    )


def _neg_loglik_vg(params, r):
    sigma, theta, nu, mu = params
    if sigma <= 0 or nu <= 0:
        return np.inf
    ll = _logpdf_vg(r, mu, sigma, theta, nu)
    if not np.all(np.isfinite(ll)):
        return np.inf
    return -np.sum(ll)


# ════════════════════════════════════════════════════════════════════════════
# VARIANCE-GAMMA — MLE
# ════════════════════════════════════════════════════════════════════════════

def fit_vg(r, n_starts=3):
    """
    MLE for the Variance-Gamma distribution VG(μ, σ, θ, ν).

    Optimiser: L-BFGS-B on the parameter vector [σ, θ, ν, μ].
    Multiple starting points guard against local optima; the best finite
    result is returned regardless of the convergence flag.
    Standard errors from the observed Fisher Information (numerical Hessian).

    Parameters
    ----------
    r        : array-like of log-returns
    n_starts : number of starting points to try (1–3)

    Returns
    -------
    dict with keys: sigma, theta, nu, mu, se_sigma, se_theta, se_nu, se_mu,
                    loglik, aic, bic, n_params
    """
    r = np.asarray(r)
    n = len(r)

    inits = [
        [r.std() * 0.80, -0.001,  0.20, r.mean()],
        [r.std() * 1.00,  0.000,  0.50, r.mean()],
        [r.std() * 0.60, -0.005,  0.10, r.mean()],
    ]
    bounds = [(1e-6, 0.50), (-0.30, 0.30), (0.01, 5.0), (-0.15, 0.15)]

    best = None
    for init in inits[:n_starts]:
        res = minimize(
            _neg_loglik_vg, init, args=(r,),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 3000},
        )
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("VG MLE: all starting points produced non-finite log-likelihood")

    sigma, theta, nu, mu = best.x
    loglik = -best.fun
    k = 4

    try:
        H  = _numerical_hessian(lambda p: _neg_loglik_vg(p, r), best.x)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception:
        se = np.full(4, np.nan)
        warnings.warn("VG: Hessian inversion failed; SEs set to NaN.", RuntimeWarning)

    return {
        "sigma":    sigma,  "se_sigma": se[0],
        "theta":    theta,  "se_theta": se[1],
        "nu":       nu,     "se_nu":    se[2],
        "mu":       mu,     "se_mu":    se[3],
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(n) - 2 * loglik,
        "n_params": k,
    }


# ════════════════════════════════════════════════════════════════════════════
# NORMAL INVERSE GAUSSIAN — LOG-PDF
# ════════════════════════════════════════════════════════════════════════════

def _logpdf_nig(x, mu, alpha, beta, delta):
    """
    Vectorised log-PDF of NIG(μ, α, β, δ).

    f(x) = αδ · exp(δγ + β(x−μ)) · K₁(αq) / (π·q)
    where γ = √(α²−β²), q = √(δ²+(x−μ)²).

    Bessel: K₁(z) = kve(1,z)·exp(−z), so log K₁(z) = log kve(1,z) − z.
    """
    gamma = np.sqrt(alpha**2 - beta**2)
    y     = x - mu
    q     = np.sqrt(delta**2 + y**2)
    z     = alpha * q

    log_k1 = np.log(kve(1, z)) - z

    return (
        np.log(alpha)
        + np.log(delta)
        - np.log(np.pi)
        + delta * gamma
        + beta * y
        + log_k1
        - 0.5 * np.log(delta**2 + y**2)
    )


def _neg_loglik_nig_xi(params, r):
    """
    NIG negative log-likelihood in the (α, ξ, δ, μ) parameterisation,
    where ξ = β/α ∈ (−1, 1).

    Box-constraining ξ to (−1, 1) automatically enforces α > |β| for
    any α > 0, making L-BFGS-B applicable without nonlinear constraints.
    """
    alpha, xi, delta, mu = params
    if alpha <= 0 or delta <= 0 or not (-1.0 < xi < 1.0):
        return np.inf
    beta = xi * alpha
    ll = _logpdf_nig(r, mu, alpha, beta, delta)
    if not np.all(np.isfinite(ll)):
        return np.inf
    return -np.sum(ll)


# ════════════════════════════════════════════════════════════════════════════
# NORMAL INVERSE GAUSSIAN — MLE
# ════════════════════════════════════════════════════════════════════════════

def fit_nig(r, n_starts=3):
    """
    MLE for the Normal Inverse Gaussian distribution NIG(μ, α, β, δ).

    Optimises over (α, ξ=β/α, δ, μ) to satisfy α > |β| via box bounds.
    Output converts back to natural parameters (α, β, δ, μ).
    Standard errors in the natural parameterisation via Jacobian:
      ∂β/∂α = ξ,  ∂β/∂ξ = α  ⟹  Cov_natural = J · Cov_xi · Jᵀ.

    Returns
    -------
    dict with keys: alpha, beta, delta, mu, se_alpha, se_beta, se_delta,
                    se_mu, loglik, aic, bic, n_params
    """
    r = np.asarray(r)
    n = len(r)

    inits = [
        [ 80.0, -0.050, 0.008, r.mean()],
        [150.0, -0.100, 0.012, r.mean()],
        [ 50.0, -0.020, 0.005, r.mean()],
    ]
    bounds = [(1.0, 3000.0), (-0.999, 0.999), (1e-6, 0.50), (-0.15, 0.15)]

    best = None
    for init in inits[:n_starts]:
        res = minimize(
            _neg_loglik_nig_xi, init, args=(r,),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 3000},
        )
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("NIG MLE: all starting points produced non-finite log-likelihood")

    alpha, xi, delta, mu = best.x
    beta   = xi * alpha
    loglik = -best.fun
    k      = 4

    try:
        H_xi  = _numerical_hessian(lambda p: _neg_loglik_nig_xi(p, r), best.x)
        cov_xi = np.linalg.inv(H_xi)
        # Jacobian: maps [α, ξ, δ, μ] → [α, β=ξα, δ, μ]
        # ∂β/∂α = ξ,  ∂β/∂ξ = α;  all other partials on diagonal = 1
        J = np.eye(4)
        J[1, 0] = xi
        J[1, 1] = alpha
        cov_nat = J @ cov_xi @ J.T
        se = np.sqrt(np.diag(cov_nat))
    except Exception:
        se = np.full(4, np.nan)
        warnings.warn("NIG: Hessian inversion failed; SEs set to NaN.", RuntimeWarning)

    return {
        "alpha":  alpha,  "se_alpha": se[0],
        "beta":   beta,   "se_beta":  se[1],
        "delta":  delta,  "se_delta": se[2],
        "mu":     mu,     "se_mu":    se[3],
        "loglik": loglik,
        "aic":    2 * k - 2 * loglik,
        "bic":    k * np.log(n) - 2 * loglik,
        "n_params": k,
    }


# ════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_vg(vg, n=1_000_000, seed=None):
    """
    Draw n samples from VG(μ, σ, θ, ν) using the Gamma–Normal mixture.

    Representation: X | G ~ N(μ + θG, σ²G),  G ~ Gamma(shape=1/ν, scale=ν)
    so E[G] = 1, Var[G] = ν, and E[X] = μ + θ as required.
    """
    rng = np.random.default_rng(seed)
    g   = rng.gamma(shape=1.0 / vg["nu"], scale=vg["nu"], size=n)
    return vg["mu"] + vg["theta"] * g + vg["sigma"] * np.sqrt(g) * rng.standard_normal(n)


def simulate_nig(nig, n=1_000_000, seed=None):
    """
    Draw n samples from NIG(μ, α, β, δ) via the Normal Variance-Mean Mixture.

    Representation: X | V ~ N(μ + βV, V),  V ~ IG(mean=δ/γ, var=δ/γ³)
    where γ = √(α²−β²).

    scipy invgauss(mu_s, scale=s) has mean = mu_s·s, var = mu_s³·s².
    Setting mu_s = 1/(δγ), s = δ² yields E[V] = δ/γ, Var[V] = δ/γ³.
    """
    rng    = np.random.default_rng(seed)
    gamma  = np.sqrt(nig["alpha"]**2 - nig["beta"]**2)
    mu_s   = 1.0 / (nig["delta"] * gamma)
    scale  = nig["delta"]**2
    v      = stats.invgauss.rvs(mu=mu_s, scale=scale, size=n, random_state=rng)
    return nig["mu"] + nig["beta"] * v + np.sqrt(v) * rng.standard_normal(n)


# ════════════════════════════════════════════════════════════════════════════
# RISK MEASURES (Monte Carlo)
# ════════════════════════════════════════════════════════════════════════════

def var_es_mc(samples, alphas=(0.95, 0.99)):
    """
    VaR and ES at each confidence level from a pre-generated simulated sample.

    Convention matches compute_risk_measures() in week2_gaussian_student_mle.py:
      VaR_α = α-quantile of the return distribution (negative value = loss)
      ES_α  = E[X | X ≤ VaR_α]  (conditional tail expectation; also negative)

    Returns
    -------
    dict keyed by confidence level, each value {"VaR": float, "ES": float}
    """
    results = {}
    for a in alphas:
        var = np.quantile(samples, 1.0 - a)
        es  = float(samples[samples <= var].mean())
        results[a] = {"VaR": var, "ES": es}
    return results


# ════════════════════════════════════════════════════════════════════════════
# LAPLACE (DOUBLE-EXPONENTIAL) — THE SYMMETRIC VG SPECIAL CASE
# ════════════════════════════════════════════════════════════════════════════

def fit_laplace(r, **kwargs):
    """
    Closed-form MLE for the Laplace (double-exponential) distribution L(μ, b),
    density f(x) = exp(−|x−μ|/b) / (2b).

    The Laplace is the symmetric Variance-Gamma special case VG(μ, σ, θ=0, ν=1):
    when the Gamma time-change has unit shape (1/ν = 1) it becomes an
    Exponential, and the symmetric normal variance-mean mixture collapses to a
    back-to-back pair of exponentials.  It is therefore the simplest
    exponential-tailed, two-parameter benchmark, sitting between the Gaussian
    (which has the same parameter count but far lighter tails) and the heavier
    Lévy models.  Its excess kurtosis is exactly 3, versus 0 for the Gaussian.

    MLE is closed form:
        μ_hat = sample median
        b_hat = mean absolute deviation about the median
    Asymptotic standard errors: SE(μ) = SE(b) = b / √n.

    `**kwargs` (e.g. n_starts) is accepted and ignored so fit_laplace is a
    drop-in match for the iterative fit_* signatures.

    Returns dict: mu, b, se_mu, se_b, loglik, aic, bic, n_params.
    """
    r = np.asarray(r)
    n = len(r)
    mu = float(np.median(r))
    b  = float(np.mean(np.abs(r - mu)))
    loglik = float(np.sum(stats.laplace.logpdf(r, loc=mu, scale=b)))
    k = 2
    se = b / np.sqrt(n)
    return {
        "mu": mu, "se_mu": se,
        "b":  b,  "se_b":  se,
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(n) - 2 * loglik,
        "n_params": k,
    }


def laplace_var_es(lap, alphas=(0.95, 0.99)):
    """
    Closed-form VaR and ES for a fitted Laplace model.

    Lower-tail convention matches var_es_mc():
        VaR_α = (1−α)-quantile = μ + b·ln(2(1−α))   (negative = loss)
        ES_α  = E[X | X ≤ VaR_α] = VaR_α − b

    The ES identity follows from the memorylessness of the exponential lower
    tail: conditional on a loss beyond the quantile, the further excess is
    Exponential(1/b) with mean b, so the conditional mean is one scale below
    the VaR.

    Returns dict keyed by confidence level, each value {"VaR": …, "ES": …}.
    """
    mu, b = lap["mu"], lap["b"]
    results = {}
    for a in alphas:
        p   = 1.0 - a
        var = mu + b * np.log(2.0 * p)
        es  = var - b
        results[a] = {"VaR": var, "ES": es}
    return results
