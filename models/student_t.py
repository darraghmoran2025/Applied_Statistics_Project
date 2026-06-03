"""
models/student_t.py
-------------------
Location-scale Student-t MLE via L-BFGS-B.
Standard errors from the observed Fisher Information (numerical Hessian).
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from utils.hessian import mle_standard_errors


def _neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    nu, mu, sigma = params
    if nu <= 2.0 or sigma <= 0.0:
        return np.inf
    return -np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma))


def fit(r: np.ndarray) -> dict:
    """
    MLE for the location-scale Student-t t(ν, μ, σ).

    Constraints: ν > 2 (finite variance), σ > 0.
    Optimiser: L-BFGS-B.

    Returns
    -------
    dict with keys: nu, mu, sigma, se_nu, se_mu, se_sigma,
                    loglik, aic, bic, n_params
    """
    init   = np.array([5.0, r.mean(), r.std()])
    bounds = [(2.01, 200.0), (None, None), (1e-8, None)]

    result = minimize(
        _neg_loglik, init, args=(r,),
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Student-t MLE failed to converge: {result.message}")

    nu, mu, sigma = result.x

    if nu < 2.5:
        import warnings
        warnings.warn(
            f"ν̂ = {nu:.4f} is close to the boundary ν = 2. "
            "Standard errors and ES estimates may be unreliable.",
            RuntimeWarning
        )

    se = mle_standard_errors(lambda p: _neg_loglik(p, r), result.x)
    loglik = -result.fun
    k = 3

    return {
        "nu":       nu,    "se_nu":    se[0],
        "mu":       mu,    "se_mu":    se[1],
        "sigma":    sigma, "se_sigma": se[2],
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(len(r)) - 2 * loglik,
        "n_params": k,
    }
