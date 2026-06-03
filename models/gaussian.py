"""
models/gaussian.py
------------------
Gaussian (Normal) MLE fitting.
Closed-form estimators; standard errors from the Normal Fisher Information.
"""

import numpy as np


def fit(r: np.ndarray) -> dict:
    """
    MLE for N(μ, σ²).

    Returns
    -------
    dict with keys: mu, sigma, se_mu, se_sigma, loglik, aic, bic
    """
    n   = len(r)
    mu  = r.mean()
    sig = r.std(ddof=0)      # biased MLE estimator (divides by n)

    loglik = -0.5 * n * np.log(2 * np.pi * sig ** 2) - \
             np.sum((r - mu) ** 2) / (2 * sig ** 2)
    k = 2
    return {
        "mu":       mu,
        "sigma":    sig,
        "se_mu":    sig / np.sqrt(n),
        "se_sigma": sig / np.sqrt(2 * n),
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(n) - 2 * loglik,
        "n_params": k,
    }
