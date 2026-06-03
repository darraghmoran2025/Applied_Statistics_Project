"""
risk/measures.py
----------------
Value-at-Risk (VaR) and Expected Shortfall (ES) computation.
Supports Gaussian and Student-t analytically.
VG and NIG (Week 3+) will use numerical integration via scipy.integrate.quad.

Convention: α = confidence level (e.g. 0.95)
            All outputs are daily log-returns; negative values = losses.
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Optional


def var_es_gaussian(g: dict, alpha: float) -> tuple:
    """
    Closed-form VaR and ES for N(μ, σ²).

    ES formula (conditional expectation of truncated Normal):
        ES_α = μ - σ · φ(Φ⁻¹(p)) / p    where p = 1 - α
    """
    p   = 1.0 - alpha
    q   = stats.norm.ppf(p)
    var = g["mu"] + g["sigma"] * q
    es  = g["mu"] - g["sigma"] * stats.norm.pdf(q) / p
    return var, es


def var_es_student_t(t: dict, alpha: float) -> tuple:
    """
    Closed-form VaR and ES for location-scale Student-t t(ν, μ, σ).

    ES formula (standard result):
        ES_α = μ + σ · [-f_ν(q_p) · (ν + q_p²) / (ν - 1)] / p
    """
    p = 1.0 - alpha
    nu, mu, sig = t["nu"], t["mu"], t["sigma"]
    q   = stats.t.ppf(p, df=nu)
    var = mu + sig * q
    es  = mu + sig * (-stats.t.pdf(q, df=nu) * (nu + q ** 2) / (nu - 1)) / p
    return var, es


def var_es_numerical(pdf_fn: Callable, mu: float, sigma: float,
                     alpha: float, lower: float = -1.0) -> tuple:
    """
    Numerical VaR and ES via scipy.integrate for VG/NIG (Week 3+).

    Parameters
    ----------
    pdf_fn : callable — the model PDF f(x)
    mu, sigma : location and scale (for standardisation)
    alpha : confidence level
    lower : lower bound for integration (default -1.0, i.e. −100% daily return)
    """
    from scipy.optimize import brentq
    from scipy.integrate import quad

    p = 1.0 - alpha

    # VaR: find q such that CDF(q) = p
    cdf = lambda x: quad(pdf_fn, lower, x)[0]
    var = brentq(lambda x: cdf(x) - p, lower, mu)

    # ES: E[r | r <= var]
    numerator, _  = quad(lambda x: x * pdf_fn(x), lower, var)
    es = numerator / p
    return var, es


def risk_table(models: dict, alphas: tuple = (0.95, 0.99)) -> pd.DataFrame:
    """
    Build a summary risk measures table for multiple models.

    Parameters
    ----------
    models : dict of {label: (var_es_fn, params_dict)}
        e.g. {"Gaussian": (var_es_gaussian, g_params),
               "Student-t": (var_es_student_t, t_params)}
    alphas : confidence levels

    Returns
    -------
    pd.DataFrame indexed by confidence level
    """
    rows = []
    for alpha in alphas:
        row = {"Confidence": f"{int(alpha * 100)}%"}
        for label, (fn, params) in models.items():
            var, es = fn(params, alpha)
            row[f"VaR ({label})"] = var
            row[f"ES ({label})"]  = es
        rows.append(row)
    return pd.DataFrame(rows).set_index("Confidence")
