"""
utils/hessian.py
----------------
Numerical Hessian for MLE standard error extraction.
Used by all model fitting modules.
"""

import numpy as np


def numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Finite-difference approximation of the Hessian of f at point x.

    Uses the 4-point central-difference formula:
        H_{ij} ≈ [f(x+εe_i+εe_j) - f(x+εe_i-εe_j)
                  - f(x-εe_i+εe_j) + f(x-εe_i-εe_j)] / 4ε²

    Inverting H (the Hessian of the *negative* log-likelihood) gives the
    asymptotic variance-covariance matrix of the MLE.
    Standard errors are sqrt of the diagonal entries.
    """
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return H


def mle_standard_errors(neg_loglik, params: np.ndarray) -> np.ndarray:
    """
    Compute MLE standard errors from the observed Fisher Information.

    Parameters
    ----------
    neg_loglik : callable
        The negative log-likelihood function.
    params : np.ndarray
        MLE parameter vector at the optimum.

    Returns
    -------
    np.ndarray
        Standard errors for each parameter.
    """
    H = numerical_hessian(neg_loglik, params)
    cov = np.linalg.inv(H)
    return np.sqrt(np.diag(cov))
