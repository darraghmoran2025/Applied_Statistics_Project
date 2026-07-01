"""
Numerical validation of the hand-written Lévy densities.

These checks were originally done interactively (Week 3/4); this file makes
them repeatable.  Run either way:

    python tests/test_densities.py
    pytest tests/

The PyTensor test is skipped automatically when pymc/pytensor are not
installed, so the file stays runnable in the light (numpy/scipy-only) env.
"""

import os
import sys

import numpy as np
from scipy import stats
from scipy.integrate import quad

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "week3", "code"))

from levy_models import (  # noqa: E402
    _logpdf_nig, _logpdf_vg, fit_laplace, laplace_var_es,
    simulate_nig, simulate_vg,
)

# Full-sample MLE values (Week 3) — realistic parameter magnitudes.
NIG = {"mu": 0.001111, "alpha": 52.341, "beta": -6.095, "delta": 0.007574}
VG = {"mu": 0.000907, "sigma": 0.011627, "theta": -0.000684, "nu": 1.17306}


def test_nig_logpdf_matches_scipy():
    """levy_models NIG log-density == scipy.stats.norminvgauss to ~1e-12.

    scipy parametrisation: a = alpha*delta, b = beta*delta, scale = delta.
    """
    x = np.linspace(-0.15, 0.15, 2001)
    ours = _logpdf_nig(x, NIG["mu"], NIG["alpha"], NIG["beta"], NIG["delta"])
    ref = stats.norminvgauss.logpdf(
        x, a=NIG["alpha"] * NIG["delta"], b=NIG["beta"] * NIG["delta"],
        loc=NIG["mu"], scale=NIG["delta"])
    assert np.max(np.abs(ours - ref)) < 1e-12


def test_vg_reduces_to_laplace():
    """VG(mu, sigma, theta=0, nu=1) is the Laplace with b = sigma/sqrt(2)."""
    mu, sigma = 0.0006, 0.0114
    x = np.linspace(-0.10, 0.10, 1001)
    x = x[np.abs(x - mu) > 1e-6]          # skip the kink at x = mu
    ours = _logpdf_vg(x, mu, sigma, 0.0, 1.0)
    ref = stats.laplace.logpdf(x, loc=mu, scale=sigma / np.sqrt(2.0))
    assert np.max(np.abs(ours - ref)) < 1e-9


def test_vg_integrates_to_one():
    f = lambda x: np.exp(_logpdf_vg(np.array([x]), VG["mu"], VG["sigma"],
                                    VG["theta"], VG["nu"]))[0]
    total, _ = quad(f, -0.5, 0.5, limit=200)
    assert abs(total - 1.0) < 1e-6


def test_nig_integrates_to_one():
    f = lambda x: np.exp(_logpdf_nig(np.array([x]), NIG["mu"], NIG["alpha"],
                                     NIG["beta"], NIG["delta"]))[0]
    total, _ = quad(f, -0.5, 0.5, limit=200)
    assert abs(total - 1.0) < 1e-6


def test_simulators_match_analytic_moments():
    """Mixture simulators reproduce the documented mean/variance formulas."""
    n = 2_000_000
    s_vg = simulate_vg(VG, n=n, seed=123)
    mean_vg = VG["mu"] + VG["theta"]
    var_vg = VG["sigma"]**2 + VG["theta"]**2 * VG["nu"]
    assert abs(s_vg.mean() - mean_vg) < 5e-5
    assert abs(s_vg.var() / var_vg - 1.0) < 5e-3

    s_nig = simulate_nig(NIG, n=n, seed=123)
    g = np.sqrt(NIG["alpha"]**2 - NIG["beta"]**2)
    mean_nig = NIG["mu"] + NIG["delta"] * NIG["beta"] / g
    var_nig = NIG["delta"] * NIG["alpha"]**2 / g**3
    assert abs(s_nig.mean() - mean_nig) < 5e-5
    assert abs(s_nig.var() / var_nig - 1.0) < 5e-3


def test_laplace_var_es_closed_form():
    """Closed-form Laplace VaR/ES vs direct numerical tail expectation."""
    lap = {"mu": 0.000602, "b": 0.008078}
    out = laplace_var_es(lap, alphas=(0.95, 0.975, 0.99))
    for a, res in out.items():
        p = 1.0 - a
        var_ref = stats.laplace.ppf(p, loc=lap["mu"], scale=lap["b"])
        assert abs(res["VaR"] - var_ref) < 1e-12
        # ES by quadrature: E[X | X <= VaR] = (1/p) ∫_{-inf}^{VaR} x f(x) dx
        f = lambda x: x * stats.laplace.pdf(x, loc=lap["mu"], scale=lap["b"])
        es_ref = quad(f, -2.0, var_ref, limit=200)[0] / p
        assert abs(res["ES"] - es_ref) < 1e-8


def test_fit_laplace_recovers_parameters():
    rng = np.random.default_rng(7)
    r = rng.laplace(0.0005, 0.008, size=200_000)
    fit = fit_laplace(r)
    assert abs(fit["mu"] - 0.0005) < 1e-4
    assert abs(fit["b"] - 0.008) < 1e-4


def test_pytensor_nig_logp_matches_scipy():
    """Week 4 PyTensor NIG logp == scipy (skipped without pymc installed)."""
    try:
        import pytensor
        import pytensor.tensor as pt
        sys.path.insert(0, os.path.join(_HERE, "..", "week4", "code"))
        from week4_bayesian import _nig_logp
    except Exception:
        print("  [skip] pymc/pytensor not available")
        return
    x = np.linspace(-0.15, 0.15, 501)
    xv = pt.dvector("x")
    fn = pytensor.function(
        [xv], _nig_logp(xv, NIG["mu"], NIG["alpha"], NIG["beta"], NIG["delta"]))
    ours = fn(x)
    ref = stats.norminvgauss.logpdf(
        x, a=NIG["alpha"] * NIG["delta"], b=NIG["beta"] * NIG["delta"],
        loc=NIG["mu"], scale=NIG["delta"])
    assert np.max(np.abs(ours - ref)) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__} ...", flush=True)
        t()
        print("  ok")
    print(f"\nAll {len(tests)} checks passed.")
