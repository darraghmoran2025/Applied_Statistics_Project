"""
Week 2 — Gaussian & Student-t MLE
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Objectives
----------
1. Download S&P 500 daily log-returns (Jan 2000 – Dec 2024) via yfinance
2. Fit Normal(μ, σ²) and Student-t(ν, μ, σ) distributions by Maximum Likelihood
3. Extract standard errors via the Fisher Information matrix (numerical Hessian)
4. Estimate Value-at-Risk (VaR) and Expected Shortfall (ES) at 95% and 99%
5. Produce diagnostic plots: density overlay + QQ plots

Install dependencies
--------------------
    pip install numpy pandas scipy yfinance matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ════════════════════════════════════════════════════════════════════════════

def fetch_returns(ticker: str = "^GSPC",
                  start:  str = "2000-01-01",
                  end:    str = "2024-12-31") -> np.ndarray:
    """
    Download adjusted closing prices and compute continuously-compounded
    daily log-returns:  r_t = log(S_t / S_{t-1})
    """
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    # yfinance >=0.2.40 returns a DataFrame; squeeze to Series if needed
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    returns = np.log(raw / raw.shift(1)).dropna().values
    return returns


# ════════════════════════════════════════════════════════════════════════════
# 2. GAUSSIAN MLE
# ════════════════════════════════════════════════════════════════════════════

def fit_gaussian(r: np.ndarray) -> dict:
    """
    Closed-form MLE for the Normal distribution N(μ, σ²).

    The MLE for μ is the sample mean; for σ the MLE uses the biased
    denominator n (not n−1).  Standard errors come from the Normal
    Fisher Information:
        Var(μ̂) = σ²/n   →   SE(μ̂) = σ/√n
        Var(σ̂) = σ²/2n  →   SE(σ̂) = σ/√(2n)
    """
    n   = len(r)
    mu  = r.mean()
    sig = r.std(ddof=0)           # MLE: biased estimator divides by n
    return {
        "mu":       mu,
        "sigma":    sig,
        "se_mu":    sig / np.sqrt(n),
        "se_sigma": sig / np.sqrt(2 * n),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. STUDENT-t MLE
# ════════════════════════════════════════════════════════════════════════════

def _neg_loglik_t(params: np.ndarray, r: np.ndarray) -> float:
    """Negative log-likelihood for the location-scale Student-t distribution."""
    nu, mu, sigma = params
    if nu <= 2.0 or sigma <= 0.0:     # ν > 2 required for finite variance
        return np.inf
    return -np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma))


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Finite-difference approximation of the Hessian of f at point x.

    Uses the 4-point central-difference formula for each (i,j) entry:
        H_{ij} ≈ [f(x+εe_i+εe_j) - f(x+εe_i-εe_j)
                  - f(x-εe_i+εe_j) + f(x-εe_i-εe_j)] / 4ε²

    Inverting H (the Hessian of the *negative* log-likelihood) gives the
    asymptotic variance-covariance matrix of the MLE — the Cramér–Rao bound.
    Standard errors are the square roots of the diagonal entries.
    """
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
    return H


def fit_student_t(r: np.ndarray) -> dict:
    """
    Numerical MLE for the location-scale Student-t distribution t(ν, μ, σ).

    Parameters
    ----------
    r : array of log-returns

    Returns
    -------
    dict containing ν (degrees of freedom), μ (location), σ (scale),
    and their asymptotic standard errors.

    Optimiser: L-BFGS-B — a bounded quasi-Newton method.
    Starting values: ν=5 (prior: moderately heavy tails), μ = x̄, σ = s.
    Constraints: ν > 2 (finite variance), σ > 0.
    """
    init   = np.array([5.0, r.mean(), r.std()])
    bounds = [(2.01, 200.0), (None, None), (1e-8, None)]

    result = minimize(
        _neg_loglik_t, init, args=(r,),
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Student-t MLE did not converge: {result.message}")

    nu, mu, sigma = result.x

    # Standard errors from the observed Fisher Information
    H  = _numerical_hessian(lambda p: _neg_loglik_t(p, r), result.x)
    se = np.sqrt(np.diag(np.linalg.inv(H)))

    return {
        "nu":    nu,    "se_nu":    se[0],
        "mu":    mu,    "se_mu":    se[1],
        "sigma": sigma, "se_sigma": se[2],
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. VALUE-AT-RISK AND EXPECTED SHORTFALL
# ════════════════════════════════════════════════════════════════════════════

def compute_risk_measures(g: dict, t: dict,
                          alphas: tuple = (0.95, 0.99)) -> pd.DataFrame:
    """
    VaR and ES at each confidence level α for both fitted distributions.

    Convention
    ----------
    α = confidence level (e.g. 0.95); p = 1−α is the left-tail probability.
    Measures are expressed as daily log-returns; negative values are losses.

    Formulas
    --------
    VaR_α  =  μ + σ · F⁻¹(p)           [quantile of fitted distribution]

    Gaussian ES (conditional expectation of truncated Normal):
        ES_α  =  μ − σ · φ(Φ⁻¹(p)) / p

    Student-t ES (standard result for the t-distribution):
        ES_α  =  μ + σ · [−f_ν(q_p) · (ν + q_p²) / (ν − 1)] / p
        where q_p = t_ν⁻¹(p)

    Both ES formulas equal E[r | r ≤ VaR_α], the average loss in the tail.
    """
    rows = []
    for a in alphas:
        p = 1.0 - a

        # — Gaussian —
        q_g   = stats.norm.ppf(p)
        var_g = g["mu"] + g["sigma"] * q_g
        es_g  = g["mu"] - g["sigma"] * stats.norm.pdf(q_g) / p

        # — Student-t —
        nu, mu, sig = t["nu"], t["mu"], t["sigma"]
        q_t   = stats.t.ppf(p, df=nu)
        var_t = mu + sig * q_t
        es_t  = mu + sig * (-stats.t.pdf(q_t, df=nu) * (nu + q_t**2) / (nu - 1)) / p

        rows.append({
            "Confidence":    f"{int(a * 100)}%",
            "VaR_Gaussian":  var_g,
            "ES_Gaussian":   es_g,
            "VaR_StudentT":  var_t,
            "ES_StudentT":   es_t,
        })

    return pd.DataFrame(rows).set_index("Confidence")


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def make_plots(r: np.ndarray, g: dict, t: dict,
               save_path: str = "week2_plots.png") -> None:
    """
    2×2 figure layout:
      Row 1 (full width) : histogram of daily returns with Gaussian and
                           Student-t PDFs overlaid.
      Row 2 left         : QQ plot against the Normal distribution.
                           A straight-line fit means Normal is appropriate;
                           an S-shaped deviation signals fat tails.
      Row 2 right        : QQ plot against the fitted Student-t distribution.
                           Better alignment here confirms fat tails are captured.
    """
    x   = np.linspace(r.min(), r.max(), 800)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38)

    # ── Panel 1: density overlay ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(r, bins=150, density=True,
             color="#cfe2f3", edgecolor="none", label="S&P 500 log-returns")
    ax1.plot(x, stats.norm.pdf(x, g["mu"], g["sigma"]),
             color="steelblue", lw=2, label="Gaussian MLE")
    ax1.plot(x, stats.t.pdf(x, t["nu"], t["mu"], t["sigma"]),
             color="crimson",   lw=2,
             label=f"Student-t MLE  (ν = {t['nu']:.2f})")
    ax1.set_xlim(-0.12, 0.12)
    ax1.set_xlabel("Daily log-return")
    ax1.set_ylabel("Density")
    ax1.set_title(
        "S&P 500 Daily Log-Returns — Gaussian vs Student-t MLE (2000–2024)",
        fontsize=12)
    ax1.legend()

    # ── Panel 2: Normal QQ ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    # Standardise using Gaussian MLE parameters before plotting
    std_g = (r - g["mu"]) / g["sigma"]
    stats.probplot(std_g, dist="norm", plot=ax2)
    ax2.set_title("QQ Plot — Gaussian")

    # ── Panel 3: Student-t QQ ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    std_t = (r - t["mu"]) / t["sigma"]
    stats.probplot(std_t, dist=stats.t(df=t["nu"]), plot=ax3)
    ax3.set_title(f"QQ Plot — Student-t  (ν = {t['nu']:.2f})")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("Week 2 — Gaussian & Student-t MLE")
    print("=" * 62)

    # 1. Data
    r = fetch_returns()
    print(f"\nData: {len(r):,} daily log-returns  (S&P 500, 2000–2024)\n")

    # 2. Gaussian MLE
    g = fit_gaussian(r)
    print("Gaussian MLE")
    print(f"  μ     = {g['mu']:+.6f}   SE = {g['se_mu']:.6f}")
    print(f"  σ     = {g['sigma']:.6f}    SE = {g['se_sigma']:.6f}")

    # 3. Student-t MLE
    print("\nFitting Student-t (numerical optimisation, ~5 s)…")
    t = fit_student_t(r)
    print("Student-t MLE")
    print(f"  ν     = {t['nu']:.4f}      SE = {t['se_nu']:.4f}")
    print(f"  μ     = {t['mu']:+.6f}   SE = {t['se_mu']:.6f}")
    print(f"  σ     = {t['sigma']:.6f}    SE = {t['se_sigma']:.6f}")

    # 4. Risk measures
    risk = compute_risk_measures(g, t)
    print("\nRisk Measures (daily log-returns; negative = loss):")
    print(risk.to_string(float_format="{:.6f}".format))

    # 5. Plots
    import os
    out = os.path.join(os.path.dirname(__file__), "week2_plots.png")
    make_plots(r, g, t, save_path=out)

    print("\nDone.")
