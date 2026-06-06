"""
Week 2 — Gaussian & Student-t MLE
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Objectives
----------
1. Download S&P 500 daily log-returns (Jan 2000 – Dec 2024) via yfinance
2. Fit Normal(μ, σ²) and Student-t(ν, μ, σ) by Maximum Likelihood
3. Extract standard errors from the observed Fisher Information matrix
4. Compute AIC and BIC for model comparison
5. Estimate VaR and ES at 95% and 99%
6. KS goodness-of-fit test on standardised residuals
7. Retrospective sub-period analysis across four market shock windows
8. Diagnostic plots: density overlay, QQ plots, return trace with shock shading

Install dependencies
--------------------
    pip install numpy pandas scipy yfinance matplotlib

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEGEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FUNCTIONS
─────────
fetch_returns(ticker, start, end)
    Downloads adjusted closing prices from Yahoo Finance and returns
    daily log-returns r_t = log(S_t / S_{t-1}) as a dated pd.Series.

fit_gaussian(r) → dict
    Closed-form MLE for the Normal distribution N(μ, σ²).
    Returns parameters, standard errors, log-likelihood, AIC, BIC.

fit_student_t(r) → dict
    Numerical MLE for the location-scale Student-t t(ν, μ, σ) using
    the L-BFGS-B optimiser. Returns the same fields as fit_gaussian
    plus the degrees-of-freedom parameter ν.

_neg_loglik_t(params, r)
    Internal helper: negative log-likelihood of the Student-t.
    Prefixed with _ to signal it is not called directly — it is passed
    as the objective function to scipy.optimize.minimize.

_numerical_hessian(f, x)
    Internal helper: approximates the matrix of second derivatives of f
    at point x using the 4-point central-difference formula. Used to
    compute standard errors after optimisation. Prefixed with _ for the
    same reason as above.

compute_risk_measures(g, t, alphas) → pd.DataFrame
    Computes VaR and ES at each confidence level in alphas for both the
    Gaussian (g) and Student-t (t) fitted models.

goodness_of_fit(r, g, t) → pd.DataFrame
    Runs a Kolmogorov-Smirnov test on the standardised residuals of
    each model to quantify how well the fitted distribution matches the
    empirical data.

fit_subperiods(r_series, periods) → pd.DataFrame
    Fits both models independently to each sub-period in the periods
    dict. Reveals how parameters shift during crises versus calm markets.

print_subperiod_summary(df)
    Prints the sub-period results table in a compact, human-readable
    format with annualised volatility for easier interpretation.

plot_marginals_by_year(r_series, save_dir)
    Produces a grid of annual kernel density plots — one panel per
    calendar year. Crisis years (2008, 2020) show wider distributions;
    calm years show narrow, peaked distributions. Saved as
    week2_marginals_by_year.png.

make_plots(r_series, g, t, save_dir)
    Produces six figures: the combined 4-panel overview (week2_plots.png)
    plus five individual files — one per panel and one for the annual
    marginals grid — for direct use in write-ups.

_save_fig(fig, save_dir, filename)
    Internal helper: saves a matplotlib figure to save_dir/filename at
    150 dpi. Prefixed with _ as it is not called directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIABLE NAMES
──────────────
r               Array of daily log-returns (numpy array).
r_series        Same data as a dated pd.Series (has a DatetimeIndex),
                used for time-series plots and sub-period slicing.
g               Dict of Gaussian MLE results returned by fit_gaussian().
t               Dict of Student-t MLE results returned by fit_student_t().
n               Number of observations.
k               Number of free parameters in the model (2 for Gaussian,
                3 for Student-t).
p               Left-tail probability = 1 − α (e.g. 0.05 at 95% level).
q_g, q_t        Quantile at probability p under the Gaussian / Student-t.
var_g, var_t    VaR under the Gaussian / Student-t model.
es_g,  es_t     Expected Shortfall under the Gaussian / Student-t model.
z_g,   z_t      Standardised residuals: (r − μ̂) / σ̂ for each model.
                Should follow N(0,1) or t(ν) if the model fits well.
H               Hessian matrix — the n×n matrix of second partial
                derivatives of the negative log-likelihood at the MLE.
                Its inverse is the asymptotic covariance matrix.
h               Per-parameter finite-difference step sizes (vector).
                Scaled as h_i = max(1e-6, 1e-4 × |θ_i|) so the step is
                always proportional to the size of each parameter.
se              Array of standard errors = sqrt(diag(H⁻¹)).
n_obs           Number of observations within a sub-period window.
loglik          Log-likelihood ℓ(θ̂) evaluated at the MLE.
init            Starting parameter values passed to the optimiser.
bounds          Box constraints on the parameters for L-BFGS-B:
                ν > 2.01 (finite variance), σ > 0.
result          The object returned by scipy.optimize.minimize containing
                the optimised parameters (result.x), objective value
                (result.fun), and convergence flag (result.success).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GREEK LETTERS AND STATISTICAL NOTATION
───────────────────────────────────────
μ  (mu)         Location parameter ≈ mean of the distribution.
σ  (sigma)      Scale parameter ≈ standard deviation. Note: for the
                Student-t, σ is NOT the true standard deviation; the
                true std dev = σ√(ν/(ν−2)), which is larger than σ.
ν  (nu)         Degrees of freedom of the Student-t. Controls tail
                heaviness: smaller ν → fatter tails. ν → ∞ recovers
                the Gaussian. Requires ν > 2 for finite variance.
α  (alpha)      Confidence level for VaR/ES (e.g. 0.95 = 95%).
φ(·)            PDF of the standard Normal distribution N(0,1).
Φ⁻¹(·)         Quantile function (inverse CDF) of N(0,1).
f_ν(·)          PDF of the standard Student-t with ν degrees of freedom.
θ̂  (theta-hat)  Generic notation for a vector of MLE parameter estimates.
ℓ(θ̂)           Log-likelihood evaluated at the MLE estimates.
SE(θ̂_i)        Standard error of the i-th parameter = √(H⁻¹)_{ii}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACRONYMS
────────
MLE         Maximum Likelihood Estimation — finds parameter values that
            maximise the probability of observing the data.
SE          Standard Error — uncertainty estimate for a fitted parameter.
VaR         Value-at-Risk — the loss level exceeded with probability 1−α.
ES          Expected Shortfall — the average loss conditional on exceeding
            VaR. Also called CVaR (Conditional VaR). More informative
            than VaR for extreme events; mandated by Basel III/IV (FRTB).
AIC         Akaike Information Criterion = 2k − 2ℓ(θ̂). Penalises model
            complexity; lower is better. Used to compare Gaussian vs
            Student-t without over-rewarding extra parameters.
BIC         Bayesian Information Criterion = k·ln(n) − 2ℓ(θ̂). Imposes
            a stronger penalty for extra parameters than AIC. Also lower
            is better.
KS          Kolmogorov-Smirnov — a non-parametric goodness-of-fit test.
            The test statistic D = max|F_n(x) − F(x)| measures the
            largest gap between the empirical and theoretical CDF.
QQ plot     Quantile-Quantile plot — compares the sorted sample values
            against theoretical quantiles. A straight diagonal line
            indicates a good fit; S-shaped deviation signals fat tails.
GFC         Global Financial Crisis (approximately 2007–2009).
FRTB        Fundamental Review of the Trading Book — Basel Committee
            regulation (BCBS 2013) that replaced VaR with ES as the
            primary capital metric.
L-BFGS-B    Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds
            — the quasi-Newton optimisation algorithm used for Student-t
            MLE. Handles box constraints (ν > 2.01, σ > 0) natively.
PDF         Probability Density Function.
CDF         Cumulative Distribution Function.
ddof        Delta degrees of freedom — the divisor used in std(). ddof=0
            divides by n (MLE / biased estimator); ddof=1 divides by n−1
            (sample / unbiased estimator). MLE uses ddof=0.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ════════════════════════════════════════════════════════════════════════════

# Four market shock periods studied throughout the project
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


def fetch_returns(ticker="^GSPC", start="2000-01-01", end="2024-12-31"):
    """Download adjusted closes and return daily log-returns as a dated pd.Series."""
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    returns = np.log(raw / raw.shift(1)).dropna()
    return returns


# ════════════════════════════════════════════════════════════════════════════
# 2. GAUSSIAN MLE
# ════════════════════════════════════════════════════════════════════════════

def fit_gaussian(r):
    """
    Closed-form MLE for N(μ, σ²).

    MLE uses the biased estimator (divides by n, not n-1).
    Standard errors from the Normal Fisher Information:
        SE(μ) = σ/√n,   SE(σ) = σ/√(2n)

    Returns dict with: mu, sigma, se_mu, se_sigma, loglik, aic, bic
    """
    r = np.asarray(r)
    n   = len(r)
    mu  = r.mean()
    sig = r.std(ddof=0)

    loglik = -0.5 * n * np.log(2 * np.pi * sig**2) - np.sum((r - mu)**2) / (2 * sig**2)
    k = 2
    return {
        "mu":       mu,       "se_mu":    sig / np.sqrt(n),
        "sigma":    sig,      "se_sigma": sig / np.sqrt(2 * n),
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(n) - 2 * loglik,
        "n_params": k,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. STUDENT-t MLE
# ════════════════════════════════════════════════════════════════════════════

def _neg_loglik_t(params, r):
    nu, mu, sigma = params
    if nu <= 2.0 or sigma <= 0.0:
        return np.inf
    return -np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma))


def _numerical_hessian(f, x):
    """
    Finite-difference Hessian using parameter-scaled step sizes.

    Step h_i = max(1e-6, 1e-4 * |x_i|) for each parameter i.
    A fixed step size degrades accuracy when parameters have very different
    magnitudes (e.g. ν ≈ 2.65, μ ≈ 6.6e-4, σ ≈ 7.1e-3).
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


def fit_student_t(r):
    """
    Numerical MLE for the location-scale Student-t t(ν, μ, σ).

    Constraint: ν > 2 (finite variance required), σ > 0.
    Optimiser: L-BFGS-B.
    Standard errors from the observed Fisher Information (numerical Hessian).

    Returns dict with: nu, mu, sigma, se_nu, se_mu, se_sigma, loglik, aic, bic
    """
    r = np.asarray(r)
    init   = np.array([5.0, r.mean(), r.std(ddof=0)])
    bounds = [(2.01, 200.0), (None, None), (1e-8, None)]

    result = minimize(
        _neg_loglik_t, init, args=(r,),
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 2000},
    )

    # ABNORMAL_TERMINATION_IN_LNSRCH occurs when ν is near the 2.01 boundary.
    # Retry from result.x with relaxed tolerance; the estimate is typically good.
    if not result.success:
        result = minimize(
            _neg_loglik_t, result.x, args=(r,),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-8, "gtol": 1e-5, "maxiter": 5000},
        )

    if not result.success:
        if np.isfinite(result.fun):
            warnings.warn(
                f"Student-t MLE convergence warning: {result.message.strip()}. "
                f"Using best available estimate (nu_hat = {result.x[0]:.4f}).",
                RuntimeWarning,
            )
        else:
            raise RuntimeError(f"Student-t MLE did not converge: {result.message}")

    nu, mu, sigma = result.x

    if nu < 2.5:
        warnings.warn(
            f"nu_hat = {nu:.4f} is close to the boundary nu = 2. "
            "Standard errors and ES estimates may be less reliable.",
            RuntimeWarning,
        )

    H  = _numerical_hessian(lambda p: _neg_loglik_t(p, r), result.x)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    loglik = -result.fun
    k = 3
    n = len(r)

    return {
        "nu":    nu,    "se_nu":    se[0],
        "mu":    mu,    "se_mu":    se[1],
        "sigma": sigma, "se_sigma": se[2],
        "loglik":   loglik,
        "aic":      2 * k - 2 * loglik,
        "bic":      k * np.log(n) - 2 * loglik,
        "n_params": k,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. VALUE-AT-RISK AND EXPECTED SHORTFALL
# ════════════════════════════════════════════════════════════════════════════

def compute_risk_measures(g, t, alphas=(0.95, 0.99)):
    """
    VaR and ES at each confidence level α under both fitted distributions.

    Convention: α = confidence level; p = 1 - α (tail probability).
    All figures are daily log-returns; negative values are losses.

    Gaussian ES:   μ - σ · φ(Φ⁻¹(p)) / p
    Student-t ES:  μ + σ · [-f_ν(q_p) · (ν + q_p²) / (ν - 1)] / p
    """
    rows = []
    for a in alphas:
        p = 1.0 - a

        q_g   = stats.norm.ppf(p)
        var_g = g["mu"] + g["sigma"] * q_g
        es_g  = g["mu"] - g["sigma"] * stats.norm.pdf(q_g) / p

        nu, mu, sig = t["nu"], t["mu"], t["sigma"]
        q_t   = stats.t.ppf(p, df=nu)
        var_t = mu + sig * q_t
        es_t  = mu + sig * (-stats.t.pdf(q_t, df=nu) * (nu + q_t**2) / (nu - 1)) / p

        rows.append({
            "Confidence":    f"{int(a * 100)}%",
            "VaR (Gaussian)":  var_g,
            "ES (Gaussian)":   es_g,
            "VaR (Student-t)": var_t,
            "ES (Student-t)":  es_t,
        })

    return pd.DataFrame(rows).set_index("Confidence")


# ════════════════════════════════════════════════════════════════════════════
# 5. GOODNESS-OF-FIT (KS TEST)
# ════════════════════════════════════════════════════════════════════════════

_LILLIEFORS_NOTE = (
    "* p-values are anti-conservative when parameters are estimated from the "
    "same data (Lilliefors effect). With n ≈ 6,300, even tiny departures "
    "from the fitted distribution reach statistical significance. "
    "The KS statistic D is the meaningful comparison metric: the relative "
    "reduction in D from Gaussian to Student-t quantifies how much of the "
    "non-Gaussianity is absorbed by the extra tail parameter ν."
)


def goodness_of_fit(r, g, t):
    """
    KS tests on standardised residuals for both fitted models.

    Standardisation: z = (r - μ) / σ, then KS test against:
      - N(0,1) for the Gaussian model
      - t(ν) for the Student-t model

    Returns a pd.DataFrame; see _LILLIEFORS_NOTE for the p-value caveat.
    """
    r = np.asarray(r)
    z_g = (r - g["mu"]) / g["sigma"]
    z_t = (r - t["mu"]) / t["sigma"]
    nu  = t["nu"]

    ks_g = stats.kstest(z_g, "norm")
    ks_t = stats.kstest(z_t, lambda x: stats.t.cdf(x, df=nu))

    df = pd.DataFrame([
        {"Model": "Gaussian",  "KS Statistic": ks_g.statistic, "p-value*": ks_g.pvalue, "n": len(r)},
        {"Model": "Student-t", "KS Statistic": ks_t.statistic, "p-value*": ks_t.pvalue, "n": len(r)},
    ]).set_index("Model")
    df.attrs["note"] = _LILLIEFORS_NOTE
    return df


# ════════════════════════════════════════════════════════════════════════════
# 6. RETROSPECTIVE SUB-PERIOD ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def fit_subperiods(r_series, periods=None):
    """
    Fit Gaussian and Student-t independently to each market shock window.

    Surfaces how μ, σ, and ν shift during crisis versus calm periods.
    For example: GFC and COVID produce ν close to 2 (near-infinite variance),
    while the dot-com crash and rate-hike cycle show ν ≈ 6.5.

    Parameters
    ----------
    r_series : pd.Series with DatetimeIndex
    periods  : dict of {name: (start, end)}; defaults to SHOCK_PERIODS

    Returns
    -------
    pd.DataFrame indexed by (period, model)
    """
    if periods is None:
        periods = SHOCK_PERIODS

    records = []
    for period_name, (start, end) in periods.items():
        r_sub = r_series.loc[start:end].values
        n_obs = len(r_sub)
        if n_obs < 30:
            continue

        for model_name, fit_fn in [("Gaussian", fit_gaussian),
                                    ("Student-t", fit_student_t)]:
            try:
                params = fit_fn(r_sub)
                row = {"period": period_name, "model": model_name, "n_obs": n_obs}
                for k, v in params.items():
                    if not k.startswith("se_") and k not in ("loglik", "aic", "bic", "n_params"):
                        row[k] = round(float(v), 6)
                records.append(row)
            except Exception as exc:
                records.append({"period": period_name, "model": model_name,
                                 "n_obs": n_obs, "error": str(exc)})

    return pd.DataFrame(records).set_index(["period", "model"])


def print_subperiod_summary(df):
    """Print a compact readable summary of sub-period parameter shifts."""
    prev_period = None
    for (period, model), row in df.iterrows():
        if period != prev_period:
            print(f"\n{period}:")
            prev_period = period
        if "error" in row and pd.notna(row.get("error")):
            print(f"    [{model}]  ERROR — {row['error']}")
            continue
        parts = [f"    [{model}]  n={int(row['n_obs'])}"]
        if "nu" in row and pd.notna(row.get("nu")):
            parts.append(f"ν={row['nu']:.3f}")
        if "mu" in row and pd.notna(row.get("mu")):
            parts.append(f"μ={float(row['mu']):+.4%}/day")
        if "sigma" in row and pd.notna(row.get("sigma")):
            ann = float(row["sigma"]) * np.sqrt(252)
            parts.append(f"σ={float(row['sigma']):.4%}/day  ({ann:.1%} ann.)")
        print("  ".join(parts))


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
#
# Output files produced by make_plots() when save_dir is set:
#
#   week2_plots.png              Combined 4-panel overview (main output)
#   week2_trace.png              Return trace with shock shading
#   week2_density.png            Density overlay — Gaussian vs Student-t
#   week2_qq_gaussian.png        QQ plot — Gaussian model
#   week2_qq_student_t.png       QQ plot — Student-t model
#   week2_marginals_by_year.png  Annual KDE grid — one panel per calendar year
#
# ════════════════════════════════════════════════════════════════════════════


def _save_fig(fig, save_dir, filename):
    """Save fig to save_dir/filename at 150 dpi if save_dir is set."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


def plot_marginals_by_year(r_series, save_dir=None):
    """
    Grid of annual kernel density plots — one panel per calendar year.

    Crisis years are highlighted with the same colours as the trace plot.
    Each panel annotates annualised volatility (σ × √252) for direct
    comparison across years.

    Saved as: week2_marginals_by_year.png
    """
    # Map each calendar year to its shock label (first match wins)
    year_shock = {}
    for label, (start, end) in SHOCK_PERIODS.items():
        for yr in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            year_shock.setdefault(yr, label)

    years = sorted(r_series.index.year.unique())
    ncols = 5
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(16, nrows * 3.0),
                             sharey=False)
    axes = axes.flatten()
    x = np.linspace(-0.12, 0.12, 400)

    for i, year in enumerate(years):
        ax = axes[i]
        r_yr = r_series[r_series.index.year == year].values
        if len(r_yr) < 20:
            ax.set_visible(False)
            continue

        shock_label = year_shock.get(year)
        if shock_label:
            ax.set_facecolor(SHOCK_COLOURS[shock_label])

        kde = stats.gaussian_kde(r_yr)
        ax.fill_between(x, kde(x), alpha=0.5, color="steelblue")
        ax.plot(x, kde(x), color="steelblue", lw=1.2)

        ann_vol = r_yr.std(ddof=0) * (252 ** 0.5)
        ax.set_title(str(year), fontsize=9, fontweight="bold",
                     color="black" if not shock_label else "#5a3e28")
        ax.annotate(f"σ={ann_vol:.0%}", xy=(0.97, 0.92),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=7, color="#333333")
        ax.set_xlim(-0.12, 0.12)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    legend_patches = [mpatches.Patch(color=SHOCK_COLOURS[lbl], alpha=0.7, label=lbl)
                      for lbl in SHOCK_PERIODS]
    fig.legend(handles=legend_patches, loc="lower right",
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.01))

    fig.suptitle("S&P 500 Annual Return Distributions (2000–2024)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, save_dir, "week2_marginals_by_year.png")
    plt.close(fig)


def make_plots(r_series, g, t, save_dir=None):
    """
    Produces and saves six figures.

    Combined overview (always shown):
        week2_plots.png              4-panel figure: trace + density + 2 QQ plots

    Individual panels (saved separately for write-up use):
        week2_trace.png              Return trace with shock shading
        week2_density.png            Density overlay
        week2_qq_gaussian.png        Gaussian QQ plot
        week2_qq_student_t.png       Student-t QQ plot
        week2_marginals_by_year.png  Annual KDE grid (via plot_marginals_by_year)
    """
    r     = r_series.values
    n     = len(r)
    probs = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    x     = np.linspace(r.min(), r.max(), 800)

    # Pre-compute standardised residuals and QQ data (shared by individual + combined)
    z_g  = (r - g["mu"]) / g["sigma"]
    z_t  = (r - t["mu"]) / t["sigma"]
    th_g = stats.norm.ppf(probs)
    th_t = stats.t.ppf(probs, df=t["nu"])
    lim_g = max(abs(th_g).max(), abs(np.sort(z_g)).max()) * 1.05
    lim_t = max(abs(th_t).max(), abs(np.sort(z_t)).max()) * 1.05

    shock_handles = [mpatches.Patch(color=SHOCK_COLOURS[k], alpha=0.6, label=k)
                     for k in SHOCK_PERIODS]

    if save_dir:
        print("\nSaving individual plot files:")

    # ── 1 of 5: Return trace ─────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(14, 4))
    for label, (start, end) in SHOCK_PERIODS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.35, color=SHOCK_COLOURS[label], zorder=0)
    ax.plot(r_series.index, r, lw=0.5, color="black", alpha=0.7, zorder=1)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily log-return")
    ax.set_title("S&P 500 Daily Log-Returns (2000–2024) — Market Shock Periods Shaded")
    ax.legend(handles=shock_handles, loc="upper right", fontsize=9, framealpha=0.85)
    fig1.tight_layout()
    _save_fig(fig1, save_dir, "week2_trace.png")
    plt.close(fig1)

    # ── 2 of 5: Density overlay ───────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.hist(r, bins=150, density=True,
            color="#cfe2f3", edgecolor="none", label="S&P 500 log-returns")
    ax.plot(x, stats.norm.pdf(x, g["mu"], g["sigma"]),
            color="steelblue", lw=2, label="Gaussian MLE")
    ax.plot(x, stats.t.pdf(x, t["nu"], t["mu"], t["sigma"]),
            color="crimson", lw=2, label=f"Student-t MLE  (ν = {t['nu']:.2f})")
    ax.set_xlim(-0.12, 0.12)
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution — Gaussian vs Student-t MLE (2000–2024)")
    ax.legend()
    fig2.tight_layout()
    _save_fig(fig2, save_dir, "week2_density.png")
    plt.close(fig2)

    # ── 3 of 5: QQ — Gaussian ────────────────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(th_g, np.sort(z_g), s=1.5, alpha=0.4, color="steelblue")
    ax.plot([-lim_g, lim_g], [-lim_g, lim_g], "k-", lw=0.8)
    ax.set_xlim(-lim_g, lim_g); ax.set_ylim(-lim_g, lim_g)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title("QQ Plot — Gaussian (2000–2024)")
    fig3.tight_layout()
    _save_fig(fig3, save_dir, "week2_qq_gaussian.png")
    plt.close(fig3)

    # ── 4 of 5: QQ — Student-t ───────────────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(th_t, np.sort(z_t), s=1.5, alpha=0.4, color="crimson")
    ax.plot([-lim_t, lim_t], [-lim_t, lim_t], "k-", lw=0.8)
    ax.set_xlim(-lim_t, lim_t); ax.set_ylim(-lim_t, lim_t)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title(f"QQ Plot — Student-t (ν = {t['nu']:.2f}, 2000–2024)")
    fig4.tight_layout()
    _save_fig(fig4, save_dir, "week2_qq_student_t.png")
    plt.close(fig4)

    # ── 5 of 5: Annual marginal distributions ─────────────────────────────────
    plot_marginals_by_year(r_series, save_dir=save_dir)

    # ── Combined 4-panel overview ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    for label, (start, end) in SHOCK_PERIODS.items():
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.35, color=SHOCK_COLOURS[label], zorder=0)
    ax1.plot(r_series.index, r, lw=0.5, color="black", alpha=0.7, zorder=1)
    ax1.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Daily log-return")
    ax1.set_title("S&P 500 Daily Log-Returns (2000–2024) — Market Shock Periods Shaded",
                  fontsize=11)
    ax1.legend(handles=shock_handles, loc="upper right", fontsize=8, framealpha=0.85)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.hist(r, bins=150, density=True,
             color="#cfe2f3", edgecolor="none", label="S&P 500 log-returns")
    ax2.plot(x, stats.norm.pdf(x, g["mu"], g["sigma"]),
             color="steelblue", lw=2, label="Gaussian MLE")
    ax2.plot(x, stats.t.pdf(x, t["nu"], t["mu"], t["sigma"]),
             color="crimson", lw=2, label=f"Student-t MLE  (ν = {t['nu']:.2f})")
    ax2.set_xlim(-0.12, 0.12)
    ax2.set_xlabel("Daily log-return"); ax2.set_ylabel("Density")
    ax2.set_title("Return Distribution — Gaussian vs Student-t MLE", fontsize=11)
    ax2.legend()

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(th_g, np.sort(z_g), s=1.5, alpha=0.4, color="steelblue")
    ax3.plot([-lim_g, lim_g], [-lim_g, lim_g], "k-", lw=0.8)
    ax3.set_xlim(-lim_g, lim_g); ax3.set_ylim(-lim_g, lim_g)
    ax3.set_xlabel("Theoretical quantiles"); ax3.set_ylabel("Sample quantiles")
    ax3.set_title("QQ Plot — Gaussian", fontsize=11)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(th_t, np.sort(z_t), s=1.5, alpha=0.4, color="crimson")
    ax4.plot([-lim_t, lim_t], [-lim_t, lim_t], "k-", lw=0.8)
    ax4.set_xlim(-lim_t, lim_t); ax4.set_ylim(-lim_t, lim_t)
    ax4.set_xlabel("Theoretical quantiles"); ax4.set_ylabel("Sample quantiles")
    ax4.set_title(f"QQ Plot — Student-t (ν = {t['nu']:.2f})", fontsize=11)

    fig.suptitle("Week 2 Diagnostics — S&P 500 Daily Log-Returns, 2000–2024",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, save_dir, "week2_plots.png")
    plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Week 2 — Gaussian & Student-t MLE")
    print("=" * 66)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    r_series = fetch_returns()
    r = r_series.values
    print(f"\nData: {len(r):,} daily log-returns  (S&P 500, 2000–2024)\n")

    # ── 2. Gaussian MLE ──────────────────────────────────────────────────────
    g = fit_gaussian(r)
    print("Gaussian MLE")
    print(f"  μ     = {g['mu']:+.6f}   SE = {g['se_mu']:.6f}")
    print(f"  σ     = {g['sigma']:.6f}    SE = {g['se_sigma']:.6f}")
    print(f"  Log-lik = {g['loglik']:.2f}   AIC = {g['aic']:.2f}   BIC = {g['bic']:.2f}")

    # ── 3. Student-t MLE ─────────────────────────────────────────────────────
    print("\nFitting Student-t (numerical optimisation, ~5 s)…")
    t = fit_student_t(r)
    print("Student-t MLE")
    print(f"  ν     = {t['nu']:.4f}      SE = {t['se_nu']:.4f}")
    print(f"  μ     = {t['mu']:+.6f}   SE = {t['se_mu']:.6f}")
    print(f"  σ     = {t['sigma']:.6f}    SE = {t['se_sigma']:.6f}")
    print(f"  Log-lik = {t['loglik']:.2f}   AIC = {t['aic']:.2f}   BIC = {t['bic']:.2f}")

    # ── 4. Risk measures ──────────────────────────────────────────────────────
    risk = compute_risk_measures(g, t)
    print("\nRisk Measures (daily log-returns; negative = loss):")
    print(risk.to_string(float_format="{:.6f}".format))

    # ── 5. Goodness-of-fit ────────────────────────────────────────────────────
    gof = goodness_of_fit(r, g, t)
    print("\nGoodness-of-Fit — KS tests on standardised residuals:")
    print(gof.to_string(float_format="{:.5f}".format))
    print(f"\n{gof.attrs['note']}")

    # ── 6. Sub-period retrospective ───────────────────────────────────────────
    all_periods = {"Full sample (2000–2024)": ("2000-01-01", "2024-12-31"),
                   **SHOCK_PERIODS}
    print("\n" + "─" * 66)
    print("Sub-period retrospective — Gaussian and Student-t per window:")
    print("─" * 66)
    sub_df = fit_subperiods(r_series, periods=all_periods)
    print_subperiod_summary(sub_df)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    make_plots(r_series, g, t, save_dir=script_dir)

    print("\nDone.")
