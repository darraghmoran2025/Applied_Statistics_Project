"""
week3_leadup_regression.py — Lead-Up Regression on Forward Risk
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

SCOPE AND GUARDRAILS (read first)
─────────────────────────────────
This module does NOT attempt to predict the direction or level of stock
returns.  That is outside the project and is not reliably possible.  The
dependent variable here is always a RISK / DISPERSION quantity — forward
realised volatility — never a return sign or level.  Forecasting the
dispersion of returns is legitimate because volatility clusters (Cont,
2001): large moves follow large moves.  This is the same empirical fact
the whole Basel VaR/ES apparatus rests on.

The analysis is framed as RETROSPECTIVE, IN-SAMPLE ATTRIBUTION, not
out-of-sample forecasting.  The question answered is:

    "Which observable, daily factors were systematically elevated in the
     run-up to historical episodes of extreme returns, and do tail-shape
     factors carry information about future risk beyond plain volatility?"

No out-of-sample skill is claimed, no trading PnL is computed, and no
held-out validation is presented as a forecasting scorecard.  The factor
trajectories overlaid on the four shock windows are a descriptive
diagnostic ("these quantities were already rising"), not an early-warning
system.

Objectives
----------
1. Build a daily feature panel from S&P 500 returns + VIX, with every
   predictor measured strictly at time t (knowable before the outcome).
2. Define the outcome as forward realised volatility over the next h days,
   RV_{t+1..t+h} — a forward RISK measure.
3. Fit a single OLS model on the full daily sample (n ≈ 6,000), with
   Newey–West HAC standard errors to correct for the serial correlation
   induced by overlapping forward windows.
4. Quantify whether tail-shape factors (rolling excess kurtosis, rolling
   skewness, drawdown) add explanatory power BEYOND a volatility-only
   baseline — the incremental-R² result is the bridge back to the Lévy
   tail story.
5. Produce a retrospective lead-up table: the average standardised level
   of each factor in the 21 trading days before each shock window.

Bridge to the Lévy framing
--------------------------
Two predictors — rolling excess kurtosis (kurt21) and rolling skewness
(skew21) — are short-window empirical analogues of the VG/NIG tail and
asymmetry parameters.  Including them lets the model speak the project's
own vocabulary: it is the dynamics of the *risk environment*, expressed in
tail terms, that the static-Gaussian risk numbers fail to capture.

Run standalone:
    python week3_leadup_regression.py
    python week3_leadup_regression.py --save_dir ../figures --horizon 21

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
build_feature_panel(r_series, horizon)  → pd.DataFrame
    Assemble the daily panel of lagged predictors and the forward
    realised-volatility target.

ols_hac(X, y, L)  → dict
    OLS with Newey–West HAC covariance (Bartlett kernel, bandwidth L).
    Returns coefficients, HAC SEs, t-stats, p-values, R², adj-R².

run_models(panel_df, horizon)  → (baseline, full, std_full)
    Fit the volatility-only baseline, the full model, and the fully
    standardised full model (for comparable coefficients).

leadup_table(panel_df, periods, window)  → pd.DataFrame
    Mean standardised factor level in the `window` trading days before
    each shock window start (retrospective attribution).

make_leadup_plots(panel_df, std_full, horizon, save_dir)
    week3_leadup_factors.png    — standardised factor trajectories + shocks
    week3_leadup_fit.png        — actual vs in-sample-fitted forward vol
    week3_leadup_coefs.png      — standardised coefficients with HAC 95% CI

run_leadup_regression(r_series, horizon, save_dir)
    Master function.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE NAMES
──────────────
horizon (h)  Forward window length in trading days (default 21 ≈ 1 month).
rv5, rv21    Trailing 5- and 21-day annualised realised volatility at t.
vix          VIX level at t (points).
dvix5        5-day change in VIX at t (momentum of fear).
absret       |r_t|, yesterday's absolute shock.
skew21       Trailing 21-day sample skewness of returns (asymmetry proxy).
kurt21       Trailing 21-day excess kurtosis (rolling tail-heaviness proxy;
             the Lévy bridge).
ddown        Drawdown from trailing 252-day peak (≤ 0).
fwd_rv       Forward realised vol over t+1..t+h, annualised (the target).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_week2():
    """Locate week2_gaussian_student_mle.py in either repo layout."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "week2", "code"),
        os.path.join(here, "..", "Week2"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "week2_gaussian_student_mle.py")):
            return c
    raise FileNotFoundError(
        "week2_gaussian_student_mle.py not found. "
        "Looked in: " + ", ".join(candidates)
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _find_week2())
sys.path.insert(0, _HERE)

from week2_gaussian_student_mle import fetch_returns
from levy_models import SHOCK_PERIODS, SHOCK_COLOURS, default_save_dir

import yfinance as yf

ANNUALISE = np.sqrt(252.0)

# Factor display labels (order defines the model's design matrix order)
_FACTORS = [
    ("rv21",   "Trailing 21d vol"),
    ("vix",    "VIX level"),
    ("dvix5",  "ΔVIX (5d)"),
    ("absret", "|return| (1d)"),
    ("skew21", "Rolling skew (21d)"),
    ("kurt21", "Rolling kurtosis (21d)"),
    ("ddown",  "Drawdown (252d)"),
]
_FACTOR_KEYS = [k for k, _ in _FACTORS]
_BASELINE_KEYS = ["rv21"]


# ════════════════════════════════════════════════════════════════════════════
# FEATURE PANEL
# ════════════════════════════════════════════════════════════════════════════

def build_feature_panel(r_series, horizon=21):
    """
    Build the daily panel of lagged predictors and the forward-vol target.

    Every predictor is computed from information available up to and
    including day t.  The target uses only days t+1 … t+h, so there is a
    strict temporal gap between predictors and outcome — the feature that
    makes this a lead-up regression rather than a contemporaneous one.

    Returns a pd.DataFrame indexed by date with columns:
        rv5, rv21, vix, dvix5, absret, skew21, kurt21, ddown, fwd_rv
    Rows with any NaN (warm-up period and the final h days) are dropped.
    """
    r = r_series.copy()
    df = pd.DataFrame(index=r.index)

    # ── Predictors at time t (trailing only) ──────────────────────────────
    df["rv5"]  = r.rolling(5).std()  * ANNUALISE
    df["rv21"] = r.rolling(21).std() * ANNUALISE
    df["absret"] = r.abs()
    df["skew21"] = r.rolling(21).skew()
    df["kurt21"] = r.rolling(21).kurt()          # excess kurtosis (Fisher)

    # Drawdown from trailing 252-day peak, reconstructed from cum-returns
    price = np.exp(r.cumsum())
    roll_max = price.rolling(252, min_periods=60).max()
    df["ddown"] = price / roll_max - 1.0

    # VIX level and 5-day change
    print("Downloading VIX (^VIX)…")
    vix = yf.download("^VIX", start="2000-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    vix = vix.reindex(df.index).ffill()
    df["vix"]   = vix
    df["dvix5"] = vix - vix.shift(5)

    # ── Forward realised volatility over t+1 … t+h (the RISK target) ──────
    fwd_var = (r**2).shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    df["fwd_rv"] = np.sqrt(fwd_var * 252.0 / horizon)

    cols = ["rv5", "rv21", "vix", "dvix5", "absret",
            "skew21", "kurt21", "ddown", "fwd_rv"]
    panel = df[cols].dropna()
    return panel


# ════════════════════════════════════════════════════════════════════════════
# OLS WITH NEWEY–WEST HAC COVARIANCE
# ════════════════════════════════════════════════════════════════════════════

def ols_hac(X, y, L):
    """
    OLS of y on X (intercept added internally) with Newey–West HAC SEs.

    Overlapping forward windows make the residuals strongly serially
    correlated, so ordinary OLS standard errors are badly understated.
    The Newey–West estimator with a Bartlett kernel and bandwidth L
    corrects for autocorrelation up to lag L; L is set to the forecast
    horizon, the mechanical overlap length.

        Cov_HAC = (X'X)^-1 [ S0 + Σ_{l=1}^L w_l (S_l + S_l') ] (X'X)^-1
        w_l = 1 − l/(L+1)              (Bartlett weights)
        S_l = Σ_t (x_t e_t)(x_{t-l} e_{t-l})'

    Parameters
    ----------
    X : (n, k) ndarray of predictors (no intercept column)
    y : (n,)   ndarray outcome
    L : int    HAC bandwidth (lags)

    Returns
    -------
    dict: beta, se, t, p, r2, adj_r2, names_have_const=True, n, k
          (beta[0] is the intercept)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])           # add intercept
    k = Xc.shape[1]

    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y
    resid = y - Xc @ beta

    # HAC meat matrix S
    u = Xc * resid[:, None]                          # x_t * e_t
    S = u.T @ u                                       # lag-0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)

    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    tss = np.sum((y - y.mean())**2)
    rss = np.sum(resid**2)
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / se
    pval = 2.0 * scipy_stats.t.sf(np.abs(tstat), df=n - k)

    return {
        "beta": beta, "se": se, "t": tstat, "p": pval,
        "r2": r2, "adj_r2": adj_r2, "n": n, "k": k, "cov": cov,
    }


# ════════════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ════════════════════════════════════════════════════════════════════════════

def _standardise(frame):
    """Return z-scored copy of `frame` plus the (mean, std) used."""
    mu = frame.mean()
    sd = frame.std(ddof=0)
    return (frame - mu) / sd, mu, sd


def run_models(panel_df, horizon=21):
    """
    Fit three specifications and return them.

    baseline   forward vol ~ trailing 21d vol            (volatility-only)
    full       forward vol ~ all seven factors (raw)
    std_full   forward vol ~ all seven factors, fully standardised
               (coefficients are directly comparable; this is what the
                coefficient figure and the lead-up table use)

    The headline result is the incremental R² of `full` over `baseline`:
    how much forecastable-risk variation the tail/asymmetry/drawdown
    factors explain beyond plain volatility persistence.
    """
    L = horizon  # HAC bandwidth = overlap length

    y = panel_df["fwd_rv"].values
    Xb = panel_df[_BASELINE_KEYS].values
    Xf = panel_df[_FACTOR_KEYS].values

    baseline = ols_hac(Xb, y, L)
    full     = ols_hac(Xf, y, L)

    # Fully standardised version for comparable coefficients
    z_X, _, _ = _standardise(panel_df[_FACTOR_KEYS])
    z_y = (panel_df["fwd_rv"] - panel_df["fwd_rv"].mean()) / panel_df["fwd_rv"].std(ddof=0)
    std_full = ols_hac(z_X.values, z_y.values, L)

    baseline["labels"] = ["const"] + _BASELINE_KEYS
    full["labels"]     = ["const"] + _FACTOR_KEYS
    std_full["labels"] = ["const"] + _FACTOR_KEYS
    return baseline, full, std_full


# ════════════════════════════════════════════════════════════════════════════
# RETROSPECTIVE LEAD-UP TABLE
# ════════════════════════════════════════════════════════════════════════════

def leadup_table(panel_df, periods=None, window=21):
    """
    Mean standardised factor level in the `window` trading days BEFORE each
    shock window start.

    Each factor is z-scored over the full sample, so a value of, say, +1.8
    means that factor stood 1.8 sample standard deviations above its
    typical level in the run-up to that crisis.  This is the retrospective
    attribution exhibit: it describes the conditions that preceded each
    episode in the historical record.  It is NOT a forecast.
    """
    if periods is None:
        periods = SHOCK_PERIODS

    z, _, _ = _standardise(panel_df[_FACTOR_KEYS])
    z = z.copy()
    rows = []
    for name, (start, _end) in periods.items():
        start_ts = pd.Timestamp(start)
        pre = z[z.index < start_ts].tail(window)
        if len(pre) == 0:
            continue
        row = {"Shock": name, "n_days": len(pre)}
        for kkey in _FACTOR_KEYS:
            row[kkey] = float(pre[kkey].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Shock")


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


# Saturated shock-window shading. The shared SHOCK_COLOURS pastels wash out once
# a PNG is flattened to RGB-on-white, so the lead-up figures use stronger, clearly
# distinct colours.
SHADE_COLOURS = {
    "Dot-com crash":  "#f4b400",   # amber
    "GFC":            "#e2504e",   # red
    "COVID-19":       "#1f9ec4",   # cyan
    "Fed rate hikes": "#3fa34d",   # green
}


def _shade_shocks(ax, periods, alpha=0.33):
    for name, (start, end) in periods.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SHADE_COLOURS[name], alpha=alpha, lw=0, zorder=0)


def make_leadup_plots(panel_df, std_full, horizon, save_dir=None):
    """Produce the three lead-up figures (see module docstring)."""
    periods = SHOCK_PERIODS

    # ── 1. Standardised factor trajectories with shock shading ────────────
    z, _, _ = _standardise(panel_df[_FACTOR_KEYS])
    show = ["rv21", "kurt21", "dvix5", "ddown"]
    show_lbl = {k: dict(_FACTORS)[k] for k in show}
    fig1, axes = plt.subplots(len(show), 1, figsize=(13, 11), sharex=True)
    for ax, kkey in zip(axes, show):
        _shade_shocks(ax, periods)
        ax.plot(z.index, z[kkey].values, color="#111111", lw=1.0, zorder=3)
        ax.axhline(0, color="gray", lw=0.7, ls=":", zorder=2)
        ax.grid(axis="y", color="0.9", lw=0.6, zorder=1)
        ax.set_ylabel(show_lbl[kkey], fontsize=12)
        ax.set_ylim(-3.5, 6)
        ax.margins(x=0.01)
    handles = [mpatches.Patch(facecolor=SHADE_COLOURS[n], alpha=0.55, label=n)
               for n in periods]
    fig1.suptitle("Standardised risk factors over time (shaded = shock windows)",
                  fontsize=13, y=0.995)
    fig1.legend(handles=handles, fontsize=11, ncol=4, loc="upper center",
                bbox_to_anchor=(0.5, 0.945), frameon=False)
    fig1.tight_layout(rect=[0, 0, 1, 0.91])
    _save_fig(fig1, save_dir, "week3_leadup_factors.png")
    plt.close(fig1)

    # ── 2. Actual vs in-sample fitted forward vol ─────────────────────────
    Xf = panel_df[_FACTOR_KEYS].values
    n = len(Xf)
    Xc = np.column_stack([np.ones(n), Xf])
    # refit raw full model to get fitted values
    full_raw = ols_hac(Xf, panel_df["fwd_rv"].values, horizon)
    fitted = Xc @ full_raw["beta"]
    fig2, ax = plt.subplots(figsize=(13, 5.5))
    _shade_shocks(ax, periods)
    ax.plot(panel_df.index, panel_df["fwd_rv"].values * 100,
            color="black", lw=0.8, label="Actual forward 21d vol (ann., %)")
    ax.plot(panel_df.index, fitted * 100,
            color="crimson", lw=0.9, alpha=0.85, label="In-sample fitted")
    ax.set_ylabel("Annualised vol (%)", fontsize=11)
    ax.margins(x=0.01)
    ax.set_title(
        f"Forward {horizon}-day realised volatility: actual vs in-sample fit "
        f"(R² = {full_raw['r2']:.2f})", fontsize=12)
    line_handles, _ = ax.get_legend_handles_labels()
    shock_handles = [mpatches.Patch(facecolor=SHADE_COLOURS[n], alpha=0.55, label=n)
                     for n in periods]
    ax.legend(handles=line_handles + shock_handles, fontsize=9,
              loc="upper center", ncol=3, framealpha=0.9)
    fig2.tight_layout()
    _save_fig(fig2, save_dir, "week3_leadup_fit.png")
    plt.close(fig2)

    # ── 3. Standardised coefficients with HAC 95% CI ──────────────────────
    beta = std_full["beta"][1:]          # drop intercept
    se   = std_full["se"][1:]
    labels = [dict(_FACTORS)[k] for k in _FACTOR_KEYS]
    ci = 1.96 * se
    order = np.argsort(beta)
    fig3, ax = plt.subplots(figsize=(9, 5))
    ypos = np.arange(len(beta))
    colors = ["crimson" if b < 0 else "steelblue" for b in beta[order]]
    ax.barh(ypos, beta[order], xerr=ci[order], color=colors, alpha=0.8,
            error_kw=dict(ecolor="gray", lw=1, capsize=3))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax.set_xlabel("Standardised coefficient (σ of forward vol per σ of factor)",
                  fontsize=9)
    ax.set_title("Lead-up regression: standardised factor loadings\n"
                 "(Newey–West HAC 95% intervals)", fontsize=11)
    fig3.tight_layout()
    _save_fig(fig3, save_dir, "week3_leadup_coefs.png")
    plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _print_model(name, res):
    print(f"\n{name}")
    print(f"  n = {res['n']:,}   R² = {res['r2']:.4f}   adj-R² = {res['adj_r2']:.4f}")
    print(f"  {'term':<22}{'coef':>12}{'HAC SE':>12}{'t':>9}{'p':>10}")
    for lab, b, s, t, p in zip(res["labels"], res["beta"],
                               res["se"], res["t"], res["p"]):
        print(f"  {lab:<22}{b:>12.5f}{s:>12.5f}{t:>9.2f}{p:>10.4f}")


# ════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_leadup_regression(r_series, horizon=21, save_dir=None):
    """Build panel, fit models, print tables, save figures."""
    print("=" * 70)
    print("Week 3 — Lead-Up Regression on Forward Realised Volatility")
    print("Target: forward RISK (volatility), not return. Retrospective, in-sample.")
    print("=" * 70)

    panel_df = build_feature_panel(r_series, horizon=horizon)
    print(f"\nPanel: {len(panel_df):,} daily observations, "
          f"{panel_df.index.min().date()} → {panel_df.index.max().date()}")

    baseline, full, std_full = run_models(panel_df, horizon=horizon)

    _print_model("Baseline (forward vol ~ trailing 21d vol):", baseline)
    _print_model("Full model (raw units):", full)
    _print_model("Full model (standardised — comparable loadings):", std_full)

    incr = full["r2"] - baseline["r2"]
    print(f"\nIncremental R² of tail/asymmetry/drawdown factors over the "
          f"volatility-only baseline: {incr:+.4f}")
    print(f"  Baseline R² = {baseline['r2']:.4f}  →  Full R² = {full['r2']:.4f}")

    print("\n" + "─" * 70)
    print("Retrospective lead-up: mean standardised factor level in the 21")
    print("trading days BEFORE each shock window (z-scores; descriptive only):")
    print("─" * 70)
    lt = leadup_table(panel_df, window=21)
    print(lt.to_string(float_format="{:+.2f}".format))

    make_leadup_plots(panel_df, std_full, horizon, save_dir=save_dir)

    print("\nLead-up regression complete.")
    return panel_df, baseline, full, std_full, lt


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3: lead-up regression on forward risk")
    parser.add_argument("--save_dir", default=None, help="Directory to save figures")
    parser.add_argument("--horizon", type=int, default=21,
                        help="Forward window in trading days (default 21)")
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir(__file__)
    r_series = fetch_returns()
    run_leadup_regression(r_series, horizon=args.horizon, save_dir=save_dir)
