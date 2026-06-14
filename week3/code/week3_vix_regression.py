"""
week3_vix_regression.py — VIX Regression on VG and NIG Parameters
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Objectives
----------
1. Fit VG and NIG to each calendar year's returns (2000–2024),
   yielding 25 annual parameter estimates per model.
2. Download annual average VIX levels from Yahoo Finance (^VIX).
3. Regress each parameter on average annual VIX via OLS.
4. Report regression coefficients, R², p-values, and 95% CIs.
5. Produce scatter + regression line plots.

The VIX is selected as the predictive variable because it is a
forward-looking implied-volatility measure derived from S&P 500 options,
providing a coherent economic link to the fat-tail parameter ν (VG),
the tail-heaviness parameter α (NIG), and the asymmetry parameters
θ (VG) and β (NIG).  The regression formalises the qualitative pattern
observed in the sub-period analysis: parameters shift systematically
across market regimes, and VIX captures that regime state quantitatively.

Run standalone:
    python week3_vix_regression.py
    python week3_vix_regression.py --save_dir ../figures

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
build_annual_panel(r_series)  → pd.DataFrame
    Fit VG and NIG to each calendar year; return a tidy DataFrame with
    columns: year, avg_vix, vg_sigma, vg_theta, vg_nu, vg_mu,
             nig_alpha, nig_beta, nig_delta, nig_mu.

run_regressions(panel_df)  → pd.DataFrame
    OLS regression of each parameter on avg_vix.
    Returns summary DataFrame: slope, intercept, R², p-value, 95% CI.

make_vix_plots(panel_df, reg_df, save_dir)
    Produces:
      week3_vix_regression.png   — 3×2 scatter+regression grid
                                   (VG: σ, θ, ν  |  NIG: α, β, δ)

run_vix_regression(r_series, save_dir)
    Master function.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE NAMES
──────────────
panel_df    pd.DataFrame with one row per calendar year, columns for
            fitted parameters and the year's average VIX level.
reg_df      pd.DataFrame summarising OLS results for each parameter.
slope       OLS slope: change in parameter per unit increase in VIX.
r_sq        R² of the regression: fraction of parameter variance
            explained by VIX variation.
p_value     Two-sided p-value for H₀: slope = 0.  Below 0.05 indicates
            a statistically significant linear relationship.
ci_lo/hi    95% confidence interval for the slope.
avg_vix     Mean VIX level over the calendar year (open-to-close scale,
            i.e., in VIX points, not percent).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from levy_models import fit_vg, fit_nig, MODEL_COLOURS, default_save_dir

import yfinance as yf


# ════════════════════════════════════════════════════════════════════════════
# DATA: ANNUAL PANEL
# ════════════════════════════════════════════════════════════════════════════

def build_annual_panel(r_series):
    """
    Fit VG and NIG to each calendar year; collect results with annual VIX.

    Years with fewer than 150 observations (e.g. partial years) are skipped.
    Fitting failures for individual years are recorded as NaN so the
    regression can proceed on the available data.

    Returns pd.DataFrame with columns:
        year, avg_vix,
        vg_sigma, vg_theta, vg_nu, vg_mu,
        nig_alpha, nig_beta, nig_delta, nig_mu
    """
    # Download VIX
    print("Downloading VIX (^VIX)…")
    vix_raw = yf.download("^VIX", start="2000-01-01", end="2024-12-31",
                           auto_adjust=True, progress=False)["Close"]
    if isinstance(vix_raw, pd.DataFrame):
        vix_raw = vix_raw.iloc[:, 0]
    vix_raw.index = pd.to_datetime(vix_raw.index)

    years   = sorted(r_series.index.year.unique())
    records = []

    for year in years:
        r_yr  = r_series[r_series.index.year == year].values
        vix_yr = vix_raw[vix_raw.index.year == year]

        if len(r_yr) < 150:
            print(f"  {year}: {len(r_yr)} obs — skipped")
            continue

        avg_vix = float(vix_yr.mean()) if len(vix_yr) > 0 else np.nan
        row     = {"year": year, "avg_vix": avg_vix}

        # VG
        try:
            vg = fit_vg(r_yr, n_starts=2)
            row.update({
                "vg_sigma": vg["sigma"],
                "vg_theta": vg["theta"],
                "vg_nu":    vg["nu"],
                "vg_mu":    vg["mu"],
            })
        except Exception as exc:
            warnings.warn(f"  VG fit failed for {year}: {exc}", RuntimeWarning)
            row.update({"vg_sigma": np.nan, "vg_theta": np.nan,
                        "vg_nu":   np.nan, "vg_mu":    np.nan})

        # NIG
        try:
            nig = fit_nig(r_yr, n_starts=2)
            row.update({
                "nig_alpha": nig["alpha"],
                "nig_beta":  nig["beta"],
                "nig_delta": nig["delta"],
                "nig_mu":    nig["mu"],
            })
        except Exception as exc:
            warnings.warn(f"  NIG fit failed for {year}: {exc}", RuntimeWarning)
            row.update({"nig_alpha": np.nan, "nig_beta": np.nan,
                        "nig_delta": np.nan, "nig_mu":   np.nan})

        records.append(row)
        print(f"  {year} done — avg VIX {avg_vix:.1f}  |  "
              f"VG ν={row.get('vg_nu', float('nan')):.3f}  "
              f"NIG α={row.get('nig_alpha', float('nan')):.1f}")

    return pd.DataFrame(records).set_index("year")


# ════════════════════════════════════════════════════════════════════════════
# OLS REGRESSIONS
# ════════════════════════════════════════════════════════════════════════════

_PARAM_LABELS = {
    "vg_sigma": ("VG σ",   "Scale (σ)"),
    "vg_theta": ("VG θ",   "Asymmetry (θ)"),
    "vg_nu":    ("VG ν",   "Variance rate (ν)"),
    "nig_alpha":("NIG α",  "Tail heaviness (α)"),
    "nig_beta": ("NIG β",  "Asymmetry (β)"),
    "nig_delta":("NIG δ",  "Scale (δ)"),
}


def run_regressions(panel_df):
    """
    OLS regression of each parameter on avg_vix using scipy.stats.linregress.

    Returns pd.DataFrame with columns:
        Parameter, slope, intercept, R², p-value, CI_lo, CI_hi
    where CI_lo/hi are the 95% confidence interval on the slope.
    """
    rows = []
    for col, (short_label, _) in _PARAM_LABELS.items():
        sub = panel_df[["avg_vix", col]].dropna()
        if len(sub) < 5:
            continue
        x = sub["avg_vix"].values
        y = sub[col].values

        slope, intercept, r_val, p_val, se_slope = scipy_stats.linregress(x, y)
        r_sq = r_val**2

        # 95% CI on slope: slope ± t_{0.975, n-2} × SE
        n    = len(sub)
        t_cv = scipy_stats.t.ppf(0.975, df=n - 2)
        ci_lo = slope - t_cv * se_slope
        ci_hi = slope + t_cv * se_slope

        rows.append({
            "Parameter":  short_label,
            "n_years":    n,
            "slope":      slope,
            "intercept":  intercept,
            "R²":         r_sq,
            "p-value":    p_val,
            "CI_lo (95%)": ci_lo,
            "CI_hi (95%)": ci_hi,
        })

    return pd.DataFrame(rows).set_index("Parameter")


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


def make_vix_plots(panel_df, reg_df, save_dir=None):
    """
    3×2 scatter + regression line grid.

    Left column: VG parameters (σ, θ, ν) vs avg VIX.
    Right column: NIG parameters (α, β, δ) vs avg VIX.

    Each panel annotates R² and p-value.  Individual years are labelled
    for the two highest-VIX years (GFC peak, COVID peak) to anchor the
    reader's intuition about which part of the parameter space each
    crisis occupies.
    """
    cols_vg  = ["vg_sigma",  "vg_theta",  "vg_nu"]
    cols_nig = ["nig_alpha", "nig_beta",  "nig_delta"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    pairs = list(zip(cols_vg, cols_nig))

    # Find the two highest-VIX years for annotations
    top2_years = panel_df["avg_vix"].nlargest(2).index.tolist()

    for row_i, (col_vg, col_nig) in enumerate(pairs):
        for col_j, col in enumerate([col_vg, col_nig]):
            ax    = axes[row_i, col_j]
            sub   = panel_df[["avg_vix", col]].dropna()
            x_all = sub["avg_vix"].values
            y_all = sub[col].values

            color = MODEL_COLOURS["VG"] if col_j == 0 else MODEL_COLOURS["NIG"]
            ax.scatter(x_all, y_all, color=color, alpha=0.7, s=40, zorder=2)

            # Label high-VIX years
            for yr in top2_years:
                if yr in sub.index:
                    ax.annotate(str(yr),
                                (sub.loc[yr, "avg_vix"], sub.loc[yr, col]),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=7, color="gray")

            # Regression line
            short_label = _PARAM_LABELS[col][0]
            if short_label in reg_df.index:
                reg_row = reg_df.loc[short_label]
                x_fit   = np.linspace(x_all.min(), x_all.max(), 200)
                y_fit   = reg_row["intercept"] + reg_row["slope"] * x_fit
                ax.plot(x_fit, y_fit, color="black", lw=1.5, ls="--", zorder=3)

                ax.annotate(
                    f"R² = {reg_row['R²']:.3f}\np = {reg_row['p-value']:.3f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

            ax.set_xlabel("Average annual VIX", fontsize=9)
            ax.set_ylabel(_PARAM_LABELS[col][1], fontsize=9)
            ax.set_title(f"{short_label} vs VIX", fontsize=10)

    fig.suptitle(
        "VG and NIG Parameters vs Annual Average VIX (2000–2024)\n"
        "Dashed line: OLS regression",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, save_dir, "week3_vix_regression.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_vix_regression(r_series, save_dir=None):
    """Build annual panel, run regressions, print table, save figure."""
    print("=" * 70)
    print("Week 3 — VIX Regression on VG and NIG Parameters")
    print("=" * 70)

    panel_df = build_annual_panel(r_series)

    print("\n" + "─" * 70)
    print("Annual parameter panel:")
    print("─" * 70)
    print(panel_df.to_string(float_format="{:.4f}".format))

    print("\n" + "─" * 70)
    print("OLS regression results (parameter ~ avg_vix):")
    print("─" * 70)
    reg_df = run_regressions(panel_df)
    print(reg_df.to_string(float_format="{:.4f}".format))

    print("\nEconomic interpretation guide:")
    print("  VG σ  vs VIX  — higher VIX → larger scale (expected +ve slope)")
    print("  VG θ  vs VIX  — higher VIX → more negative skew (expected −ve slope)")
    print("  VG ν  vs VIX  — higher VIX → smaller ν, fatter tails (expected −ve slope)")
    print("  NIG α vs VIX  — higher VIX → smaller α, heavier tails (expected −ve slope)")
    print("  NIG β vs VIX  — higher VIX → more negative β (expected −ve slope)")
    print("  NIG δ vs VIX  — higher VIX → larger δ (expected +ve slope)")

    make_vix_plots(panel_df, reg_df, save_dir=save_dir)

    print("\nVIX regression complete.")
    return panel_df, reg_df


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3: VIX regression")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save figures")
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir(__file__)
    r_series = fetch_returns()
    run_vix_regression(r_series, save_dir=save_dir)
