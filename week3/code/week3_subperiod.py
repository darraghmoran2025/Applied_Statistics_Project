"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
fit_all_models_subperiod(r_series, periods)  → pd.DataFrame
    Fits all four models to each window.  Returns a long-format DataFrame
    indexed by (period, model) with key parameter columns.

empirical_shock_stats(r_series, periods)  → pd.DataFrame
    Computes descriptive statistics for each window: n, mean, volatility,
    skewness, excess kurtosis, VaR-5% empirical, max drawdown,
    proportion of extreme observations |r| > 2σ_full.

aic_improvement_table(sub_df)  → pd.DataFrame
    For each window: AIC of Gaussian minus AIC of each Lévy model.
    Positive values = Lévy model preferred.  Shows whether tail-fit
    benefit is concentrated in crisis windows.

run_subperiod_analysis(r_series, save_dir)
    Master function: fits, prints, saves figure.

make_subperiod_plots(sub_df, emp_stats, save_dir)
    Produces:
      week3_subperiod_params.png   — grouped bar chart of σ and ν across windows
      week3_subperiod_aic.png      — AIC improvement of Lévy vs Gaussian per window
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

from week2_gaussian_student_mle import (
    fetch_returns, fit_gaussian, fit_student_t,
)
from levy_models import (
    fit_vg, fit_nig, SHOCK_PERIODS, SHOCK_COLOURS, MODEL_COLOURS,
    default_save_dir,
)


# ════════════════════════════════════════════════════════════════════════════
# FULL-PERIOD LABEL (prepended to the shock dict for comparison)
# ════════════════════════════════════════════════════════════════════════════

ALL_PERIODS = {
    "Full sample": ("2000-01-01", "2024-12-31"),
    **SHOCK_PERIODS,
}


# ════════════════════════════════════════════════════════════════════════════
# MULTI-MODEL SUB-PERIOD FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_all_models_subperiod(r_series, periods=None):
    """
    Fit Gaussian, Student-t, VG, and NIG to each sub-period window.

    Returns a long-format pd.DataFrame indexed by (period, model) with
    columns for key parameters and information criteria.  Fitting failures
    are recorded as NaN rows rather than propagating exceptions.
    """
    if periods is None:
        periods = ALL_PERIODS

    records = []
    models = [
        ("Gaussian",  fit_gaussian,  ["mu", "sigma"]),
        ("Student-t", fit_student_t, ["nu", "mu", "sigma"]),
        ("VG",        fit_vg,        ["sigma", "theta", "nu", "mu"]),
        ("NIG",       fit_nig,       ["alpha", "beta", "delta", "mu"]),
    ]

    for period_name, (start, end) in periods.items():
        r_sub = r_series.loc[start:end].values
        n_obs = len(r_sub)

        if n_obs < 50:
            print(f"  Skipping {period_name}: only {n_obs} observations.")
            continue
        if n_obs < 100:
            warnings.warn(
                f"{period_name}: {n_obs} observations — VG/NIG SEs may be unreliable.",
                RuntimeWarning,
            )

        for model_name, fit_fn, param_keys in models:
            row = {"period": period_name, "model": model_name, "n_obs": n_obs}
            try:
                params = fit_fn(r_sub)
                row["loglik"]   = params["loglik"]
                row["aic"]      = params["aic"]
                row["bic"]      = params["bic"]
                row["n_params"] = params["n_params"]
                for k in param_keys:
                    row[k] = params.get(k, np.nan)
            except Exception as exc:
                warnings.warn(f"  {period_name} / {model_name}: {exc}", RuntimeWarning)
                row["loglik"] = np.nan
                row["aic"]    = np.nan
                for k in param_keys:
                    row[k] = np.nan
            records.append(row)

    df = pd.DataFrame(records).set_index(["period", "model"])
    return df


# ════════════════════════════════════════════════════════════════════════════
# EMPIRICAL SHOCK STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def empirical_shock_stats(r_series, periods=None, full_sigma=None):
    """
    Descriptive statistics for each window.

    Columns returned:
      n           number of daily observations
      mean_%      mean daily return as a percentage
      ann_vol_%   annualised volatility (σ × √252) as a percentage
      skewness    sample skewness
      ex_kurt     excess kurtosis (= 0 for Gaussian)
      var5_emp    empirical 5th percentile (negative = left-tail VaR)
      max_dd      largest single-day loss in the window
      pct_2sigma  proportion of observations with |r| > 2 × full-sample σ
    """
    if periods is None:
        periods = ALL_PERIODS
    if full_sigma is None:
        full_sigma = r_series.std()

    rows = []
    for period_name, (start, end) in periods.items():
        r_sub = r_series.loc[start:end].values
        n     = len(r_sub)
        if n < 10:
            continue
        rows.append({
            "Period":      period_name,
            "n":           n,
            "mean_%":      r_sub.mean() * 100,
            "ann_vol_%":   r_sub.std(ddof=0) * np.sqrt(252) * 100,
            "skewness":    float(scipy_stats.skew(r_sub)),
            "ex_kurt":     float(scipy_stats.kurtosis(r_sub)),
            "var5_emp":    float(np.percentile(r_sub, 5)) * 100,
            "max_dd":      float(r_sub.min()) * 100,
            "pct_2sigma":  float(np.mean(np.abs(r_sub) > 2 * full_sigma)) * 100,
        })
    return pd.DataFrame(rows).set_index("Period")


# ════════════════════════════════════════════════════════════════════════════
# AIC IMPROVEMENT TABLE
# ════════════════════════════════════════════════════════════════════════════

def aic_improvement_table(sub_df):
    """
    AIC(Gaussian) − AIC(model) for each period × model combination.

    Positive = the Lévy model is preferred over Gaussian.
    Large positive values in crisis windows confirm that fat-tail models
    add most value precisely when markets are most dislocated.
    """
    rows = []
    periods = sub_df.index.get_level_values("period").unique()
    for period in periods:
        aic_g = sub_df.loc[(period, "Gaussian"), "aic"]
        row   = {"Period": period}
        for model in ["Student-t", "VG", "NIG"]:
            try:
                aic_m = sub_df.loc[(period, model), "aic"]
                row[f"ΔAIC vs Gaussian ({model})"] = float(aic_g - aic_m)
            except KeyError:
                row[f"ΔAIC vs Gaussian ({model})"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Period")


# ════════════════════════════════════════════════════════════════════════════
# PRINT HELPER
# ════════════════════════════════════════════════════════════════════════════

def print_subperiod_summary(sub_df):
    """Print a compact readable summary of all fitted parameters per window."""
    prev_period = None
    for (period, model), row in sub_df.iterrows():
        if period != prev_period:
            print(f"\n{'─'*60}")
            print(f"  {period}   (n = {int(row.get('n_obs', 0))})")
            print(f"{'─'*60}")
            prev_period = period

        parts = [f"  [{model:<10}]"]

        # Print whichever parameters are present (varies by model)
        for key, fmt in [("nu",    "ν={:.3f}"), ("alpha", "α={:.1f}"),
                          ("mu",   "μ={:+.4%}/day"), ("sigma", "σ={:.4%}/day"),
                          ("theta","θ={:+.5f}"), ("beta",  "β={:+.2f}"),
                          ("delta","δ={:.5f}")]:
            val = row.get(key, np.nan)
            if pd.notna(val):
                try:
                    parts.append(fmt.format(float(val)))
                except Exception:
                    pass

        if pd.notna(row.get("aic")):
            parts.append(f"AIC={row['aic']:.1f}")
        print("  ".join(parts))


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


def make_subperiod_plots(sub_df, emp_stats, save_dir=None):
    """
    Produce two sub-period figures.

    week3_subperiod_params.png
        Grouped bar chart: annualised σ (VG) and ν (Student-t + VG) per window.
        Shock windows coloured by their SHOCK_COLOURS; full sample in grey.

    week3_subperiod_aic.png
        Horizontal bar chart of AIC improvement (Gaussian − model) per window
        for Student-t, VG, and NIG.  Quantifies where tail models add most value.
    """
    periods     = list(ALL_PERIODS.keys())
    period_cols = ["#cccccc"] + [SHOCK_COLOURS[p] for p in SHOCK_PERIODS]
    x           = np.arange(len(periods))
    w           = 0.20

    # ── 1. Parameter comparison ───────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: annualised daily σ across all four models
    ax = axes[0]
    for i, (model, col) in enumerate(
        [("Gaussian", MODEL_COLOURS["Gaussian"]),
         ("Student-t", MODEL_COLOURS["Student-t"]),
         ("VG", MODEL_COLOURS["VG"]),
         ("NIG", MODEL_COLOURS["NIG"])],
    ):
        sigma_vals = []
        for p in periods:
            try:
                key = "sigma" if model in ("Gaussian", "Student-t", "VG") else "delta"
                val = sub_df.loc[(p, model), key]
                # Annualise σ (σ × √252); for NIG δ is the scale, not daily σ directly
                if model == "NIG":
                    ann = float(val) * np.sqrt(252) if pd.notna(val) else np.nan
                else:
                    ann = float(val) * np.sqrt(252) if pd.notna(val) else np.nan
            except KeyError:
                ann = np.nan
            sigma_vals.append(ann)

        offset = (i - 1.5) * w
        ax.bar(x + offset, sigma_vals, w, label=model,
               color=col, alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Annualised scale (σ or δ) × √252")
    ax.set_title("Scale Parameter by Period — All Four Models")
    ax.legend(fontsize=8)

    # Panel B: tail heaviness — ν (Student-t) and ν (VG)
    ax2 = axes[1]
    nu_t  = []
    nu_vg = []
    for p in periods:
        try:
            nu_t.append(float(sub_df.loc[(p, "Student-t"), "nu"]))
        except KeyError:
            nu_t.append(np.nan)
        try:
            nu_vg.append(float(sub_df.loc[(p, "VG"), "nu"]))
        except KeyError:
            nu_vg.append(np.nan)

    ax2.bar(x - w / 2, nu_t,  w, label="Student-t ν",
            color=MODEL_COLOURS["Student-t"], alpha=0.8)
    ax2.bar(x + w / 2, nu_vg, w, label="VG ν",
            color=MODEL_COLOURS["VG"], alpha=0.8)

    ax2.axhline(2.0, color="red", lw=0.8, linestyle="--",
                label="ν = 2  (variance singularity)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Degrees of freedom / variance rate ν")
    ax2.set_title("Tail Parameter ν — Student-t and VG")
    ax2.legend(fontsize=8)

    fig1.suptitle("Sub-Period Parameter Comparison (S&P 500, 2000–2024)", fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig1, save_dir, "week3_subperiod_params.png")
    plt.close(fig1)

    # ── 2. AIC improvement ────────────────────────────────────────────────
    aic_df = aic_improvement_table(sub_df)
    models_aic = ["Student-t", "VG", "NIG"]
    fig2, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(periods))
    heights = len(models_aic)
    spacing = 0.25

    for i, model in enumerate(models_aic):
        col_name = f"ΔAIC vs Gaussian ({model})"
        vals     = [float(aic_df.loc[p, col_name]) if p in aic_df.index
                    and pd.notna(aic_df.loc[p, col_name]) else 0.0
                    for p in periods]
        offset   = (i - 1) * spacing
        ax.barh(y + offset, vals, spacing * 0.9,
                label=model, color=MODEL_COLOURS[model], alpha=0.8)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(periods, fontsize=9)
    ax.set_xlabel("ΔAIC = AIC(Gaussian) − AIC(model)   [positive = model preferred]")
    ax.set_title("AIC Improvement over Gaussian — Lévy Models per Period")
    ax.legend(fontsize=9)
    fig2.tight_layout()
    _save_fig(fig2, save_dir, "week3_subperiod_aic.png")
    plt.close(fig2)


# ════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_subperiod_analysis(r_series, save_dir=None):
    """Fit all four models to each window, print tables, save figures."""
    print("=" * 70)
    print("Week 3 — Sub-Period Analysis: All Four Models")
    print("=" * 70)

    print("\nFitting models to each sub-period (this may take several minutes)…")
    sub_df = fit_all_models_subperiod(r_series)

    print_subperiod_summary(sub_df)

    emp_stats = empirical_shock_stats(r_series)
    print("\n" + "─" * 70)
    print("Empirical shock statistics:")
    print("─" * 70)
    print(emp_stats.to_string(float_format="{:.3f}".format))

    aic_imp = aic_improvement_table(sub_df)
    print("\n" + "─" * 70)
    print("AIC improvement over Gaussian (positive = Lévy model preferred):")
    print("─" * 70)
    print(aic_imp.to_string(float_format="{:.1f}".format))

    make_subperiod_plots(sub_df, emp_stats, save_dir=save_dir)

    print("\nSub-period analysis complete.")
    return sub_df, emp_stats


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3: sub-period analysis")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save figures")
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir(__file__)
    r_series = fetch_returns()
    run_subperiod_analysis(r_series, save_dir=save_dir)
