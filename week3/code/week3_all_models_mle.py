"""
Run standalone:
    python week3_all_models_mle.py
    python week3_all_models_mle.py --save_dir ../figures --n_sim 500000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
run_all_models_mle(r_series, n_sim, save_dir) → dict
    Master function.  Fits all five models, prints results, saves figures.
    Returns a dict with keys "gaussian", "laplace", "student_t", "vg", "nig".

model_comparison_table(g, lap, t, vg, nig)  → pd.DataFrame
    Unified AIC/BIC/log-likelihood comparison.

risk_table_all_models(g, lap, t, vg, nig, n_sim)  → pd.DataFrame
    VaR and ES at 95%/99% for all five models side by side.

goodness_of_fit_all(r, g, lap, t, vg, nig, n_sim)  → pd.DataFrame
    KS statistics and two-sample tests for all five models.

make_plots_week3_mle(r_series, g, lap, t, vg, nig, n_sim, save_dir)
    Produces:
      week3_density_all_models.png  — histogram + 4 PDFs
      week3_qq_all_models.png       — 2×2 QQ plot grid
      week3_risk_comparison.png     — VaR/ES bar chart

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE NAMES
──────────────
g, t, vg, nig   Dicts of fitted parameters from fit_gaussian(),
                fit_student_t(), fit_vg(), fit_nig() respectively.
samp_vg         np.ndarray of Monte Carlo draws from fitted VG model.
samp_nig        np.ndarray of Monte Carlo draws from fitted NIG model.
n_sim           Number of Monte Carlo draws for VaR/ES and QQ quantiles.
x_grid          np.linspace over return range for density evaluation.
probs           Probability grid (0.5/n … 1−0.5/n) for QQ construction.
th_q_*          Theoretical quantiles derived from model (analytical or
                from simulation for VG/NIG).
emp_q           Sorted empirical log-returns for QQ plot.
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
from scipy import stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Imports from sibling modules ─────────────────────────────────────────────
def _find_week2():
    """Locate week2_gaussian_student_mle.py in either repo layout."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "week2", "code"),  # repo_clone layout
        os.path.join(here, "..", "Week2"),                 # top-level layout
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
    fetch_returns, fit_gaussian, fit_student_t, compute_risk_measures,
    goodness_of_fit as gof_week2,
)
from levy_models import (
    fit_vg, fit_nig, fit_laplace, laplace_var_es,
    simulate_vg, simulate_nig, var_es_mc,
    _logpdf_vg, _logpdf_nig,
    SHOCK_PERIODS, SHOCK_COLOURS, MODEL_COLOURS,
    default_save_dir,
)


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLES
# ════════════════════════════════════════════════════════════════════════════

def model_comparison_table(g, lap, t, vg, nig):
    """
    Unified AIC/BIC/log-likelihood table for all five fitted models.

    Lower AIC/BIC indicates a better-fitting model penalised for complexity.
    The Laplace, Student-t, VG, and NIG each relax the Gaussian's thin-tail
    assumption; the AIC/BIC column deltas quantify whether the improvement
    in fit justifies the parameter count.  The Laplace is notable as a
    two-parameter model (like the Gaussian) that already captures
    exponential tails, and as the symmetric VG special case (θ=0, ν=1).
    """
    rows = []
    for name, params in [("Gaussian", g), ("Laplace", lap), ("Student-t", t),
                          ("Variance-Gamma", vg), ("NIG", nig)]:
        rows.append({
            "Model":     name,
            "k":         params["n_params"],
            "Log-lik":   params["loglik"],
            "AIC":       params["aic"],
            "BIC":       params["bic"],
            "ΔAIC (vs Gaussian)": params["aic"] - g["aic"],
            "ΔBIC (vs Gaussian)": params["bic"] - g["bic"],
        })
    df = pd.DataFrame(rows).set_index("Model")
    df["AIC rank"] = df["AIC"].rank().astype(int)
    return df


def risk_table_all_models(g, lap, t, vg, nig, n_sim=500_000):
    """
    VaR and ES at 95% and 99% for all five models.

    Gaussian, Laplace and Student-t use closed-form expressions; VG and NIG
    use Monte Carlo with n_sim draws.  The Laplace VaR/ES come from
    laplace_var_es() (exponential lower-tail identities).
    """
    # Closed-form risk measures for Gaussian and Student-t (Week 2 convention)
    risk_gt  = compute_risk_measures(g, t, alphas=(0.95, 0.99))
    rm_lap   = laplace_var_es(lap, alphas=(0.95, 0.99))

    samp_vg  = simulate_vg(vg,  n=n_sim, seed=42)
    samp_nig = simulate_nig(nig, n=n_sim, seed=42)

    rm_vg  = var_es_mc(samp_vg,  alphas=(0.95, 0.99))
    rm_nig = var_es_mc(samp_nig, alphas=(0.95, 0.99))

    rows = []
    for label, a in [("95%", 0.95), ("99%", 0.99)]:
        rows.append({
            "Confidence":    label,
            "VaR (Gaussian)":    risk_gt.loc[label, "VaR (Gaussian)"],
            "ES (Gaussian)":     risk_gt.loc[label, "ES (Gaussian)"],
            "VaR (Laplace)":     rm_lap[a]["VaR"],
            "ES (Laplace)":      rm_lap[a]["ES"],
            "VaR (Student-t)":   risk_gt.loc[label, "VaR (Student-t)"],
            "ES (Student-t)":    risk_gt.loc[label, "ES (Student-t)"],
            "VaR (VG)":          rm_vg[a]["VaR"],
            "ES (VG)":           rm_vg[a]["ES"],
            "VaR (NIG)":         rm_nig[a]["VaR"],
            "ES (NIG)":          rm_nig[a]["ES"],
        })
    return pd.DataFrame(rows).set_index("Confidence")


# ════════════════════════════════════════════════════════════════════════════
# GOODNESS-OF-FIT
# ════════════════════════════════════════════════════════════════════════════

_KS_NOTE = (
    "Gaussian and Student-t: one-sample KS test on standardised residuals.\n"
    "VG and NIG: two-sample KS test against a simulated draw of 1,000,000 "
    "from the fitted model.  With n ≈ 6,300, p-values are near-zero for all "
    "models; the KS statistic D is the meaningful comparison metric."
)


def goodness_of_fit_all(r, g, lap, t, vg, nig, n_sim=1_000_000):
    """
    KS goodness-of-fit tests for all five models.

    One-sample KS (analytical CDF) for Gaussian, Laplace and Student-t.
    Two-sample KS (vs 1M Monte Carlo draws) for VG and NIG, which lack
    closed-form CDFs.  The KS statistic D measures the maximum deviation
    between the empirical CDF and the fitted model CDF.
    """
    r = np.asarray(r)

    # Gaussian and Student-t: one-sample KS on standardised residuals
    gof_gt = gof_week2(r, g, t)
    ks_g = float(gof_gt.loc["Gaussian",  "KS Statistic"])
    ks_t = float(gof_gt.loc["Student-t", "KS Statistic"])
    pv_g = float(gof_gt.loc["Gaussian",  "p-value*"])
    pv_t = float(gof_gt.loc["Student-t", "p-value*"])

    # Laplace: one-sample KS against the analytical Laplace CDF
    ks_lap = stats.kstest(r, "laplace", args=(lap["mu"], lap["b"]))

    # VG and NIG: two-sample KS against large simulated sample
    samp_vg  = simulate_vg(vg,  n=n_sim, seed=99)
    samp_nig = simulate_nig(nig, n=n_sim, seed=99)

    ks_vg  = stats.ks_2samp(r, samp_vg)
    ks_nig = stats.ks_2samp(r, samp_nig)

    df = pd.DataFrame([
        {"Model": "Gaussian",  "KS Statistic": ks_g,              "p-value": pv_g,              "n_obs": len(r)},
        {"Model": "Laplace",   "KS Statistic": ks_lap.statistic,  "p-value": ks_lap.pvalue,     "n_obs": len(r)},
        {"Model": "Student-t", "KS Statistic": ks_t,              "p-value": pv_t,              "n_obs": len(r)},
        {"Model": "VG",        "KS Statistic": ks_vg.statistic,   "p-value": ks_vg.pvalue,      "n_obs": len(r)},
        {"Model": "NIG",       "KS Statistic": ks_nig.statistic,  "p-value": ks_nig.pvalue,     "n_obs": len(r)},
    ]).set_index("Model")
    df.attrs["note"] = _KS_NOTE
    return df


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


def make_plots_week3_mle(r_series, g, lap, t, vg, nig,
                          n_sim=500_000, save_dir=None):
    """
    Produce three figures for the Week 3 five-model MLE section.

    week3_density_all_models.png
        Histogram of empirical returns with all five model PDFs overlaid.

    week3_qq_all_models.png
        2×3 QQ plot grid (five panels, sixth blank).  Gaussian, Laplace and
        Student-t use analytical quantile functions; VG and NIG use quantiles
        interpolated from n_sim simulated draws, which approximates the
        theoretical CDF to within 1/√n_sim.

    week3_risk_comparison.png
        Grouped bar chart of VaR and ES at 95% and 99% for all five models.
    """
    r    = r_series.values
    n    = len(r)
    x_gr = np.linspace(r.min() * 1.05, r.max() * 1.05, 800)

    print("\nSimulating VG and NIG samples for plots…")
    samp_vg  = simulate_vg(vg,  n=n_sim, seed=1)
    samp_nig = simulate_nig(nig, n=n_sim, seed=1)

    # ── 1. Density overlay ────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(12, 5))
    ax.hist(r, bins=150, density=True,
            color="#d8e8f5", edgecolor="none", label="S&P 500 log-returns", zorder=0)

    ax.plot(x_gr,
            stats.norm.pdf(x_gr, g["mu"], g["sigma"]),
            color=MODEL_COLOURS["Gaussian"], lw=2, label="Gaussian MLE", zorder=2)

    ax.plot(x_gr,
            stats.laplace.pdf(x_gr, lap["mu"], lap["b"]),
            color=MODEL_COLOURS["Laplace"], lw=2, ls="--",
            label="Laplace MLE  (VG: θ=0, ν=1)", zorder=2)

    ax.plot(x_gr,
            stats.t.pdf(x_gr, t["nu"], t["mu"], t["sigma"]),
            color=MODEL_COLOURS["Student-t"], lw=2,
            label=f"Student-t MLE  (ν = {t['nu']:.2f})", zorder=2)

    ax.plot(x_gr,
            np.exp(_logpdf_vg(x_gr, vg["mu"], vg["sigma"], vg["theta"], vg["nu"])),
            color=MODEL_COLOURS["VG"], lw=2,
            label=f"VG MLE  (ν = {vg['nu']:.3f}, θ = {vg['theta']:+.4f})", zorder=2)

    ax.plot(x_gr,
            np.exp(_logpdf_nig(x_gr, nig["mu"], nig["alpha"], nig["beta"], nig["delta"])),
            color=MODEL_COLOURS["NIG"], lw=2,
            label=f"NIG MLE  (α = {nig['alpha']:.1f}, β = {nig['beta']:+.1f})", zorder=2)

    ax.set_xlim(-0.12, 0.12)
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution — All Five Models vs Empirical (2000–2024)")
    ax.legend(fontsize=9)
    fig1.tight_layout()
    _save_fig(fig1, save_dir, "week3_density_all_models.png")
    plt.close(fig1)

    # ── 2. QQ plots (2×2 grid) ────────────────────────────────────────────
    probs  = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    emp_q  = np.sort(r)

    # Analytical theoretical quantiles
    th_g   = stats.norm.ppf(probs, g["mu"], g["sigma"])
    th_lap = stats.laplace.ppf(probs, lap["mu"], lap["b"])
    th_t   = stats.t.ppf(probs, t["nu"], t["mu"], t["sigma"])
    # Simulation-based theoretical quantiles
    th_vg  = np.quantile(samp_vg,  probs)
    th_nig = np.quantile(samp_nig, probs)

    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    configs = [
        (axes[0, 0], th_g,   emp_q, "Gaussian",  MODEL_COLOURS["Gaussian"]),
        (axes[0, 1], th_lap, emp_q, "Laplace",   MODEL_COLOURS["Laplace"]),
        (axes[0, 2], th_t,   emp_q, "Student-t", MODEL_COLOURS["Student-t"]),
        (axes[1, 0], th_vg,  emp_q, "VG",        MODEL_COLOURS["VG"]),
        (axes[1, 1], th_nig, emp_q, "NIG",       MODEL_COLOURS["NIG"]),
    ]
    for ax, th_q, em_q, name, color in configs:
        lim = max(np.abs(th_q).max(), np.abs(em_q).max()) * 1.05
        ax.scatter(th_q, em_q, s=1.5, alpha=0.35, color=color)
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Theoretical quantiles", fontsize=9)
        ax.set_ylabel("Sample quantiles", fontsize=9)
        ax.set_title(f"QQ Plot — {name}", fontsize=10)
    axes[1, 2].axis("off")   # sixth cell unused (five models)

    fig2.suptitle("QQ Plots — All Five Models (S&P 500, 2000–2024)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig2, save_dir, "week3_qq_all_models.png")
    plt.close(fig2)

    # ── 3. VaR/ES bar chart ───────────────────────────────────────────────
    rm_vg  = var_es_mc(samp_vg,  alphas=(0.95, 0.99))
    rm_nig = var_es_mc(samp_nig, alphas=(0.95, 0.99))

    labels  = ["Gaussian", "Laplace", "Student-t", "VG", "NIG"]
    colours = [MODEL_COLOURS[m] for m in labels]

    risk_gt = compute_risk_measures(g, t, alphas=(0.95, 0.99))
    rm_lap  = laplace_var_es(lap, alphas=(0.95, 0.99))

    fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_i, (conf_label, a) in enumerate([("95%", 0.95), ("99%", 0.99)]):
        ax = axes[ax_i]
        var_vals = [
            risk_gt.loc[conf_label, "VaR (Gaussian)"],
            rm_lap[a]["VaR"],
            risk_gt.loc[conf_label, "VaR (Student-t)"],
            rm_vg[a]["VaR"],
            rm_nig[a]["VaR"],
        ]
        es_vals = [
            risk_gt.loc[conf_label, "ES (Gaussian)"],
            rm_lap[a]["ES"],
            risk_gt.loc[conf_label, "ES (Student-t)"],
            rm_vg[a]["ES"],
            rm_nig[a]["ES"],
        ]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, [abs(v) for v in var_vals], w,
               color=colours, alpha=0.75, label="VaR")
        ax.bar(x + w / 2, [abs(v) for v in es_vals],  w,
               color=colours, alpha=0.45, hatch="//", label="ES")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("|Daily loss|")
        ax.set_title(f"VaR and ES — {conf_label} Confidence")
        ax.legend(fontsize=8)

    model_patches = [mpatches.Patch(color=MODEL_COLOURS[m], label=m) for m in labels]
    fig3.legend(handles=model_patches, loc="lower center",
                ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig3.suptitle("Risk Measures — All Five Models (S&P 500, 2000–2024)", fontsize=12)
    fig3.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save_fig(fig3, save_dir, "week3_risk_comparison.png")
    plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_all_models_mle(r_series, n_sim=500_000, save_dir=None):
    """
    Fit all four models, print tables, save figures.
    Returns dict keyed "gaussian", "student_t", "vg", "nig".
    """
    r = r_series.values

    print("=" * 70)
    print("Week 3 — Five-Model MLE Comparison")
    print("=" * 70)
    print(f"\nData: {len(r):,} daily log-returns  (S&P 500, 2000–2024)\n")

    # ── Gaussian ─────────────────────────────────────────────────────────
    g = fit_gaussian(r)
    print("Gaussian MLE")
    print(f"  μ = {g['mu']:+.6f}  SE {g['se_mu']:.6f}")
    print(f"  σ = {g['sigma']:.6f}   SE {g['se_sigma']:.6f}")
    print(f"  Log-lik = {g['loglik']:.2f}   AIC = {g['aic']:.2f}   BIC = {g['bic']:.2f}")

    # ── Laplace (symmetric VG special case: θ=0, ν=1) ─────────────────────
    lap = fit_laplace(r)
    print("\nLaplace MLE  (double-exponential; VG with θ=0, ν=1)")
    print(f"  μ = {lap['mu']:+.6f}  SE {lap['se_mu']:.6f}")
    print(f"  b = {lap['b']:.6f}   SE {lap['se_b']:.6f}")
    print(f"  Log-lik = {lap['loglik']:.2f}   AIC = {lap['aic']:.2f}   BIC = {lap['bic']:.2f}")

    # ── Student-t ─────────────────────────────────────────────────────────
    print("\nFitting Student-t…")
    t = fit_student_t(r)
    print("Student-t MLE")
    print(f"  ν = {t['nu']:.4f}   SE {t['se_nu']:.4f}")
    print(f"  μ = {t['mu']:+.6f}  SE {t['se_mu']:.6f}")
    print(f"  σ = {t['sigma']:.6f}   SE {t['se_sigma']:.6f}")
    print(f"  Log-lik = {t['loglik']:.2f}   AIC = {t['aic']:.2f}   BIC = {t['bic']:.2f}")

    # ── VG ────────────────────────────────────────────────────────────────
    print("\nFitting Variance-Gamma (multiple starting points)…")
    vg = fit_vg(r)
    print("VG MLE")
    print(f"  σ = {vg['sigma']:.6f}   SE {vg['se_sigma']:.6f}")
    print(f"  θ = {vg['theta']:+.6f}   SE {vg['se_theta']:.6f}")
    print(f"  ν = {vg['nu']:.6f}   SE {vg['se_nu']:.6f}")
    print(f"  μ = {vg['mu']:+.6f}   SE {vg['se_mu']:.6f}")
    print(f"  Log-lik = {vg['loglik']:.2f}   AIC = {vg['aic']:.2f}   BIC = {vg['bic']:.2f}")

    # ── NIG ───────────────────────────────────────────────────────────────
    print("\nFitting Normal Inverse Gaussian (multiple starting points)…")
    nig = fit_nig(r)
    print("NIG MLE")
    print(f"  α = {nig['alpha']:.4f}   SE {nig['se_alpha']:.4f}")
    print(f"  β = {nig['beta']:+.4f}   SE {nig['se_beta']:.4f}")
    print(f"  δ = {nig['delta']:.6f}   SE {nig['se_delta']:.6f}")
    print(f"  μ = {nig['mu']:+.6f}   SE {nig['se_mu']:.6f}")
    print(f"  Log-lik = {nig['loglik']:.2f}   AIC = {nig['aic']:.2f}   BIC = {nig['bic']:.2f}")

    # ── Comparison table ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Model Comparison (AIC/BIC — lower is better):")
    print("─" * 70)
    comp = model_comparison_table(g, lap, t, vg, nig)
    print(comp.to_string(float_format="{:.2f}".format))

    # ── Risk measures ─────────────────────────────────────────────────────
    print(f"\nComputing risk measures (Monte Carlo n = {n_sim:,})…")
    risk = risk_table_all_models(g, lap, t, vg, nig, n_sim=n_sim)
    print("\nVaR and ES — all four models (daily log-returns; negative = loss):")
    print(risk.to_string(float_format="{:.6f}".format))

    # ── Goodness-of-fit ───────────────────────────────────────────────────
    print("\nGoodness-of-fit — KS tests:")
    gof = goodness_of_fit_all(r, g, lap, t, vg, nig, n_sim=1_000_000)
    print(gof.to_string(float_format="{:.5f}".format))
    print(f"\nNote: {gof.attrs['note']}")

    # ── Plots ─────────────────────────────────────────────────────────────
    make_plots_week3_mle(r_series, g, lap, t, vg, nig,
                          n_sim=n_sim, save_dir=save_dir)

    print("\nFive-model MLE complete.")
    return {"gaussian": g, "laplace": lap, "student_t": t, "vg": vg, "nig": nig}


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3: four-model MLE comparison")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save figures (default: script directory)")
    parser.add_argument("--n_sim", type=int, default=500_000,
                        help="Monte Carlo draws for VaR/ES and QQ plots (default: 500000)")
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir(__file__)

    r_series = fetch_returns()
    run_all_models_mle(r_series, n_sim=args.n_sim, save_dir=save_dir)
