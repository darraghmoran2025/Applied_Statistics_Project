"""
scripts/week2_run.py
--------------------
Week 2 runner: Gaussian and Student-t MLE benchmarks.

Usage
-----
    python scripts/week2_run.py                          # default: full 2000-2024
    python scripts/week2_run.py --start 2007-01-01       # custom date range
    python scripts/week2_run.py --no-plots               # suppress figures
    python scripts/week2_run.py --save-plots             # save to docs/figures/
"""

import argparse
import numpy as np
from scipy import stats

from data.fetch import fetch_returns, fetch_returns_with_dates, SHOCK_PERIODS
from models import gaussian, student_t
from risk.measures import var_es_gaussian, var_es_student_t, risk_table
from utils.plots import (plot_density_overlay, plot_qq,
                         plot_trace, plot_marginals_by_year)


def parse_args():
    p = argparse.ArgumentParser(description="Week 2: Gaussian & Student-t MLE")
    p.add_argument("--start",      default="2000-01-01")
    p.add_argument("--end",        default="2024-12-31")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = "docs/figures/" if args.save_plots else None

    # ── 1. Data ──────────────────────────────────────────────────────────────
    r        = fetch_returns(start=args.start, end=args.end)
    r_series = fetch_returns_with_dates(start=args.start, end=args.end)
    print(f"\nData: {len(r):,} daily log-returns  (S&P 500, {args.start}–{args.end})\n")

    # ── 2. Fit models ─────────────────────────────────────────────────────────
    g = gaussian.fit(r)
    print("Gaussian MLE")
    print(f"  μ     = {g['mu']:+.6f}   SE = {g['se_mu']:.6f}")
    print(f"  σ     = {g['sigma']:.6f}    SE = {g['se_sigma']:.6f}")
    print(f"  Log-lik = {g['loglik']:.2f}   AIC = {g['aic']:.2f}   BIC = {g['bic']:.2f}")

    print("\nFitting Student-t (numerical optimisation)…")
    t = student_t.fit(r)
    print("Student-t MLE")
    print(f"  ν     = {t['nu']:.4f}      SE = {t['se_nu']:.4f}")
    print(f"  μ     = {t['mu']:+.6f}   SE = {t['se_mu']:.6f}")
    print(f"  σ     = {t['sigma']:.6f}    SE = {t['se_sigma']:.6f}")
    print(f"  Log-lik = {t['loglik']:.2f}   AIC = {t['aic']:.2f}   BIC = {t['bic']:.2f}")

    # ── 3. Risk measures ──────────────────────────────────────────────────────
    models = {
        "Gaussian":  (var_es_gaussian,  g),
        "Student-t": (var_es_student_t, t),
    }
    risk = risk_table(models)
    print("\nRisk Measures (daily log-returns; negative = loss):")
    print(risk.to_string(float_format="{:.6f}".format))

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        # Density overlay
        plot_density_overlay(
            r,
            fitted_models={
                "Gaussian MLE":               (lambda x: stats.norm.pdf(x, g["mu"], g["sigma"]),
                                               "steelblue"),
                f"Student-t MLE (ν={t['nu']:.2f})": (lambda x: stats.t.pdf(x, t["nu"], t["mu"], t["sigma"]),
                                                       "crimson"),
            },
            title="S&P 500 Daily Log-Returns — Gaussian vs Student-t MLE (2000–2024)",
            save_path=f"{save_dir}week2_density.png" if save_dir else None,
        )

        # QQ — Gaussian (corrected (0,1) reference line)
        plot_qq(
            (r - g["mu"]) / g["sigma"],
            dist=stats.norm,
            label="Gaussian",
            colour="steelblue",
            save_path=f"{save_dir}week2_qq_gaussian.png" if save_dir else None,
        )

        # QQ — Student-t
        plot_qq(
            (r - t["mu"]) / t["sigma"],
            dist=stats.t(df=t["nu"]),
            label=f"Student-t (ν={t['nu']:.2f})\n"
                  "Note: large axis range reflects ν < 3 (heavy-tailed standardisation)",
            colour="crimson",
            save_path=f"{save_dir}week2_qq_student_t.png" if save_dir else None,
        )

        # Trace plot with shock shading
        plot_trace(
            r_series, SHOCK_PERIODS,
            save_path=f"{save_dir}week2_trace.png" if save_dir else None,
        )

        # Annual marginal distributions
        plot_marginals_by_year(
            r_series,
            save_path=f"{save_dir}week2_marginals_by_year.png" if save_dir else None,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
