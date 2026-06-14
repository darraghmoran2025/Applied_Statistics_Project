"""
week2_marginals_enhanced.py
5 separate figures (one per 5-year block), each with distribution panels
on top and a full MLE parameter table beneath.

Each figure covers 5 years. Top section: 5 distribution panels with
KDE, fitted Normal, fitted Student-t, x=0 reference, and year mean.
Bottom section: table of Gaussian and Student-t MLE results for those years.

Saved as:
  week2_marginals_2000_2004.png
  week2_marginals_2005_2009.png
  week2_marginals_2010_2014.png
  week2_marginals_2015_2019.png
  week2_marginals_2020_2024.png

Run: python week2_marginals_enhanced.py
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from week2_gaussian_student_mle import (
    fetch_returns,
    fit_gaussian,
    fit_student_t,
    SHOCK_PERIODS,
    SHOCK_COLOURS,
)

FULLSAMPLE_NU = 2.648


# ── MLE fitting ───────────────────────────────────────────────────────────────

def _fit_year(r_yr):
    """
    Fit Gaussian and Student-t to one year of returns.
    Returns (g_dict, t_dict) with full MLE results including log-lik and AIC.
    Student-t falls back to full-sample nu if n < 100 or optimisation fails.
    """
    r_yr = np.asarray(r_yr)
    g = fit_gaussian(r_yr)

    try:
        if len(r_yr) < 100:
            raise ValueError("too few obs")
        t = fit_student_t(r_yr)
        if not (2.01 < t["nu"] < 300) or not np.isfinite(t["nu"]):
            raise ValueError("nu out of range")
    except Exception:
        # Use full-sample nu; compute log-lik analytically at those params
        nu0  = FULLSAMPLE_NU
        mu0  = float(r_yr.mean())
        sig0 = float(r_yr.std(ddof=0))
        ll   = float(np.sum(stats.t.logpdf(r_yr, df=nu0, loc=mu0, scale=sig0)))
        t = {
            "nu": nu0, "mu": mu0, "sigma": sig0,
            "loglik": ll,
            "aic":    2 * 3 - 2 * ll,
            "bic":    3 * np.log(len(r_yr)) - 2 * ll,
        }

    return g, t


# ── shock map ─────────────────────────────────────────────────────────────────

def _shock_map():
    m = {}
    for label, (start, end) in SHOCK_PERIODS.items():
        for yr in range(int(start[:4]), int(end[:4]) + 1):
            m.setdefault(yr, label)
    return m


# ── single 5-year figure ──────────────────────────────────────────────────────

def _make_figure(r_series, years_block, shock_map, save_dir):
    period_label = f"{years_block[0]}-{years_block[-1]}"
    print(f"  Building figure {period_label}...")

    # ── fit all five years ────────────────────────────────────────────────────
    year_results = {}
    for year in years_block:
        r_yr = r_series[r_series.index.year == year].values
        g, t = _fit_year(r_yr)
        year_results[year] = (r_yr, g, t)
        print(f"    {year}: sigma_ann={g['sigma']*np.sqrt(252):.1%}  nu={t['nu']:.2f}")

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, dist_axes = plt.subplots(
        1, 5,
        figsize=(24, 6),
    )
    fig.subplots_adjust(
        left=0.04, right=0.99,
        top=0.88, bottom=0.18,
        wspace=0.25,
    )

    LINE_TOP = 0.66   # axvline ymax — stops lines below the annotation box

    # ── distribution panels ───────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#a0c4e8", alpha=0.7, label="Empirical KDE"),
        plt.Line2D([0], [0], color="#1a3a8f", lw=2.2, ls="--",
                   label="Fitted Normal"),
        plt.Line2D([0], [0], color="crimson", lw=2.2, ls="-",
                   label="Fitted Student-t"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=1.2, ls="--",
                   label="x = 0"),
        plt.Line2D([0], [0], color="#222222", lw=1.4, ls=":",
                   label="Year mean"),
    ]
    shock_patches = [
        mpatches.Patch(color=SHOCK_COLOURS[lbl], alpha=0.75, label=lbl)
        for lbl in SHOCK_PERIODS
    ]

    for col, year in enumerate(years_block):
        ax = dist_axes[col]
        r_yr, g, t = year_results[year]

        shock_label = shock_map.get(year)
        ax.set_facecolor(SHOCK_COLOURS[shock_label] if shock_label else "#f5f5f5")

        half = max(4.5 * r_yr.std(), 0.08)
        x    = np.linspace(-half, half, 700)

        # KDE
        kde = stats.gaussian_kde(r_yr)
        ax.fill_between(x, kde(x), alpha=0.22, color="#5090c8", zorder=1)
        ax.plot(x, kde(x), color="#5090c8", lw=0.9, alpha=0.6, zorder=2)

        g_mu  = float(g["mu"])
        g_sig = float(g["sigma"])
        nu    = float(t["nu"])
        t_mu  = float(t["mu"])
        t_sig = float(t["sigma"])

        # Fitted Normal
        ax.plot(x, stats.norm.pdf(x, g_mu, g_sig),
                color="#1a3a8f", lw=2.2, ls="--", zorder=4)

        # Fitted Student-t
        ax.plot(x, stats.t.pdf(x, df=nu, loc=t_mu, scale=t_sig),
                color="crimson", lw=2.2, ls="-", zorder=5)

        # Reference lines (clipped below annotation headroom)
        ax.axvline(0.0,  ymin=0, ymax=LINE_TOP,
                   color="#aaaaaa", lw=1.2, ls="--", zorder=3)
        ax.axvline(g_mu, ymin=0, ymax=LINE_TOP,
                   color="#222222", lw=1.4, ls=":",  zorder=6)

        # y-axis headroom so annotation box never overlaps the curves
        y_peak = max(
            kde(x).max(),
            stats.norm.pdf(x, g_mu, g_sig).max(),
            stats.t.pdf(x, df=nu, loc=t_mu, scale=t_sig).max(),
        )
        ax.set_ylim(0, y_peak * 1.45)

        # Year title above panel
        title_color = "#3a1800" if shock_label else "black"
        ax.set_title(str(year), fontsize=16, fontweight="bold",
                     color=title_color, pad=8)

        # Compact annotation box — just sigma and nu (full table is below)
        ann_vol = g_sig * np.sqrt(252)
        info = f"σ = {ann_vol:.1%}\nν = {nu:.1f}"
        ax.text(0.04, 0.97, info,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="#111111", linespacing=1.6,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", alpha=0.82, ec="#cccccc", lw=0.5))

        ax.set_xlim(-half, half)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yticks([])
        ax.set_xlabel("Daily log-return", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)

    # ── figure legend and title ───────────────────────────────────────────────
    fig.legend(
        handles=legend_elements + shock_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=9,
        framealpha=0.95,
    )
    fig.suptitle(
        f"S&P 500 Annual Return Distributions and MLE Results  {period_label}",
        fontsize=14, y=0.975,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"week2_marginals_{years_block[0]}_{years_block[-1]}.png"
        path  = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {path}")

    plt.close(fig)


# ── main function ─────────────────────────────────────────────────────────────

def plot_marginals_by_period(r_series, save_dir=None):
    """Generate 5 figures (one per 5-year block, 2000-2024)."""
    shock_map = _shock_map()
    all_years = list(range(2000, 2025))
    blocks    = [all_years[i:i+5] for i in range(0, 25, 5)]

    print("Generating 5 figures with MLE tables...\n")
    for block in blocks:
        _make_figure(r_series, block, shock_map, save_dir)

    print("\nAll five figures complete.")


plot_marginals_enhanced = plot_marginals_by_period


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r_series = fetch_returns()
    print(f"Data: {len(r_series):,} daily log-returns (S&P 500, 2000-2024)\n")
    plot_marginals_by_period(r_series, save_dir=HERE)
