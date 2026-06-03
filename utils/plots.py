"""
utils/plots.py
--------------
Shared plotting utilities used across all weekly scripts.
Includes density overlays, QQ plots, trace plots with shock shading,
and marginal distribution grids.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import pandas as pd
from typing import Optional

# Consistent colour palette across all figures
COLOURS = {
    "gaussian":  "steelblue",
    "student_t": "crimson",
    "vg":        "darkorange",
    "nig":       "mediumseagreen",
    "data":      "#cfe2f3",
    "vix":       "slategray",
}

# Shock period shading colours (light fills)
SHOCK_COLOURS = {
    "Dot-com crash":  "#fff3cd",
    "GFC":            "#f8d7da",
    "COVID-19":       "#d1ecf1",
    "Fed rate hikes": "#d4edda",
}


def shade_shock_periods(ax, returns_series: pd.Series, shock_periods: dict) -> None:
    """
    Shade four market shock periods on a time-series axis.

    Parameters
    ----------
    ax : matplotlib Axes
    returns_series : pd.Series with DatetimeIndex
    shock_periods : dict of {label: (start_str, end_str)}
    """
    for label, (start, end) in shock_periods.items():
        colour = SHOCK_COLOURS.get(label, "#eeeeee")
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.35, color=colour, label=label, zorder=0)


def plot_density_overlay(r: np.ndarray, fitted_models: dict,
                         title: str = "Return Distribution — Model Comparison",
                         save_path: Optional[str] = None) -> None:
    """
    Histogram of daily returns with fitted PDF curves overlaid.

    Parameters
    ----------
    r : np.ndarray of log-returns
    fitted_models : dict of {label: (pdf_callable, colour)}
        e.g. {"Gaussian": (gauss_pdf, "steelblue"), "Student-t": (t_pdf, "crimson")}
    """
    x = np.linspace(r.min(), r.max(), 800)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(r, bins=150, density=True,
            color=COLOURS["data"], edgecolor="none", label="S&P 500 log-returns")
    for label, (pdf_fn, colour) in fitted_models.items():
        ax.plot(x, pdf_fn(x), color=colour, lw=2, label=label)
    ax.set_xlim(-0.12, 0.12)
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=12)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_qq(r_std: np.ndarray, dist, label: str,
            colour: str = "steelblue",
            save_path: Optional[str] = None) -> None:
    """
    QQ plot of standardised returns against a theoretical distribution.
    Uses a (0, 1) reference line — not an OLS fit — for correct visual calibration.

    Parameters
    ----------
    r_std : standardised return array (r - mu) / sigma
    dist  : scipy.stats frozen distribution for theoretical quantiles
    """
    n = len(r_std)
    probs = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    theoretical = dist.ppf(probs)
    sample = np.sort(r_std)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, sample, s=2, alpha=0.5, color=colour)
    lim = max(abs(theoretical).max(), abs(sample).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)   # (0,1) line, not OLS
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(f"QQ Plot — {label}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trace(returns_series: pd.Series, shock_periods: dict,
               title: str = "S&P 500 Daily Log-Returns (2000–2024)",
               save_path: Optional[str] = None) -> None:
    """
    Time-series trace plot of log-returns with shock periods shaded.
    Central figure for the GitHub Pages showcase.

    Parameters
    ----------
    returns_series : pd.Series with DatetimeIndex
    shock_periods  : dict from data.fetch.SHOCK_PERIODS
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    shade_shock_periods(ax, returns_series, shock_periods)
    ax.plot(returns_series.index, returns_series.values,
            lw=0.6, color="black", alpha=0.7, zorder=1)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily log-return")
    ax.set_title(title, fontsize=12)
    # Shock legend
    handles = [mpatches.Patch(color=SHOCK_COLOURS[k], alpha=0.6, label=k)
               for k in shock_periods]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_marginals_by_year(returns_series: pd.Series,
                           save_path: Optional[str] = None) -> None:
    """
    Grid of annual kernel density plots — one panel per calendar year.
    Illustrates how the marginal return distribution shifts year-to-year.
    """
    years = sorted(returns_series.index.year.unique())
    ncols = 5
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.8))
    axes = axes.flatten()
    x = np.linspace(-0.12, 0.12, 400)
    for i, year in enumerate(years):
        r_yr = returns_series[returns_series.index.year == year].values
        if len(r_yr) < 20:
            continue
        kde = stats.gaussian_kde(r_yr)
        axes[i].fill_between(x, kde(x), alpha=0.4, color="steelblue")
        axes[i].plot(x, kde(x), color="steelblue", lw=1)
        axes[i].set_title(str(year), fontsize=9)
        axes[i].set_xlim(-0.12, 0.12)
        axes[i].tick_params(labelsize=7)
        axes[i].set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("S&P 500 Annual Return Distributions (2000–2024)", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
