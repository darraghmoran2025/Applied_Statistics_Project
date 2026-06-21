"""
week4_yearly_levy_mle.py — Year-by-year VG and NIG MLE (zero-mean)
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

PURPOSE
───────
A companion to the Week 4 Bayesian work.  Weeks 2-3 fitted the Variance-Gamma
and Normal-Inverse-Gaussian densities once, on the full 2000-2024 sample.  This
script refits both, by maximum likelihood, ONE CALENDAR YEAR AT A TIME, so the
parameters can be read as a time series and the figures broken into sections
for easier year-by-year analysis (Neil, 18/06).

Two deliberate choices follow that brief:

  • μ → 0.  The location is fixed at zero in every fit.  Daily drift is tiny
    and barely identified within a single year (~250 observations); pinning it
    removes a noisy nuisance parameter and lets the scale, tail and asymmetry
    parameters carry the year's shape cleanly.  Any small mean is absorbed by
    the asymmetry parameter (θ for VG, β for NIG), which is where it belongs.

  • Annualised scale.  The daily scale parameters σ (VG) and δ (NIG) are
    reported annualised (× √252), matching how volatility is quoted elsewhere
    in the project, so a year's scale is directly comparable to the VIX and to
    the Gaussian σ.

The densities, the (α, ξ=β/α) reparametrisation that enforces α > |β|, and the
numerical-Hessian standard errors are all reused from week3/code/levy_models.py,
so these zero-mean yearly fits are numerically consistent with the full-sample
fits reported earlier.

Run standalone:
    python week4_yearly_levy_mle.py
    python week4_yearly_levy_mle.py --save_dir ../figures --start 2000 --end 2024

OUTPUTS
───────
    week4_yearly_vg_params.png    VG σ(ann), θ, ν by year (forest-green)
    week4_yearly_nig_params.png   NIG α, β, δ(ann) by year (dark-orange)
    week4_marginals_<block>.png   per-year empirical KDE + fitted VG/NIG (μ=0),
                                  one figure per 5-year block (Week-2 format)
    week4/data/week4_yearly_levy_params.csv   the annual estimates + SEs
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the Week 3 Lévy densities, colours, shock windows and Hessian helper.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from levy_models import (                       # noqa: E402
    _logpdf_vg, _logpdf_nig, _numerical_hessian,
    MODEL_COLOURS, SHOCK_PERIODS, SHOCK_COLOURS,
)
# Reuse the cached-returns loader from the Bayesian script.
sys.path.insert(0, _HERE)
from week4_bayesian import load_returns          # noqa: E402

ANNUALISE = np.sqrt(252.0)


# ════════════════════════════════════════════════════════════════════════════
# ZERO-MEAN MLE  (μ fixed at 0; reuses the levy_models densities)
# ════════════════════════════════════════════════════════════════════════════

def _nll_vg0(p, r):
    sigma, theta, nu = p
    if sigma <= 0 or nu <= 0:
        return np.inf
    ll = _logpdf_vg(r, 0.0, sigma, theta, nu)
    return -np.sum(ll) if np.all(np.isfinite(ll)) else np.inf


def fit_vg0(r, n_starts=3):
    """MLE for VG(μ=0, σ, θ, ν).  Returns params, SEs, loglik, AIC/BIC."""
    r = np.asarray(r)
    n = len(r)
    inits = [
        [r.std() * 0.80, -0.001, 0.20],
        [r.std() * 1.00,  0.000, 0.50],
        [r.std() * 0.60, -0.005, 0.10],
    ]
    bounds = [(1e-6, 0.50), (-0.30, 0.30), (0.01, 5.0)]
    best = None
    for init in inits[:n_starts]:
        res = minimize(_nll_vg0, init, args=(r,), method="L-BFGS-B",
                       bounds=bounds,
                       options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 3000})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    sigma, theta, nu = best.x
    loglik, k = -best.fun, 3
    try:
        H = _numerical_hessian(lambda p: _nll_vg0(p, r), best.x)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception:
        se = np.full(3, np.nan)
    return {"sigma": sigma, "theta": theta, "nu": nu,
            "se_sigma": se[0], "se_theta": se[1], "se_nu": se[2],
            "loglik": loglik, "aic": 2 * k - 2 * loglik,
            "bic": k * np.log(n) - 2 * loglik, "n": n}


def _nll_nig0(p, r):
    alpha, xi, delta = p
    if alpha <= 0 or delta <= 0 or not (-1.0 < xi < 1.0):
        return np.inf
    beta = xi * alpha
    ll = _logpdf_nig(r, 0.0, alpha, beta, delta)
    return -np.sum(ll) if np.all(np.isfinite(ll)) else np.inf


def fit_nig0(r, n_starts=3):
    """MLE for NIG(μ=0, α, β, δ) via the ξ=β/α reparametrisation (α>|β|)."""
    r = np.asarray(r)
    n = len(r)
    inits = [
        [ 80.0, -0.050, 0.008],
        [150.0, -0.100, 0.012],
        [ 50.0, -0.020, 0.005],
    ]
    bounds = [(1.0, 3000.0), (-0.999, 0.999), (1e-6, 0.50)]
    best = None
    for init in inits[:n_starts]:
        res = minimize(_nll_nig0, init, args=(r,), method="L-BFGS-B",
                       bounds=bounds,
                       options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 3000})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    alpha, xi, delta = best.x
    beta = xi * alpha
    loglik, k = -best.fun, 3
    try:
        H = _numerical_hessian(lambda p: _nll_nig0(p, r), best.x)
        cov_xi = np.linalg.inv(H)
        J = np.eye(3)
        J[1, 0], J[1, 1] = xi, alpha          # β = ξα  →  Jacobian to natural
        se = np.sqrt(np.diag(J @ cov_xi @ J.T))
    except Exception:
        se = np.full(3, np.nan)
    return {"alpha": alpha, "beta": beta, "delta": delta,
            "se_alpha": se[0], "se_beta": se[1], "se_delta": se[2],
            "loglik": loglik, "aic": 2 * k - 2 * loglik,
            "bic": k * np.log(n) - 2 * loglik, "n": n}


# ════════════════════════════════════════════════════════════════════════════
# YEAR-BY-YEAR FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_by_year(r_series, start=2000, end=2024, min_obs=120):
    """Fit VG(μ=0) and NIG(μ=0) on each calendar year; return a tidy frame."""
    rows = []
    for yr in range(start, end + 1):
        rr = r_series[r_series.index.year == yr].values.astype(float)
        if len(rr) < min_obs:
            print(f"  {yr}: only {len(rr)} obs — skipped")
            continue
        vg = fit_vg0(rr)
        ng = fit_nig0(rr)
        rows.append({
            "year": yr, "n": len(rr),
            # VG (σ annualised for reporting)
            "vg_sigma_ann": vg["sigma"] * ANNUALISE,
            "vg_sigma_ann_se": vg["se_sigma"] * ANNUALISE,
            "vg_theta": vg["theta"], "vg_theta_se": vg["se_theta"],
            "vg_nu": vg["nu"], "vg_nu_se": vg["se_nu"],
            # NIG (δ annualised for reporting)
            "nig_alpha": ng["alpha"], "nig_alpha_se": ng["se_alpha"],
            "nig_beta": ng["beta"], "nig_beta_se": ng["se_beta"],
            "nig_delta_ann": ng["delta"] * ANNUALISE,
            "nig_delta_ann_se": ng["se_delta"] * ANNUALISE,
        })
        print(f"  {yr}: VG σ_ann={rows[-1]['vg_sigma_ann']:.3f} "
              f"θ={vg['theta']:+.4f} ν={vg['nu']:.3f} | "
              f"NIG α={ng['alpha']:.1f} β={ng['beta']:+.2f} "
              f"δ_ann={rows[-1]['nig_delta_ann']:.3f}")
    return pd.DataFrame(rows).set_index("year")


# ════════════════════════════════════════════════════════════════════════════
# FIGURES  (split into per-model sections, coloured by model)
# ════════════════════════════════════════════════════════════════════════════

def _shade_shock_years(ax):
    """Shade the shock windows in the project's standard shock colours."""
    for label, (start, end) in SHOCK_PERIODS.items():
        x0 = pd.Timestamp(start).year + (pd.Timestamp(start).month - 1) / 12.0
        x1 = pd.Timestamp(end).year + (pd.Timestamp(end).month - 1) / 12.0
        ax.axvspan(x0, x1, color=SHOCK_COLOURS[label], alpha=0.85, lw=0, zorder=0)


def _shock_legend(fig):
    """Figure-level legend mapping the shock colours to their windows."""
    patches = [mpatches.Patch(color=SHOCK_COLOURS[lbl], alpha=0.85, label=lbl)
               for lbl in SHOCK_PERIODS]
    fig.legend(handles=patches, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.945))


def _panel(ax, years, vals, ses, colour, ylabel, hline=None, logy=False):
    if logy:
        # On a log axis the ±SE bars run negative in the Gaussian-limit years
        # (α, δ → ∞ with huge, meaningless SEs), so they are omitted here.
        ax.set_yscale("log")
    else:
        if hline is not None:
            ax.axhline(hline, color="0.7", lw=0.8, ls="--")
        else:
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
        finite = np.isfinite(ses)
        ax.errorbar(years[finite], vals[finite], yerr=ses[finite], fmt="none",
                    ecolor=colour, elinewidth=0.9, capsize=2, alpha=0.5, zorder=2)
    ax.plot(years, vals, "-o", color=colour, lw=1.3, ms=4.5, zorder=3)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="0.93", lw=0.6, zorder=0)
    ax.margins(x=0.02)


def make_vg_figure(df, save_dir):
    c = MODEL_COLOURS["VG"]
    yrs = df.index.values.astype(float)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for a in ax:
        _shade_shock_years(a)
    _panel(ax[0], yrs, df["vg_sigma_ann"].values, df["vg_sigma_ann_se"].values,
           c, "σ  (annualised scale)")
    _panel(ax[1], yrs, df["vg_theta"].values, df["vg_theta_se"].values,
           c, "θ  (asymmetry; <0 = left skew)")
    _panel(ax[2], yrs, df["vg_nu"].values, df["vg_nu_se"].values,
           c, "ν  (variance rate; tail heaviness)")
    ax[2].set_xlabel("Year", fontsize=10)
    fig.suptitle("Year-by-year Variance-Gamma fit (μ = 0); bars = ±1 SE",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _shock_legend(fig)
    out = os.path.join(save_dir, "week4_yearly_vg_params.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def make_nig_figure(df, save_dir):
    c = MODEL_COLOURS["NIG"]
    yrs = df.index.values.astype(float)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for a in ax:
        _shade_shock_years(a)
    _panel(ax[0], yrs, df["nig_alpha"].values, df["nig_alpha_se"].values,
           c, "α  (log; larger = lighter tails)", logy=True)
    _panel(ax[1], yrs, df["nig_beta"].values, df["nig_beta_se"].values,
           c, "β  (asymmetry; <0 = left skew)")
    _panel(ax[2], yrs, df["nig_delta_ann"].values, df["nig_delta_ann_se"].values,
           c, "δ  (annualised scale, log)", logy=True)
    ax[2].set_xlabel("Year", fontsize=10)
    fig.suptitle("Year-by-year Normal-Inverse-Gaussian fit (μ = 0); α, δ on log "
                 "axes (Gaussian-limit years 2004/05/23); β bars = ±1 SE",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _shock_legend(fig)
    out = os.path.join(save_dir, "week4_yearly_nig_params.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# PER-YEAR MARGINAL DENSITIES  (VG and NIG, 5-year blocks; Week-2 format)
# ════════════════════════════════════════════════════════════════════════════

def _shock_map():
    """Map each calendar year to the shock window it falls in (if any)."""
    m = {}
    for label, (start, end) in SHOCK_PERIODS.items():
        for yr in range(int(start[:4]), int(end[:4]) + 1):
            m.setdefault(yr, label)
    return m


def _make_marginal_figure(r_series, years_block, shock_map, save_dir):
    """One 5-year figure: per-year empirical KDE with fitted VG and NIG (μ=0).

    Mirrors the Week 2 marginal figures (week2_marginals_*), but overlays the
    two Lévy densities instead of the Gaussian and Student-t, with the same
    shock-coloured panel backgrounds.
    """
    period = f"{years_block[0]}-{years_block[-1]}"
    print(f"  marginals {period} ...")
    cvg, cnig = MODEL_COLOURS["VG"], MODEL_COLOURS["NIG"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.20, wspace=0.25)
    LINE_TOP = 0.66

    for col, year in enumerate(years_block):
        ax = axes[col]
        r_yr = r_series[r_series.index.year == year].values.astype(float)
        shock = shock_map.get(year)
        ax.set_facecolor(SHOCK_COLOURS[shock] if shock else "#f5f5f5")
        if len(r_yr) < 120:
            ax.set_title(str(year), fontsize=16, fontweight="bold")
            continue

        vg = fit_vg0(r_yr)
        ng = fit_nig0(r_yr)

        half = max(4.5 * r_yr.std(), 0.08)
        x = np.linspace(-half, half, 700)
        kde = stats.gaussian_kde(r_yr)
        ax.fill_between(x, kde(x), alpha=0.22, color="#5090c8", zorder=1)
        ax.plot(x, kde(x), color="#5090c8", lw=0.9, alpha=0.6, zorder=2)

        vg_pdf = np.exp(_logpdf_vg(x, 0.0, vg["sigma"], vg["theta"], vg["nu"]))
        ng_pdf = np.exp(_logpdf_nig(x, 0.0, ng["alpha"], ng["beta"], ng["delta"]))
        ax.plot(x, vg_pdf, color=cvg, lw=2.2, ls="-", zorder=4)
        ax.plot(x, ng_pdf, color=cnig, lw=2.0, ls="--", zorder=5)

        ax.axvline(0.0, ymin=0, ymax=LINE_TOP, color="#aaaaaa", lw=1.2, ls="--",
                   zorder=3)

        y_peak = max(kde(x).max(), np.nanmax(vg_pdf), np.nanmax(ng_pdf))
        ax.set_ylim(0, y_peak * 1.45)
        ax.set_title(str(year), fontsize=16, fontweight="bold",
                     color="#3a1800" if shock else "black", pad=8)

        info = (f"σ = {vg['sigma']*ANNUALISE:.0%}\n"
                f"VG ν = {vg['nu']:.2f}\n"
                f"NIG α = {ng['alpha']:.0f}")
        ax.text(0.04, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="#111111", linespacing=1.6,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82,
                          ec="#cccccc", lw=0.5))

        ax.set_xlim(-half, half)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yticks([])
        ax.set_xlabel("Daily log-return", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)

    legend_elements = [
        mpatches.Patch(color="#5090c8", alpha=0.5, label="Empirical KDE"),
        plt.Line2D([0], [0], color=cvg, lw=2.2, ls="-", label="Fitted VG (μ=0)"),
        plt.Line2D([0], [0], color=cnig, lw=2.0, ls="--", label="Fitted NIG (μ=0)"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=1.2, ls="--", label="x = 0"),
    ]
    shock_patches = [mpatches.Patch(color=SHOCK_COLOURS[lbl], alpha=0.75, label=lbl)
                     for lbl in SHOCK_PERIODS]
    fig.legend(handles=legend_elements + shock_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9, framealpha=0.95)
    fig.suptitle(f"S&P 500 annual return distributions: fitted VG and NIG "
                 f"(μ = 0)  {period}", fontsize=14, y=0.965)

    out = os.path.join(save_dir, f"week4_marginals_{years_block[0]}_{years_block[-1]}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {out}")


def plot_levy_marginals_by_period(r_series, save_dir, start=2000, end=2024):
    """Five 5-year figures of per-year VG and NIG marginal fits."""
    shock_map = _shock_map()
    years = list(range(start, end + 1))
    for i in range(0, len(years), 5):
        _make_marginal_figure(r_series, years[i:i + 5], shock_map, save_dir)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Week 4: year-by-year VG/NIG MLE (zero-mean)")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 64)
    print("Week 4 — year-by-year VG and NIG MLE (μ = 0)")
    print("=" * 64)
    r = load_returns(use_cache=not args.no_cache)
    df = fit_by_year(r, start=args.start, end=args.end)

    make_vg_figure(df, save_dir)
    make_nig_figure(df, save_dir)
    plot_levy_marginals_by_period(r, save_dir, start=args.start, end=args.end)

    data_dir = os.path.abspath(os.path.join(_HERE, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "week4_yearly_levy_params.csv")
    df.round(6).to_csv(out_csv)
    print(f"  annual estimates saved -> {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
