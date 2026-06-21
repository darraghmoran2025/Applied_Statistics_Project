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
    week4/data/week4_yearly_levy_params.csv   the annual estimates + SEs
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the Week 3 Lévy densities, colours, shock windows and Hessian helper.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from levy_models import (                       # noqa: E402
    _logpdf_vg, _logpdf_nig, _numerical_hessian,
    MODEL_COLOURS, SHOCK_PERIODS,
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
    """Light shading of the shock windows on a calendar-year x-axis."""
    for (start, end) in SHOCK_PERIODS.values():
        x0 = pd.Timestamp(start).year + (pd.Timestamp(start).month - 1) / 12.0
        x1 = pd.Timestamp(end).year + (pd.Timestamp(end).month - 1) / 12.0
        ax.axvspan(x0, x1, color="0.85", alpha=0.6, lw=0, zorder=0)


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
    fig.suptitle("Year-by-year Variance-Gamma fit (μ = 0)\n"
                 "shaded = shock windows; bars = ±1 SE", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    fig.suptitle("Year-by-year Normal-Inverse-Gaussian fit (μ = 0)\n"
                 "shaded = shock windows; α, δ on log axes (Gaussian-limit "
                 "years 2004/05/23); β bars = ±1 SE", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(save_dir, "week4_yearly_nig_params.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


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

    data_dir = os.path.abspath(os.path.join(_HERE, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "week4_yearly_levy_params.csv")
    df.round(6).to_csv(out_csv)
    print(f"  annual estimates saved -> {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
