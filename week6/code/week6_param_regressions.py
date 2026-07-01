"""
week6_param_regressions.py: quarterly Levy parameter fits and one
regression per parameter.

The Week 6 plan asks for a regression-based treatment of the Levy
parameters: regress each individual parameter separately, with the
regression as a predictor for the delta parameter first, then more
predictors added. Yearly fits (Week 4) give only 25 points, so this
script refits VG(sigma, theta, nu) and NIG(alpha, beta, delta) with
mu = 0 one CALENDAR QUARTER at a time (100 quarters, roughly 63 daily
returns each), then runs, for every parameter:

  spec 1:  param_q ~ avg VIX_q                    (the baseline)
  spec 2:  param_q ~ avg VIX_q + realised vol_q + drawdown_q + param_(q-1)

and an AR(1), param_q ~ param_(q-1), which is the plan's "two parameters
as a function of themselves" applied to all six (delta and alpha are the
two headline ones). Standard errors are Newey-West (Bartlett, 4 lags).

Identification caveat: with ~63 observations the scale parameters
(delta, sigma) are well determined, but the tail parameters are noisy
and NIG alpha runs into its Gaussian limit in calm quarters (alpha
large and weakly identified). Alpha is therefore regressed in logs and
quarters at the optimiser bound are dropped from its regression.

The zero-mean fits mirror fit_vg0 / fit_nig0 in
week4/code/week4_yearly_levy_mle.py; they are restated here so this
script does not import the PyMC stack.

Run standalone:
    python week6_param_regressions.py
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from levy_models import _logpdf_vg, _logpdf_nig, SHOCK_PERIODS  # noqa: E402
from week3_leadup_regression import ols_hac  # noqa: E402

ANN = np.sqrt(252.0)
ALPHA_BOUND = 3000.0

SHADE_COLOURS = {
    "Dot-com crash":  "#f4b400",
    "GFC":            "#e2504e",
    "COVID-19":       "#1f9ec4",
    "Fed rate hikes": "#3fa34d",
}


# ── zero-mean quarterly fits (mirror week4_yearly_levy_mle) ──────────────────

def _nll_vg0(p, r):
    sigma, theta, nu = p
    if sigma <= 0 or nu <= 0:
        return np.inf
    ll = _logpdf_vg(r, 0.0, sigma, theta, nu)
    return -np.sum(ll) if np.all(np.isfinite(ll)) else np.inf


def fit_vg0(r):
    r = np.asarray(r)
    inits = [[r.std() * 0.8, -0.001, 0.2], [r.std(), 0.0, 0.5],
             [r.std() * 0.6, -0.005, 0.1]]
    bounds = [(1e-6, 0.50), (-0.30, 0.30), (0.01, 5.0)]
    best = None
    for init in inits:
        res = minimize(_nll_vg0, init, args=(r,), method="L-BFGS-B",
                       bounds=bounds, options={"ftol": 1e-14, "maxiter": 3000})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    sigma, theta, nu = best.x
    return {"vg_sigma_ann": sigma * ANN, "vg_theta": theta, "vg_nu": nu}


def _nll_nig0(p, r):
    alpha, xi, delta = p
    if alpha <= 0 or delta <= 0 or not (-1.0 < xi < 1.0):
        return np.inf
    ll = _logpdf_nig(r, 0.0, alpha, xi * alpha, delta)
    return -np.sum(ll) if np.all(np.isfinite(ll)) else np.inf


def fit_nig0(r):
    r = np.asarray(r)
    inits = [[80.0, -0.05, 0.008], [150.0, -0.10, 0.012], [50.0, -0.02, 0.005]]
    bounds = [(1.0, ALPHA_BOUND), (-0.999, 0.999), (1e-6, 0.50)]
    best = None
    for init in inits:
        res = minimize(_nll_nig0, init, args=(r,), method="L-BFGS-B",
                       bounds=bounds, options={"ftol": 1e-14, "maxiter": 3000})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    alpha, xi, delta = best.x
    return {"nig_alpha": alpha, "nig_beta": xi * alpha,
            "nig_delta_ann": delta * ANN,
            "nig_alpha_bound": alpha > 0.99 * ALPHA_BOUND}


# ── quarterly panel ──────────────────────────────────────────────────────────

def build_panel():
    r = pd.read_csv(os.path.join(_HERE, "..", "..", "week4", "data",
                                 "sp500_returns.csv"),
                    index_col=0, parse_dates=True).iloc[:, 0]
    vix = pd.read_csv(os.path.join(_HERE, "..", "data", "vix.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0]
    price = np.exp(r.cumsum())
    ddown = price / price.rolling(252, min_periods=60).max() - 1.0

    rows = []
    for q, x in r.groupby(r.index.to_period("Q")):
        if len(x) < 50:
            continue
        row = {"quarter": str(q), "n": len(x),
               "avg_vix": float(vix.loc[x.index.min():x.index.max()].mean()),
               "rv_ann": float(x.std() * ANN * 100),
               "ret_q": float(x.sum() * 100),
               "ddown_end": float(ddown.loc[:x.index.max()].iloc[-1] * 100)}
        row.update(fit_vg0(x.values))
        row.update(fit_nig0(x.values))
        rows.append(row)
        print(f"  {q}  n={len(x)}  VIX {row['avg_vix']:5.1f}  "
              f"delta_ann {row['nig_delta_ann']:.3f}  alpha {row['nig_alpha']:7.1f}"
              f"{'  [alpha at bound]' if row['nig_alpha_bound'] else ''}")
    return pd.DataFrame(rows).set_index("quarter")


# ── regressions ──────────────────────────────────────────────────────────────

PARAMS = ["nig_delta_ann", "nig_alpha_log", "nig_beta",
          "vg_sigma_ann", "vg_theta", "vg_nu"]


def run_regressions(panel):
    panel = panel.copy()
    panel["nig_alpha_log"] = np.log(panel["nig_alpha"])
    # In near-Gaussian quarters the NIG likelihood has a ridge: alpha and
    # delta grow together with only delta * alpha^2 / gamma^3 (the variance)
    # identified. Delta values from those quarters are artifacts of the
    # ridge, so the delta regressions run on the well-identified subset.
    panel["weak_id"] = (panel["nig_alpha"] > 500) | panel["nig_alpha_bound"]
    print(f"\nWeakly identified quarters (NIG alpha > 500 or at bound): "
          f"{int(panel['weak_id'].sum())} of {len(panel)}")
    results = []
    print("\nPer-parameter regressions (Newey-West SEs, 4 lags):")
    print(f"  {'parameter':<15}{'spec':<26}{'R^2':>7}{'VIX beta':>10}{'t':>7}{'n':>5}")
    for param in PARAMS:
        d = panel.copy()
        if param == "nig_alpha_log":
            d = d[~d["nig_alpha_bound"]]
        if param == "nig_delta_ann":
            d = d[~d["weak_id"]]
        y = d[param].values
        lag = d[param].shift(1).values

        # spec 1: VIX only
        X1 = d[["avg_vix"]].values
        m1 = ols_hac(X1, y, L=4)
        # spec 2: add realised vol, drawdown and the parameter's own lag
        keep = ~np.isnan(lag)
        X2 = np.column_stack([d["avg_vix"].values, d["rv_ann"].values,
                              d["ddown_end"].values, lag])[keep]
        m2 = ols_hac(X2, y[keep], L=4)

        for spec, m, note in [("VIX only", m1, ""),
                              ("+ rv, drawdown, own lag", m2, "")]:
            print(f"  {param:<15}{spec:<26}{m['r2']:>7.3f}"
                  f"{m['beta'][1]:>10.4f}{m['t'][1]:>7.2f}{m['n']:>5}")
            results.append({"param": param, "spec": spec, "r2": m["r2"],
                            "vix_beta": m["beta"][1], "vix_t": m["t"][1],
                            "n": m["n"]})

        # AR(1): the parameter as a function of itself
        m_ar = ols_hac(lag[keep].reshape(-1, 1), y[keep], L=4)
        print(f"  {param:<15}{'AR(1): own lag only':<26}{m_ar['r2']:>7.3f}"
              f"{m_ar['beta'][1]:>10.4f}{m_ar['t'][1]:>7.2f}{m_ar['n']:>5}")
        results.append({"param": param, "spec": "AR(1)", "r2": m_ar["r2"],
                        "vix_beta": m_ar["beta"][1], "vix_t": m_ar["t"][1],
                        "n": m_ar["n"]})
    return panel, pd.DataFrame(results)


# ── figures ──────────────────────────────────────────────────────────────────

def _qindex(panel):
    return pd.PeriodIndex(panel.index, freq="Q").to_timestamp()


def _shade(ax):
    for name, (start, end) in SHOCK_PERIODS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SHADE_COLOURS[name], alpha=0.30, lw=0, zorder=0)


def make_figures(panel, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = _qindex(panel)

    # 1. quarterly NIG delta and alpha through time. Ridge (weakly
    #    identified) quarters are shown as grey crosses in both panels:
    #    there delta and alpha are determined only through their ratio,
    #    so their individual values carry no information.
    ok = ~panel["weak_id"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    _shade(ax1)
    d_ok = (panel["nig_delta_ann"] * 100).where(ok)
    ax1.semilogy(ts, d_ok, color="darkorange", lw=1.0, marker="o", ms=3)
    ax1.semilogy(ts[~ok], panel.loc[~ok, "nig_delta_ann"] * 100, ls="none",
                 marker="x", color="gray", label="weakly identified (ridge)")
    ax1.set_ylabel("NIG delta (annualised %, log scale)")
    ax1.set_title("Quarterly NIG fits (mu = 0), 2000-2024")
    ax1.legend(fontsize=8, loc="upper right")
    _shade(ax2)
    a_ok = panel["nig_alpha"].where(ok)
    ax2.semilogy(ts, a_ok, color="darkorange", lw=1.0, marker="o", ms=3)
    ax2.semilogy(ts[~ok], panel.loc[~ok, "nig_alpha"], ls="none", marker="x",
                 color="gray")
    ax2.set_ylabel("NIG alpha (log scale)")
    ax2.set_xlabel("Quarter")
    handles = [mpatches.Patch(facecolor=SHADE_COLOURS[n], alpha=0.5, label=n)
               for n in SHOCK_PERIODS]
    ax2.legend(handles=handles, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_quarterly_nig.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # 2. delta vs VIX, and delta as a function of itself
    # (well-identified quarters only; ridge quarters carry no delta signal)
    panel = panel[~panel["weak_id"]]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
    x, y = panel["avg_vix"].values, panel["nig_delta_ann"].values * 100
    axL.scatter(x, y, s=18, color="darkorange", alpha=0.7, edgecolors="none")
    b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    axL.plot(xs, np.polyval(b, xs), color="black", lw=1.3,
             label=f"slope {b[0]:.2f}, R^2 {r2:.2f}")
    axL.set_xlabel("Average VIX in quarter")
    axL.set_ylabel("NIG delta (annualised %)")
    axL.set_title(f"Delta against the VIX ({len(panel)} well-identified quarters)")
    axL.legend(fontsize=9)
    axL.grid(color="0.92", lw=0.6, zorder=0)

    y1, y0 = y[1:], y[:-1]
    axR.scatter(y0, y1, s=18, color="darkorange", alpha=0.7, edgecolors="none")
    b = np.polyfit(y0, y1, 1)
    xs = np.array([y0.min(), y0.max()])
    r2 = np.corrcoef(y0, y1)[0, 1] ** 2
    axR.plot(xs, np.polyval(b, xs), color="black", lw=1.3,
             label=f"AR(1) slope {b[0]:.2f}, R^2 {r2:.2f}")
    axR.set_xlabel("delta, quarter q-1 (%)")
    axR.set_ylabel("delta, quarter q (%)")
    axR.set_title("Delta as a function of itself")
    axR.legend(fontsize=9)
    axR.grid(color="0.92", lw=0.6, zorder=0)
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_delta_regressions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Week 6: quarterly parameter regressions")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    args = ap.parse_args()

    print("Fitting VG and NIG quarter by quarter (mu = 0) ...")
    panel = build_panel()
    print(f"\nPanel: {len(panel)} quarters, "
          f"{int(panel['nig_alpha_bound'].sum())} with alpha at the bound")

    panel, reg = run_regressions(panel)

    data_dir = os.path.join(_HERE, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    panel.to_csv(os.path.join(data_dir, "week6_quarterly_params.csv"))
    reg.to_csv(os.path.join(data_dir, "week6_param_regressions.csv"), index=False)

    make_figures(panel, os.path.abspath(args.save_dir))
    print("\nParameter regressions complete.")


if __name__ == "__main__":
    main()
