"""
week7_backtest.py: rolling one-day-ahead VaR backtest with the
Christoffersen (1998) conditional coverage framework.

Each model is refitted on a rolling window of past returns and asked for
the next day's Value-at-Risk. A violation ("hit") is a realised return
below the VaR forecast. Three likelihood-ratio tests score the hit
sequence at each confidence level:

  LR_uc   Kupiec unconditional coverage: is the violation RATE right?
  LR_ind  Christoffersen independence: do violations CLUSTER? (first-order
          Markov alternative)
  LR_cc   conditional coverage = LR_uc + LR_ind, chi-squared with 2 df.

On top of the formal tests, the script reports the Basel traffic-light
view (rolling 250-day count of 99% violations: green <= 4, yellow 5-9,
red >= 10) and an FRTB-flavoured Expected Shortfall check at 97.5%: on
violation days, how does the average realised loss compare with the
average ES the model had promised?

Models: Gaussian, Laplace, Student-t, NIG. As in Weeks 4-6, the Laplace
stands in for the Variance-Gamma family as its symmetric special case
(theta = 0, nu = 1); the NIG carries the full four-parameter tail.

Run standalone:
    python week7_backtest.py
    python week7_backtest.py --window 250 --refit 21
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import xlogy
from scipy.integrate import quad

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week2", "code"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t  # noqa: E402
from levy_models import (fit_laplace, fit_nig,                      # noqa: E402
                         MODEL_COLOURS, SHOCK_PERIODS, SHOCK_COLOURS)

LEVELS = (0.95, 0.975, 0.99)          # VaR confidence levels
ES_LEVEL = 0.975                      # the FRTB ES confidence level
MODELS = ["Gaussian", "Laplace", "Student-t", "NIG"]


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════

def load_returns(path=os.path.join(_HERE, "..", "..", "week4", "data",
                                   "sp500_returns.csv")):
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "logret"
    print(f"Loaded {len(s)} returns from {path}")
    return s


# ════════════════════════════════════════════════════════════════════════════
# ONE-DAY-AHEAD VaR AND ES PER MODEL
# ════════════════════════════════════════════════════════════════════════════
# Convention throughout (matches compute_risk_measures in week2 and
# var_es_mc in week3): VaR at level a is the (1-a)-quantile of the return
# distribution, a negative number; ES is the mean return conditional on
# falling at or below that quantile.

def fit_all(x):
    """Refit the four models on one window of returns."""
    return {
        "Gaussian":  fit_gaussian(x),
        "Laplace":   fit_laplace(x),
        "Student-t": fit_student_t(x),
        "NIG":       fit_nig(x),
    }


def _nig_quantile(fit, p):
    """
    Lower-tail NIG quantile by Brent root-finding on the CDF.

    scipy's generic norminvgauss.ppf fails to bracket the root when a fit
    lands on the Gaussian-limit ridge (alpha very large), which rolling
    windows over calm markets regularly produce. Bracketing by hand with
    the NIG's own mean and standard deviation is robust in that regime.
    """
    from scipy.optimize import brentq
    dist = stats.norminvgauss(a=fit["alpha"] * fit["delta"],
                              b=fit["beta"] * fit["delta"],
                              loc=fit["mu"], scale=fit["delta"])
    gamma = np.sqrt(fit["alpha"]**2 - fit["beta"]**2)
    mean = fit["mu"] + fit["delta"] * fit["beta"] / gamma
    sd = np.sqrt(fit["delta"] * fit["alpha"]**2 / gamma**3)
    lo = mean - 8.0 * sd
    while dist.cdf(lo) > p and lo > mean - 500.0 * sd:
        lo = mean - 2.0 * (mean - lo)
    return brentq(lambda x: dist.cdf(x) - p, lo, mean, xtol=1e-12)


def var_forecast(name, fit, level):
    """(1-level)-quantile of the fitted distribution."""
    p = 1.0 - level
    if name == "Gaussian":
        return fit["mu"] + fit["sigma"] * stats.norm.ppf(p)
    if name == "Laplace":
        return fit["mu"] + fit["b"] * np.log(2.0 * p)   # p <= 0.5 branch
    if name == "Student-t":
        return fit["mu"] + fit["sigma"] * stats.t.ppf(p, df=fit["nu"])
    if name == "NIG":
        # scipy parameterisation: a = alpha*delta, b = beta*delta,
        # loc = mu, scale = delta (as used for the Week 4 random draws)
        return _nig_quantile(fit, p)
    raise ValueError(name)


def es_forecast(name, fit, level=ES_LEVEL):
    """Expected Shortfall E[X | X <= VaR] at the given level."""
    p = 1.0 - level
    var = var_forecast(name, fit, level)
    if name == "Gaussian":
        return fit["mu"] - fit["sigma"] * stats.norm.pdf(stats.norm.ppf(p)) / p
    if name == "Laplace":
        # memoryless exponential lower tail: one scale below the VaR
        return var - fit["b"]
    if name == "Student-t":
        q = stats.t.ppf(p, df=fit["nu"])
        return fit["mu"] - fit["sigma"] * (stats.t.pdf(q, df=fit["nu"]) / p) \
            * (fit["nu"] + q**2) / (fit["nu"] - 1.0)
    if name == "NIG":
        # no closed form: integrate x f(x) over the lower tail
        a, b = fit["alpha"] * fit["delta"], fit["beta"] * fit["delta"]
        integrand = lambda x: x * stats.norminvgauss.pdf(
            x, a=a, b=b, loc=fit["mu"], scale=fit["delta"])
        tail_mean, _ = quad(integrand, -0.5, var, limit=200)
        return tail_mean / p
    raise ValueError(name)


# ════════════════════════════════════════════════════════════════════════════
# ROLLING BACKTEST
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(r, window=500, refit_every=21):
    """
    Walk forward through the sample. Every refit_every days the four models
    are refitted on the trailing `window` returns; between refits the
    parameters are held fixed, which mirrors the (at least) monthly update
    cycle a desk would run. Each day the previous fit issues VaR forecasts
    for today's return, which the day then confirms or violates.
    """
    dates, values = r.index, r.values
    n = len(values)
    rows = []
    fits = None
    n_refits = 0

    forecasts = None
    for t in range(window, n):
        if (t - window) % refit_every == 0:
            fits = fit_all(values[t - window:t])
            # forecasts are constant between refits, so compute them once
            forecasts = {}
            for name in MODELS:
                for level in LEVELS:
                    forecasts[f"var{int(level * 1000)}_{name}"] = \
                        var_forecast(name, fits[name], level)
                forecasts[f"es{int(ES_LEVEL * 1000)}_{name}"] = \
                    es_forecast(name, fits[name])
            n_refits += 1
            if n_refits % 50 == 0:
                print(f"  refit {n_refits} at {dates[t].date()}", flush=True)
        row = {"date": dates[t], "ret": values[t], **forecasts}
        for name in MODELS:
            for level in LEVELS:
                tag = int(level * 1000)
                row[f"hit{tag}_{name}"] = int(values[t] < forecasts[f"var{tag}_{name}"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    print(f"Backtest: {len(df)} out-of-sample days "
          f"({df.index[0].date()} to {df.index[-1].date()}), "
          f"{n_refits} refits of window {window}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# CHRISTOFFERSEN (1998) LIKELIHOOD-RATIO TESTS
# ════════════════════════════════════════════════════════════════════════════

def kupiec_uc(hits, p):
    """
    Unconditional coverage. H0: P(hit) = p.
    LR_uc ~ chi2(1). xlogy keeps 0*log(0) = 0 when there are no hits.
    """
    n = len(hits)
    n1 = int(hits.sum())
    n0 = n - n1
    pi = n1 / n
    ll0 = xlogy(n0, 1.0 - p) + xlogy(n1, p)
    ll1 = xlogy(n0, 1.0 - pi) + xlogy(n1, pi)
    lr = -2.0 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, df=1), n1, pi


def christoffersen_ind(hits):
    """
    Independence against a first-order Markov alternative. H0: the chance
    of a hit today does not depend on whether yesterday was a hit.
    LR_ind ~ chi2(1). Returns NaN if the sequence has no hits to condition on.
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    if n10 + n11 == 0:
        return np.nan, np.nan, n11
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll1 = (xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01)
           + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11))
    lr = -2.0 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, df=1), n11


def test_table(df):
    """All three LR tests for every model and confidence level."""
    n = len(df)
    rows = []
    for name in MODELS:
        for level in LEVELS:
            p = 1.0 - level
            hits = df[f"hit{int(level * 1000)}_{name}"].values
            lr_uc, p_uc, n1, pi = kupiec_uc(hits, p)
            lr_ind, p_ind, n11 = christoffersen_ind(hits)
            lr_cc = lr_uc + lr_ind
            rows.append({
                "model": name, "level": level,
                "n": n, "expected": n * p, "hits": n1,
                "rate_%": 100.0 * pi,
                "hit_after_hit": n11,
                "LR_uc": lr_uc, "p_uc": p_uc,
                "LR_ind": lr_ind, "p_ind": p_ind,
                "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, df=2),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# BASEL TRAFFIC LIGHT (rolling 250-day count of 99% violations)
# ════════════════════════════════════════════════════════════════════════════

def basel_traffic_light(df):
    """
    Rolling 250-day violation count at the 99% level, the quantity the
    Basel backtesting multiplier keys off: green <= 4, yellow 5-9, red >= 10.
    Returns the rolling counts plus the share of days each model spends in
    each zone.
    """
    counts = pd.DataFrame({
        name: df[f"hit990_{name}"].rolling(250).sum() for name in MODELS
    }).dropna()
    rows = []
    for name in MODELS:
        c = counts[name]
        rows.append({
            "model": name,
            "green_%": 100.0 * float((c <= 4).mean()),
            "yellow_%": 100.0 * float(((c >= 5) & (c <= 9)).mean()),
            "red_%": 100.0 * float((c >= 10).mean()),
            "worst_250d_count": int(c.max()),
        })
    return counts, pd.DataFrame(rows).set_index("model")


# ════════════════════════════════════════════════════════════════════════════
# EXPECTED SHORTFALL CHECK AT 97.5% (the FRTB level)
# ════════════════════════════════════════════════════════════════════════════

def es_check(df, level=ES_LEVEL):
    """
    On days that breach the 97.5% VaR, compare the average realised loss
    with the average ES the model had forecast for those same days. A
    ratio above 1 means the tail bites harder than the model promised.
    """
    tag = int(level * 1000)
    rows = []
    for name in MODELS:
        mask = df[f"hit{tag}_{name}"] == 1
        realised = df.loc[mask, "ret"].mean()
        promised = df.loc[mask, f"es{tag}_{name}"].mean()
        rows.append({
            "model": name, "breach_days": int(mask.sum()),
            "mean_realised_%": 100.0 * realised,
            "mean_forecast_ES_%": 100.0 * promised,
            "shortfall_ratio": realised / promised,
        })
    return pd.DataFrame(rows).set_index("model")


# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════

def _shade_shocks(ax):
    for label, (start, end) in SHOCK_PERIODS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SHOCK_COLOURS[label], zorder=0)


def plot_var_series(df, save_dir):
    """2x2 grid: each model's 99% VaR line over the returns, hits marked."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), MODELS):
        _shade_shocks(ax)
        ax.plot(df.index, 100 * df["ret"], ".", color="0.75", ms=1.5,
                zorder=1, label="daily return")
        ax.plot(df.index, 100 * df[f"var990_{name}"],
                color=MODEL_COLOURS[name], lw=1.0, zorder=3,
                label="99% VaR forecast")
        hits = df[df[f"hit990_{name}"] == 1]
        ax.plot(hits.index, 100 * hits["ret"], "x", color="black", ms=4,
                zorder=4, label=f"violations ({len(hits)})")
        ax.set_title(name)
        ax.set_ylim(-14, 8)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    for ax in axes[:, 0]:
        ax.set_ylabel("daily return (%)")
    fig.suptitle("Rolling one-day 99% VaR forecasts and violations", y=0.995)
    fig.tight_layout()
    out = os.path.join(save_dir, "week7_var_series.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_basel(counts, save_dir):
    """Rolling 250-day 99% violation counts against the Basel zones."""
    fig, ax = plt.subplots(figsize=(12, 5))
    top = max(12.0, counts.values.max() + 1)
    ax.axhspan(0, 4.5, color="#d4edda", zorder=0)
    ax.axhspan(4.5, 9.5, color="#fff3cd", zorder=0)
    ax.axhspan(9.5, top, color="#f8d7da", zorder=0)
    for name in MODELS:
        ax.plot(counts.index, counts[name], color=MODEL_COLOURS[name],
                lw=1.2, label=name)
    ax.set_ylim(0, top)
    ax.set_ylabel("99% VaR violations, trailing 250 days")
    ax.set_title("Basel traffic light: green $\\leq$ 4, yellow 5-9, red $\\geq$ 10")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(save_dir, "week7_basel_traffic.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_cumulative_hits(df, save_dir):
    """Cumulative violations against the expected straight line."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, level in zip(axes, (0.95, 0.99)):
        tag = int(level * 1000)
        p = 1.0 - level
        _shade_shocks(ax)
        t = np.arange(1, len(df) + 1)
        for name in MODELS:
            ax.plot(df.index, df[f"hit{tag}_{name}"].cumsum(),
                    color=MODEL_COLOURS[name], lw=1.2, label=name)
        ax.plot(df.index, p * t, "k--", lw=1.0, label="expected")
        ax.set_title(f"{level:.0%} VaR")
        ax.set_ylabel("cumulative violations")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("Cumulative VaR violations vs the nominal rate", y=0.99)
    fig.tight_layout()
    out = os.path.join(save_dir, "week7_cumulative_hits.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Week 7 rolling VaR backtest")
    ap.add_argument("--window", type=int, default=500,
                    help="rolling estimation window in trading days")
    ap.add_argument("--refit", type=int, default=21,
                    help="refit frequency in trading days")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    ap.add_argument("--data_dir", default=os.path.join(_HERE, "..", "data"))
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    r = load_returns()
    df = run_backtest(r, window=args.window, refit_every=args.refit)
    df.to_csv(os.path.join(args.data_dir, "week7_var_series.csv"))

    tests = test_table(df)
    tests.to_csv(os.path.join(args.data_dir, "week7_tests.csv"), index=False)
    print("\nChristoffersen test table")
    print(tests.round(4).to_string(index=False))

    counts, zones = basel_traffic_light(df)
    zones.to_csv(os.path.join(args.data_dir, "week7_basel_zones.csv"))
    print("\nBasel traffic-light zone shares (99% VaR, trailing 250 days)")
    print(zones.round(1).to_string())

    es = es_check(df)
    es.to_csv(os.path.join(args.data_dir, "week7_es_check.csv"))
    print(f"\nES check at {ES_LEVEL:.1%} (breach days only)")
    print(es.round(3).to_string())

    plot_var_series(df, args.save_dir)
    plot_basel(counts, args.save_dir)
    plot_cumulative_hits(df, args.save_dir)


if __name__ == "__main__":
    main()
