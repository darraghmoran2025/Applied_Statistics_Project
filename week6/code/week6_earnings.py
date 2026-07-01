"""
week6_earnings.py: volatility around quarterly earnings seasons.

The index itself has no earnings dates, so this uses the aggregate
reporting calendar as a proxy: the bulk of S&P 500 constituents report
in the month starting roughly two weeks after each quarter ends. Each
earnings season is defined as the window from the 15th of January,
April, July and October to the 14th of the following month. This is a
proxy, and the writeup says so; single-name event studies would need
constituent-level dates.

Two exhibits:
  1. In-season vs out-of-season distributions: annualised volatility,
     Student-t nu, Laplace scale, kurtosis, plus zero-mean NIG fits.
  2. An event-study profile: mean daily |return| (annualised) over the
     21 trading days before each season, the season itself, and the 21
     days after, averaged across the 100 seasons.

Run standalone:
    python week6_earnings.py
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week2", "code"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t  # noqa: E402
from levy_models import fit_laplace, _logpdf_nig  # noqa: E402

ANN = np.sqrt(252.0)
SEASON_STARTS = [(1, 15), (4, 15), (7, 15), (10, 15)]
SEASON_DAYS = 21          # trading days per season window
FLANK_DAYS = 21           # trading days before and after


def load_returns():
    r = pd.read_csv(os.path.join(_HERE, "..", "..", "week4", "data",
                                 "sp500_returns.csv"),
                    index_col=0, parse_dates=True).iloc[:, 0]
    print(f"Loaded {len(r)} returns")
    return r


def season_mask(index):
    """Boolean mask: True on days inside an earnings-season window."""
    mask = pd.Series(False, index=index)
    for year in range(index.min().year, index.max().year + 1):
        for month, day in SEASON_STARTS:
            start = pd.Timestamp(year, month, day)
            end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            mask.loc[start:end] = True
    return mask


def fit_nig0(r):
    """Zero-mean NIG fit, as in week6_param_regressions.py."""
    def nll(p):
        alpha, xi, delta = p
        if alpha <= 0 or delta <= 0 or not (-1.0 < xi < 1.0):
            return np.inf
        ll = _logpdf_nig(r, 0.0, alpha, xi * alpha, delta)
        return -np.sum(ll) if np.all(np.isfinite(ll)) else np.inf
    best = None
    for init in [[80.0, -0.05, 0.008], [150.0, -0.1, 0.012], [50.0, -0.02, 0.005]]:
        res = minimize(nll, init, method="L-BFGS-B",
                       bounds=[(1.0, 3000.0), (-0.999, 0.999), (1e-6, 0.5)],
                       options={"ftol": 1e-14, "maxiter": 3000})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    alpha, xi, delta = best.x
    return {"alpha": alpha, "beta": xi * alpha, "delta_ann": delta * ANN}


def in_out_table(r, mask):
    rows = []
    for label, x in [("in season", r[mask].values),
                     ("out of season", r[~mask].values)]:
        g = fit_gaussian(x)
        t = fit_student_t(x)
        lap = fit_laplace(x)
        nig = fit_nig0(x)
        rows.append({
            "window": label, "n": len(x),
            "sigma_ann_%": g["sigma"] * ANN * 100,
            "se_sigma_%": g["se_sigma"] * ANN * 100,
            "t_nu": t["nu"], "se_nu": t["se_nu"],
            "laplace_b_bp": lap["b"] * 1e4,
            "ex_kurt": stats.kurtosis(x),
            "nig_alpha": nig["alpha"], "nig_beta": nig["beta"],
            "nig_delta_ann_%": nig["delta_ann"] * 100,
        })
    return pd.DataFrame(rows).set_index("window")


def event_profile(r):
    """Average annualised vol before, during and after each season."""
    idx = r.index
    recs = []
    for year in range(idx.min().year, idx.max().year + 1):
        for month, day in SEASON_STARTS:
            start = pd.Timestamp(year, month, day)
            if start < idx.min() or start > idx.max():
                continue
            pos = idx.searchsorted(start)
            pre = r.iloc[max(0, pos - FLANK_DAYS):pos]
            dur = r.iloc[pos:pos + SEASON_DAYS]
            post = r.iloc[pos + SEASON_DAYS:pos + SEASON_DAYS + FLANK_DAYS]
            if len(pre) < FLANK_DAYS or len(post) < FLANK_DAYS:
                continue
            recs.append({"season": f"{year}Q{[1,4,7,10].index(month)+1}",
                         "pre": pre.std() * ANN * 100,
                         "during": dur.std() * ANN * 100,
                         "post": post.std() * ANN * 100})
    prof = pd.DataFrame(recs).set_index("season")
    print(f"\nEvent profile across {len(prof)} seasons "
          f"(annualised vol, mean with SE):")
    for col in ["pre", "during", "post"]:
        m, se = prof[col].mean(), prof[col].std() / np.sqrt(len(prof))
        print(f"  {col:<8}{m:6.2f}%  (SE {se:.2f})")
    # paired tests: same season's window vs window
    for a, b in [("pre", "during"), ("during", "post"), ("pre", "post")]:
        d = prof[b] - prof[a]
        t = d.mean() / (d.std() / np.sqrt(len(d)))
        print(f"  {b} minus {a}: {d.mean():+.2f}pp  (paired t = {t:+.2f})")
    return prof


def make_figure(prof, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means = [prof[c].mean() for c in ["pre", "during", "post"]]
    ses = [prof[c].std() / np.sqrt(len(prof)) for c in ["pre", "during", "post"]]
    ax.bar(range(3), means, yerr=[1.96 * s for s in ses], capsize=5,
           color=["0.7", "seagreen", "0.7"], alpha=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{FLANK_DAYS}d before", f"season ({SEASON_DAYS}d)",
                        f"{FLANK_DAYS}d after"])
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title(f"Volatility around earnings seasons, mean of "
                 f"{len(prof)} seasons (95% CI)")
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_earnings_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Week 6: earnings-season volatility")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    args = ap.parse_args()

    r = load_returns()
    mask = season_mask(r.index)
    print(f"Earnings-season days: {int(mask.sum())} of {len(r)} "
          f"({mask.mean():.1%} of the sample)")

    print("\nIn-season vs out-of-season fits:")
    table = in_out_table(r, mask)
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    prof = event_profile(r)

    data_dir = os.path.join(_HERE, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    table.to_csv(os.path.join(data_dir, "week6_earnings_fits.csv"))
    prof.to_csv(os.path.join(data_dir, "week6_earnings_profile.csv"))

    make_figure(prof, os.path.abspath(args.save_dir))
    print("\nEarnings analysis complete.")


if __name__ == "__main__":
    main()
