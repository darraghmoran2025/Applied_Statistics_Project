"""
week6_weekday.py: day-of-week structure in S&P 500 daily returns.

Fits the models separately by weekday and tests the two week definitions
from the Week 6 plan against each other:

  A. "Monday (inclusive) -> Friday": one set of parameters for the whole
     week. Monday returns span the weekend (Friday close to Monday close),
     so the weekend gap is included rather than modelled separately.
  B. "Monday -> Midweek -> Friday": three groups, {Mon}, {Tue, Wed, Thu},
     {Fri}, each with its own parameters.
  C. Five separate weekdays, the fully saturated version.

Nested likelihood-ratio tests (Gaussian) decide whether the extra
day-of-week parameters earn their keep. Per-day Student-t fits show
whether tail heaviness, not just scale, varies through the week.

Run standalone:
    python week6_weekday.py
    python week6_weekday.py --save_dir ../figures
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week2", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t  # noqa: E402

ANN = np.sqrt(252.0)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

GROUPINGS = {
    "A: pooled week (Mon inclusive -> Fri)": [DAYS],
    "B: Mon / midweek / Fri":                [["Monday"], ["Tuesday", "Wednesday", "Thursday"], ["Friday"]],
    "C: five separate days":                 [[d] for d in DAYS],
}


def load_returns(path=os.path.join(_HERE, "..", "..", "week4", "data",
                                   "sp500_returns.csv")):
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "logret"
    print(f"Loaded {len(s)} returns from {path}")
    return s


def per_day_table(r):
    """Gaussian and Student-t MLE per weekday."""
    rows = []
    for day in DAYS:
        x = r[r.index.day_name() == day].values
        g = fit_gaussian(x)
        try:
            t = fit_student_t(x)
            nu, se_nu = t["nu"], t["se_nu"]
        except Exception:
            nu, se_nu = np.nan, np.nan
        rows.append({
            "day": day, "n": len(x),
            "mean_bp": g["mu"] * 1e4,
            "sigma_ann_%": g["sigma"] * ANN * 100,
            "se_sigma_ann_%": g["se_sigma"] * ANN * 100,
            "t_nu": nu, "se_nu": se_nu,
            "ex_kurt": stats.kurtosis(x),
            "loglik_gauss": g["loglik"],
        })
    return pd.DataFrame(rows).set_index("day")


def grouping_loglik(r, groups):
    """Total Gaussian log-likelihood when each group gets its own (mu, sigma)."""
    total, k = 0.0, 0
    for grp in groups:
        x = r[r.index.day_name().isin(grp)].values
        total += fit_gaussian(x)["loglik"]
        k += 2
    return total, k


def lr_tests(r):
    """Nested LR tests between the week definitions."""
    fits = {name: grouping_loglik(r, groups) for name, groups in GROUPINGS.items()}
    print("\nGaussian log-likelihood by week definition:")
    for name, (ll, k) in fits.items():
        print(f"  {name:<40} loglik = {ll:,.1f}   k = {k}")

    pairs = [
        ("B: Mon / midweek / Fri", "A: pooled week (Mon inclusive -> Fri)"),
        ("C: five separate days",  "A: pooled week (Mon inclusive -> Fri)"),
        ("C: five separate days",  "B: Mon / midweek / Fri"),
    ]
    print("\nLikelihood-ratio tests (larger model vs smaller):")
    recs = []
    for big, small in pairs:
        ll1, k1 = fits[big]
        ll0, k0 = fits[small]
        lr = 2.0 * (ll1 - ll0)
        df = k1 - k0
        p = stats.chi2.sf(lr, df)
        print(f"  {big:<28} vs {small:<40} LR = {lr:7.2f}  df = {df}  p = {p:.4f}")
        recs.append({"big": big, "small": small, "LR": lr, "df": df, "p": p})
    return pd.DataFrame(recs)


def make_figure(day_df, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(day_df))
    ax1.bar(x, day_df["sigma_ann_%"], color="steelblue", alpha=0.8,
            yerr=1.96 * day_df["se_sigma_ann_%"], capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d[:3] for d in day_df.index])
    ax1.set_ylabel("Annualised volatility (%)")
    ax1.set_title("Gaussian volatility by weekday (95% CI)")
    ax1.grid(axis="y", color="0.92", lw=0.6, zorder=0)

    ax2.bar(x, day_df["t_nu"], color="crimson", alpha=0.8,
            yerr=1.96 * day_df["se_nu"], capsize=4)
    ax2.axhline(2.648, color="black", ls="--", lw=1,
                label="full-sample nu = 2.648")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[:3] for d in day_df.index])
    ax2.set_ylabel("Student-t nu")
    ax2.set_title("Tail parameter by weekday (95% CI)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", color="0.92", lw=0.6, zorder=0)

    fig.suptitle("Day-of-week structure, S&P 500 daily returns 2000-2024")
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "week6_weekday_params.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Week 6: day-of-week MLEs")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    args = ap.parse_args()

    r = load_returns()

    print("\nPer-weekday MLEs (mean in basis points/day, sigma annualised):")
    day_df = per_day_table(r)
    print(day_df.to_string(float_format=lambda v: f"{v:,.3f}"))

    lr_df = lr_tests(r)

    data_dir = os.path.join(_HERE, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    day_df.to_csv(os.path.join(data_dir, "week6_weekday_params.csv"))
    lr_df.to_csv(os.path.join(data_dir, "week6_weekday_lr.csv"), index=False)

    make_figure(day_df, os.path.abspath(args.save_dir))
    print("\nWeekday analysis complete.")


if __name__ == "__main__":
    main()
