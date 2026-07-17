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
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t  # noqa: E402
from levy_models import SHOCK_PERIODS  # noqa: E402

ANN = np.sqrt(252.0)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Week-open / week-close lens: Monday opens the trading week and Friday
# closes it, with Tuesday to Thursday as the midweek in between.
WEEK_GROUPS = {
    "Monday (week open)":  ["Monday"],
    "Midweek (Tue-Thu)":   ["Tuesday", "Wednesday", "Thursday"],
    "Friday (week close)": ["Friday"],
}

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


def _group_stats(x, min_n_for_t=50):
    """Volatility, variance and fat-tail measures for one set of returns.

    The Student-t fit needs a reasonable sample, so nu is only reported
    when n >= min_n_for_t; the COVID window has about 20 Mondays and a
    three-parameter fit on 20 points would be noise dressed up as a number.
    """
    x = np.asarray(x)
    out = {
        "n": len(x),
        "sigma_ann_%": float(np.std(x) * ANN * 100),
        "daily_var_%2": float(np.var(x) * 1e4),
        "ex_kurt": float(stats.kurtosis(x)),
        "worst_day_%": float(np.min(x) * 100),
    }
    if len(x) >= min_n_for_t:
        try:
            t = fit_student_t(x)
            out["t_nu"], out["se_nu"] = t["nu"], t["se_nu"]
        except Exception:
            out["t_nu"], out["se_nu"] = np.nan, np.nan
    else:
        out["t_nu"], out["se_nu"] = np.nan, np.nan
    return out


def crisis_weekday_table(r):
    """Week-open (Monday) vs midweek vs week-close (Friday) volatility and
    fat tails, inside each of the four crisis windows and over the full
    sample."""
    windows = {"Full sample": (r.index.min(), r.index.max()), **SHOCK_PERIODS}
    rows = []
    for wname, (start, end) in windows.items():
        sub = r.loc[str(start):str(end)]
        for gname, days in WEEK_GROUPS.items():
            x = sub[sub.index.day_name().isin(days)].values
            row = {"window": wname, "group": gname}
            row.update(_group_stats(x))
            rows.append(row)
    return pd.DataFrame(rows).set_index(["window", "group"])


def make_crisis_figure(crisis_df, save_dir):
    """Grouped bars: volatility and excess kurtosis by window and group."""
    windows = list(dict.fromkeys(crisis_df.index.get_level_values(0)))
    groups = list(WEEK_GROUPS)
    colours = {"Monday (week open)": "darkorange",
               "Midweek (Tue-Thu)": "0.6",
               "Friday (week close)": "steelblue"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(windows))
    w = 0.26
    for k, g in enumerate(groups):
        vols = [crisis_df.loc[(win, g), "sigma_ann_%"] for win in windows]
        kurt = [crisis_df.loc[(win, g), "ex_kurt"] for win in windows]
        ax1.bar(x + (k - 1) * w, vols, w, color=colours[g], alpha=0.9, label=g)
        ax2.bar(x + (k - 1) * w, kurt, w, color=colours[g], alpha=0.9, label=g)
    for ax, ylab, title in [(ax1, "Annualised volatility (%)",
                             "Volatility by window and week position"),
                            (ax2, "Excess kurtosis",
                             "Fat tails by window and week position")]:
        ax.set_xticks(x)
        ax.set_xticklabels([w_.replace(" ", "\n", 1) for w_ in windows],
                           fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    ax1.legend(fontsize=8)
    fig.suptitle("Week open (Monday) vs week close (Friday), full sample "
                 "and the four crises")
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "week6_crisis_weekday.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


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

    print("\nWeek open (Monday) vs midweek vs week close (Friday),")
    print("full sample and inside each crisis window:")
    crisis_df = crisis_weekday_table(r)
    print(crisis_df.to_string(float_format=lambda v: f"{v:,.3f}"))

    data_dir = os.path.join(_HERE, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    day_df.to_csv(os.path.join(data_dir, "week6_weekday_params.csv"))
    lr_df.to_csv(os.path.join(data_dir, "week6_weekday_lr.csv"), index=False)
    crisis_df.to_csv(os.path.join(data_dir, "week6_crisis_weekday.csv"))

    make_figure(day_df, os.path.abspath(args.save_dir))
    make_crisis_figure(crisis_df, os.path.abspath(args.save_dir))
    print("\nWeekday analysis complete.")


if __name__ == "__main__":
    main()
