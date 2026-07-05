"""
week6_open_close.py: market-open vs market-closed volatility, and
week-on-week volatility.

Splits the close-to-close return into the part earned while the market
is closed (overnight: prior close to today's open) and the part earned
while it is open (intraday: open to close):

    r_cc = ln(C_t / C_{t-1}) = r_overnight + r_intraday
    r_overnight = ln(O_t / C_{t-1}),   r_intraday = ln(C_t / O_t)

DATA CAVEAT: Yahoo's ^GSPC open prices are stale for most of the early
sample (Open set equal to the prior close on 96% of 2000-2004 days, 31%
of 2005-2009, 12% of 2010-2014, near zero from 2015). The overnight and
intraday split therefore runs on 2015-2024 by default, where the opens
are real. The weekly volatility section uses the full 2000-2024 sample,
which needs closes only.

Run standalone:
    python week6_open_close.py
    python week6_open_close.py --split_start 2010-01-01
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week2", "code"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t  # noqa: E402
from levy_models import fit_laplace, SHOCK_PERIODS  # noqa: E402

ANN = np.sqrt(252.0)
ANN_W = np.sqrt(52.0)

SHADE_COLOURS = {
    "Dot-com crash":  "#f4b400",
    "GFC":            "#e2504e",
    "COVID-19":       "#1f9ec4",
    "Fed rate hikes": "#3fa34d",
}


def load_ohlc(path=os.path.join(_HERE, "..", "data", "sp500_ohlc.csv")):
    if not os.path.isfile(path):
        import yfinance as yf
        print("No OHLC cache; downloading ^GSPC ...")
        df = yf.download("^GSPC", start="1999-12-20", end="2024-12-31",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded OHLC: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    return df


def stale_open_audit(ohlc, threshold=1e-8):
    """Measure the fake-open problem that motivates the 2015 cutoff.

    A day is flagged stale if Open(t) equals Close(t-1), i.e. the vendor
    filled the open with the prior close (|ln(O_t / C_{t-1})| < threshold).
    Prints the stale fraction by five-year block and returns the series.
    """
    r_on = np.log(ohlc["Open"] / ohlc["Close"].shift(1)).dropna()
    stale = np.abs(r_on) < threshold
    blocks = [("2000-2004", "2000", "2004"), ("2005-2009", "2005", "2009"),
              ("2010-2014", "2010", "2014"), ("2015-2024", "2015", "2024")]
    print("\nStale-open audit (Open(t) = Close(t-1)):")
    for label, a, b in blocks:
        s = stale.loc[a:b]
        print(f"  {label}: {s.mean():6.2%} of {len(s)} days")
    return stale


def build_components(ohlc, split_start):
    """Overnight and intraday return components on the clean-opens sample."""
    c_prev = ohlc["Close"].shift(1)
    r_on = np.log(ohlc["Open"] / c_prev)
    r_id = np.log(ohlc["Close"] / ohlc["Open"])
    r_cc = np.log(ohlc["Close"] / c_prev)
    df = pd.DataFrame({"overnight": r_on, "intraday": r_id, "close_close": r_cc}).dropna()
    df = df.loc[split_start:]
    stale = float(np.mean(np.abs(df["overnight"]) < 1e-8))
    print(f"Split sample {df.index.min().date()} to {df.index.max().date()}: "
          f"n = {len(df)}, stale-open fraction = {stale:.2%}")
    return df


def component_table(df):
    """Distributional summary of each return component."""
    rows = []
    for name in ["close_close", "overnight", "intraday"]:
        x = df[name].values
        g = fit_gaussian(x)
        lap = fit_laplace(x)
        try:
            t = fit_student_t(x)
            nu, se_nu = t["nu"], t["se_nu"]
        except Exception:
            nu, se_nu = np.nan, np.nan
        rows.append({
            "component": name, "n": len(x),
            "mean_bp": np.mean(x) * 1e4,
            "sigma_ann_%": g["sigma"] * ANN * 100,
            "var_share_%": 100 * np.var(x) / np.var(df["close_close"].values),
            "t_nu": nu, "se_nu": se_nu,
            "laplace_b_bp": lap["b"] * 1e4,
            "ex_kurt": stats.kurtosis(x),
            "skew": stats.skew(x),
        })
    out = pd.DataFrame(rows).set_index("component")
    corr = float(np.corrcoef(df["overnight"], df["intraday"])[0, 1])
    out.attrs["corr_on_id"] = corr
    return out


def weekly_vol(r_cc_full):
    """Weekly realised volatility from daily close-to-close returns, plus
    week-on-week persistence and the weekly-return Student-t fit
    (aggregational Gaussianity check against the daily nu of 2.648)."""
    wk_var = (r_cc_full**2).resample("W-FRI").sum()
    wk_n = r_cc_full.resample("W-FRI").count()
    wk_var = wk_var[wk_n >= 3]
    vol = np.sqrt(wk_var * 252.0 / wk_n[wk_n >= 3]) * 100      # annualised %
    vol.name = "weekly_vol"

    v1, v0 = vol.values[1:], vol.values[:-1]
    slope, intercept = np.polyfit(v0, v1, 1)
    r2 = np.corrcoef(v0, v1)[0, 1] ** 2
    print(f"\nWeek-on-week volatility persistence: vol_w = {intercept:.2f} + "
          f"{slope:.3f} * vol_(w-1),  R^2 = {r2:.3f},  n = {len(v1)} weeks")

    wk_ret = r_cc_full.resample("W-FRI").sum()
    wk_ret = wk_ret[wk_n >= 3]
    t_w = fit_student_t(wk_ret.values)
    print(f"Weekly-return Student-t: nu = {t_w['nu']:.2f} (SE {t_w['se_nu']:.2f}) "
          f"vs daily nu = 2.648; aggregation lightens the tail")
    return vol, (slope, intercept, r2), t_w


def make_figures(df, comp, vol, ar, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Overnight vs intraday densities, with a log-scale tail panel
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = np.linspace(-0.04, 0.04, 400)
    for name, colour in [("overnight", "darkorange"), ("intraday", "steelblue")]:
        kde = stats.gaussian_kde(df[name].values)
        ax.plot(grid, kde(grid), color=colour, lw=1.8, label=name)
        axz.semilogy(grid, kde(grid) + 1e-4, color=colour, lw=1.8)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title("Overnight vs intraday return densities (2015-2024)")
    ax.legend()
    axz.set_xlabel("Return")
    axz.set_ylabel("Density (log)")
    axz.set_title("Same densities, log scale")
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_open_close_density.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # 2. Volatility by weekday for each component (weekend gap sits in
    #    Monday's overnight bar)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    width = 0.35
    x = np.arange(len(days))
    for k, (name, colour) in enumerate([("overnight", "darkorange"),
                                        ("intraday", "steelblue")]):
        vols = [df[df.index.day_name() == d][name].std() * ANN * 100 for d in days]
        ax.bar(x + (k - 0.5) * width, vols, width, color=colour, alpha=0.85,
               label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:3] for d in days])
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Volatility by weekday and session (2015-2024). "
                 "Monday overnight includes the weekend gap.")
    ax.legend()
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_open_close_weekday.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # 3. Weekly realised volatility, 2000-2024, with shocks shaded and the
    #    week-on-week persistence scatter
    slope, intercept, r2 = ar
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.5),
                                   gridspec_kw={"width_ratios": [2.4, 1.0]})
    for name, (start, end) in SHOCK_PERIODS.items():
        axL.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    color=SHADE_COLOURS[name], alpha=0.30, lw=0, zorder=0)
    axL.plot(vol.index, vol.values, color="black", lw=0.7, zorder=2)
    axL.set_ylabel("Weekly realised vol (annualised %)")
    axL.set_title("Week-on-week realised volatility, 2000-2024")
    handles = [mpatches.Patch(facecolor=SHADE_COLOURS[n], alpha=0.5, label=n)
               for n in SHOCK_PERIODS]
    axL.legend(handles=handles, fontsize=8, ncol=2)
    axL.margins(x=0.01)

    axR.scatter(vol.values[:-1], vol.values[1:], s=6, alpha=0.3,
                color="steelblue", edgecolors="none")
    xs = np.array([vol.min(), vol.max()])
    axR.plot(xs, intercept + slope * xs, color="black", lw=1.3,
             label=f"slope {slope:.2f}, R^2 {r2:.2f}")
    axR.set_xlabel("vol, week w-1 (%)")
    axR.set_ylabel("vol, week w (%)")
    axR.set_title("Week-on-week persistence")
    axR.legend(fontsize=8)
    axR.grid(color="0.92", lw=0.6, zorder=0)
    fig.tight_layout()
    out = os.path.join(save_dir, "week6_weekly_vol.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Week 6: open/close and weekly vol")
    ap.add_argument("--split_start", default="2015-01-01",
                    help="start of the clean-opens sample for the split")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    args = ap.parse_args()

    ohlc = load_ohlc()
    stale_open_audit(ohlc)
    df = build_components(ohlc, args.split_start)

    print("\nReturn components (2015-2024 unless overridden):")
    comp = component_table(df)
    print(comp.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"corr(overnight, intraday) = {comp.attrs['corr_on_id']:+.3f}")

    r_cc_full = np.log(ohlc["Close"] / ohlc["Close"].shift(1)).dropna()
    r_cc_full = r_cc_full.loc["2000-01-01":]
    vol, ar, t_w = weekly_vol(r_cc_full)

    data_dir = os.path.join(_HERE, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    comp.to_csv(os.path.join(data_dir, "week6_components.csv"))
    vol.to_csv(os.path.join(data_dir, "week6_weekly_vol.csv"))

    make_figures(df, comp, vol, ar, os.path.abspath(args.save_dir))
    print("\nOpen/close analysis complete.")


if __name__ == "__main__":
    main()
