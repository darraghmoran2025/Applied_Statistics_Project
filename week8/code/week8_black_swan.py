"""
week8_black_swan.py: were the four crisis events Black Swans?

Taleb (2007) calls an event a Black Swan when it lies outside the realm of
regular expectation, carries an extreme impact, and is rationalised only
after the fact. The first criterion is the one statistics can touch: an
event is only "outside regular expectation" RELATIVE TO A MODEL of what
regular expectation is. This script therefore asks, for each of the four
crisis windows of the project (dot-com, GFC, COVID-19, 2022 Fed hikes) and
each of the five fitted marginals (Gaussian, Laplace, Student-t, VG, NIG):
how surprising was the event under that model? Four complementary
measurements, one per mode:

  surprise    Exceedance probability and implied return period (in years)
              of each crisis's worst single day under the full-sample MLE
              of every model, plus the Bayesian version: the posterior
              distribution of the return period of the worst day in the
              sample (16 March 2020), using the Week 4 posterior draws.

  oos         The honest version of the same question, using only
              information available BEFORE each crisis: each model is
              refitted on the `--window` trading days preceding the crisis
              window, then scored on the crisis days themselves (worst-day
              return period, mean log predictive density, and the count of
              crisis days the pre-crisis fit priced below one-in-a-
              thousand). The dot-com window is skipped: the sample starts
              in January 2000, so there is no pre-crisis history to fit on.

  extremes    Simulation check of extreme-value consistency: simulate many
              25-year histories (n = len(sample)) from each fitted model
              and ask how often a history contains a day as bad as the
              observed worst day, a 21-day stretch as bad as the observed
              worst, or as many 3-sigma / 5-sigma down days as 2000-2024
              actually delivered.

  clustering  The timing question. Reusing the Week 7 rolling backtest hit
              sequences (99% VaR), compare each model's largest observed
              21-day violation cluster with the distribution of the same
              statistic when the SAME NUMBER of violations is scattered
              independently through the sample (conditional permutation
              test). Heavy tails can absolve the magnitude of a crisis day;
              this tests whether any iid model can absolve the timing.

Run standalone:
    python week8_black_swan.py                  # all four modes
    python week8_black_swan.py --mode surprise
    python week8_black_swan.py --mode extremes --paths 2000
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week2", "code"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "week3", "code"))
from week2_gaussian_student_mle import fit_gaussian, fit_student_t   # noqa: E402
from levy_models import (fit_laplace, fit_vg, fit_nig,               # noqa: E402
                         simulate_vg, simulate_nig,
                         _logpdf_vg, _logpdf_nig,
                         MODEL_COLOURS, SHOCK_PERIODS)

MODELS = ["Gaussian", "Laplace", "Student-t", "VG", "NIG"]
BAYES_MODELS = ["Gaussian", "Laplace", "Student-t", "NIG"]   # Week 4 posteriors
TRADING_DAYS = 252


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
# MODEL INTERFACE: fit, log-pdf, lower-tail probability, simulate
# ════════════════════════════════════════════════════════════════════════════
# Convention as everywhere in the project: losses are negative returns, so
# the lower tail P(X <= x) is the probability that matters.

def fit_model(name, r):
    return {
        "Gaussian":  fit_gaussian,
        "Laplace":   fit_laplace,
        "Student-t": fit_student_t,
        "VG":        fit_vg,
        "NIG":       fit_nig,
    }[name](r)


def logpdf(name, fit, x):
    x = np.asarray(x, dtype=float)
    if name == "Gaussian":
        return stats.norm.logpdf(x, loc=fit["mu"], scale=fit["sigma"])
    if name == "Laplace":
        return stats.laplace.logpdf(x, loc=fit["mu"], scale=fit["b"])
    if name == "Student-t":
        return stats.t.logpdf(x, df=fit["nu"], loc=fit["mu"], scale=fit["sigma"])
    if name == "VG":
        return _logpdf_vg(x, fit["mu"], fit["sigma"], fit["theta"], fit["nu"])
    if name == "NIG":
        return _logpdf_nig(x, fit["mu"], fit["alpha"], fit["beta"], fit["delta"])
    raise ValueError(name)


def lower_tail_p(name, fit, x):
    """
    P(X <= x) under the fitted model. Closed form where one exists;
    for VG and NIG the density is integrated over (-inf, x], which quad
    handles through its infinite-interval transformation. exp(logcdf) is
    used for the Gaussian because the COVID day sits ~10 sigma out, where
    the direct cdf would be computed from a difference of near-equal terms.
    """
    if name == "Gaussian":
        return float(np.exp(stats.norm.logcdf(x, loc=fit["mu"], scale=fit["sigma"])))
    if name == "Laplace":
        # x is always far below the median here, so only the lower branch
        return float(0.5 * np.exp((x - fit["mu"]) / fit["b"]))
    if name == "Student-t":
        return float(stats.t.cdf(x, df=fit["nu"], loc=fit["mu"], scale=fit["sigma"]))
    if name in ("VG", "NIG"):
        pdf = lambda t: float(np.exp(logpdf(name, fit, t)))
        p, _ = quad(pdf, -np.inf, x, limit=300)
        return float(p)
    raise ValueError(name)


def simulate(name, fit, n, seed):
    rng = np.random.default_rng(seed)
    if name == "Gaussian":
        return rng.normal(fit["mu"], fit["sigma"], size=n)
    if name == "Laplace":
        return rng.laplace(fit["mu"], fit["b"], size=n)
    if name == "Student-t":
        return fit["mu"] + fit["sigma"] * rng.standard_t(fit["nu"], size=n)
    if name == "VG":
        return simulate_vg(fit, n=n, seed=seed)
    if name == "NIG":
        return simulate_nig(fit, n=n, seed=seed)
    raise ValueError(name)


def return_period_years(p):
    """Expected years between days at least this bad under the model."""
    if p <= 0.0:
        return np.inf
    return 1.0 / (TRADING_DAYS * p)


# ════════════════════════════════════════════════════════════════════════════
# CRISIS DESCRIPTIVES
# ════════════════════════════════════════════════════════════════════════════

def crisis_stats(r):
    """Worst 1-day, 5-day and 21-day log-returns inside each shock window."""
    rows = []
    for label, (start, end) in SHOCK_PERIODS.items():
        w = r.loc[start:end]
        worst_day = w.idxmin()
        rows.append({
            "crisis": label, "start": start, "end": end, "n_days": len(w),
            "worst_day": worst_day.date(), "worst_1d_%": 100 * w.min(),
            "worst_5d_%": 100 * w.rolling(5).sum().min(),
            "worst_21d_%": 100 * w.rolling(21).sum().min(),
            "window_total_%": 100 * w.sum(),
        })
    return pd.DataFrame(rows).set_index("crisis")


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: SURPRISE — full-sample exceedance probabilities and return periods
# ════════════════════════════════════════════════════════════════════════════

def surprise_table(r, fits):
    """Return period of each crisis's worst day under each full-sample fit."""
    rows = []
    for label, (start, end) in SHOCK_PERIODS.items():
        w = r.loc[start:end]
        x = float(w.min())
        row = {"crisis": label, "worst_day": w.idxmin().date(),
               "worst_1d_%": 100 * x}
        for name in MODELS:
            p = lower_tail_p(name, fits[name], x)
            row[f"p_{name}"] = p
            row[f"RP_years_{name}"] = return_period_years(p)
        rows.append(row)
    return pd.DataFrame(rows).set_index("crisis")


def posterior_return_periods(r, data_dir, n_draws=2000, seed=42):
    """
    Posterior distribution of the return period of the worst day in the
    sample under each Week 4 Bayesian model. Parameter uncertainty widens
    or narrows the verdict; the point here is that even across the whole
    posterior the model-to-model gap dwarfs the within-model uncertainty.
    """
    x = float(r.min())
    rng = np.random.default_rng(seed)
    out = {}
    for name in BAYES_MODELS:
        fname = {"Gaussian": "week4_posterior_gaussian.csv",
                 "Laplace": "week4_posterior_laplace.csv",
                 "Student-t": "week4_posterior_student_t.csv",
                 "NIG": "week4_posterior_nig.csv"}[name]
        path = os.path.join(_HERE, "..", "..", "week4", "data", fname)
        if not os.path.exists(path):
            print(f"  [posterior] {path} not found; skipping {name}")
            continue
        d = pd.read_csv(path)
        take = rng.choice(len(d), size=min(n_draws, len(d)), replace=False)
        d = d.iloc[take]
        if name == "Gaussian":
            p = np.exp(stats.norm.logcdf(x, loc=d["mu"], scale=d["sigma"]))
        elif name == "Laplace":
            p = 0.5 * np.exp((x - d["mu"]) / d["b"])
        elif name == "Student-t":
            p = stats.t.cdf(x, df=d["nu"], loc=d["mu"], scale=d["sigma"])
        else:  # NIG: scipy parameterisation a = alpha*delta, b = beta*delta
            p = stats.norminvgauss.cdf(
                x, a=d["alpha"] * d["delta"], b=d["beta"] * d["delta"],
                loc=d["mu"], scale=d["delta"])
        rp = 1.0 / (TRADING_DAYS * np.maximum(np.asarray(p, dtype=float), 1e-300))
        out[name] = np.log10(rp)
        print(f"  [posterior] {name}: median RP 10^{np.median(out[name]):.2f} years")
    return x, out


def plot_return_periods(surp, save_dir):
    """Grouped bars, log scale: return period of each crisis's worst day."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    crises = list(surp.index)
    width = 0.15
    xs = np.arange(len(crises))
    for i, name in enumerate(MODELS):
        vals = surp[f"RP_years_{name}"].values
        ax.bar(xs + (i - 2) * width, vals, width,
               color=MODEL_COLOURS[name], label=name)
    ax.set_yscale("log")
    ax.axhline(25, color="black", ls="--", lw=1.0)
    ax.text(len(crises) - 0.42, 32, "sample length (25 y)", fontsize=8)
    ax.axhline(1.38e10, color="black", ls=":", lw=1.0)
    ax.text(len(crises) - 0.42, 2.2e10, "age of the universe", fontsize=8)
    labels = [f"{c}\n({surp.loc[c, 'worst_day']}, "
              f"{surp.loc[c, 'worst_1d_%']:.1f}%)" for c in crises]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("implied return period of the worst day (years, log scale)")
    ax.set_title("How rare was each crisis's worst day under each fitted model?")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(save_dir, "week8_return_periods.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_posterior_rp(log10_rp, worst_x, save_dir):
    """Posterior densities of log10 return period for the worst sample day."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, v in log10_rp.items():
        v = v[np.isfinite(v)]
        grid = np.linspace(v.min() - 0.5, v.max() + 0.5, 400)
        kde = stats.gaussian_kde(v)
        ax.plot(grid, kde(grid), color=MODEL_COLOURS[name], lw=1.6, label=name)
        ax.fill_between(grid, kde(grid), color=MODEL_COLOURS[name], alpha=0.15)
    ax.axvline(np.log10(25), color="black", ls="--", lw=1.0)
    ax.text(np.log10(25) + 0.15, ax.get_ylim()[1] * 0.9,
            "once per sample (25 y)", fontsize=8)
    ax.set_xlabel(r"$\log_{10}$ return period in years of the worst sample day "
                  f"({100 * worst_x:.1f}%)")
    ax.set_ylabel("posterior density")
    ax.set_title("Posterior return period of 16 March 2020 under each model")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(save_dir, "week8_posterior_return_period.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# MODE 2: OUT-OF-SAMPLE SURPRISE — pre-crisis fits only
# ════════════════════════════════════════════════════════════════════════════

def oos_table(r, window=500):
    """
    For each crisis with at least `window` prior observations: refit every
    model on the `window` days before the crisis window opens, then score
    the crisis days under that frozen pre-crisis fit.
    """
    rows = []
    fits_by_crisis = {}
    for label, (start, end) in SHOCK_PERIODS.items():
        pre = r.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)]
        if len(pre) < window:
            print(f"  [oos] {label}: only {len(pre)} pre-crisis days "
                  f"(< {window}), skipped")
            continue
        train = pre.iloc[-window:]
        w = r.loc[start:end]
        x = float(w.min())
        fits = {name: fit_model(name, train.values) for name in MODELS}
        fits_by_crisis[label] = fits
        for name in MODELS:
            p_worst = lower_tail_p(name, fits[name], x)
            pit = np.array([lower_tail_p(name, fits[name], v) for v in w.values])
            rows.append({
                "crisis": label, "model": name,
                "train_end": train.index[-1].date(),
                "worst_1d_%": 100 * x,
                "p_worst": p_worst,
                "RP_years": return_period_years(p_worst),
                "mean_logscore": float(np.mean(logpdf(name, fits[name], w.values))),
                "n_days_p<0.001": int((pit < 1e-3).sum()),
                "n_crisis_days": len(w),
            })
    return pd.DataFrame(rows), fits_by_crisis


def plot_oos_vs_full(surp, oos, save_dir):
    """Return period of each crisis's worst day: full-sample vs pre-crisis."""
    crises = [c for c in SHOCK_PERIODS if c in set(oos["crisis"])]
    fig, axes = plt.subplots(1, len(crises), figsize=(4.2 * len(crises), 5),
                             sharey=True)
    if len(crises) == 1:
        axes = [axes]
    for ax, label in zip(axes, crises):
        for i, name in enumerate(MODELS):
            full = surp.loc[label, f"RP_years_{name}"]
            pre = float(oos.loc[(oos["crisis"] == label)
                                & (oos["model"] == name), "RP_years"].iloc[0])
            ax.plot([i, i], [full, pre], color="0.7", lw=1.0, zorder=1)
            ax.plot(i, full, "o", color=MODEL_COLOURS[name], ms=8, zorder=2)
            ax.plot(i, pre, "^", color=MODEL_COLOURS[name], ms=8, zorder=2)
        ax.set_yscale("log")
        ax.axhline(25, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{label}\nworst day "
                     f"{surp.loc[label, 'worst_1d_%']:.1f}%", fontsize=10)
    axes[0].set_ylabel("return period of the worst day (years, log scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color="black",
                          label="full-sample fit"),
               plt.Line2D([], [], marker="^", ls="", color="black",
                          label="pre-crisis fit"),
               plt.Line2D([], [], ls="--", color="black",
                          label="once per sample (25 y)")]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=8,
                    framealpha=0.9)
    fig.suptitle("Surprise with hindsight vs surprise at the time", y=0.99)
    fig.tight_layout()
    out = os.path.join(save_dir, "week8_oos_surprise.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# MODE 3: EXTREME-VALUE CONSISTENCY BY SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_extremes(r, fits, n_paths=4000, seed=42, chunk=250):
    """
    Simulate n_paths iid histories of length len(r) from each fitted model
    and record, per history: the worst day, the worst 21-day sum, and the
    number of days below -3 and -5 empirical standard deviations. Compare
    the observed history's values with each simulated distribution.
    """
    n = len(r)
    sd = float(r.std(ddof=0))
    obs = {
        "worst_1d": float(r.min()),
        "worst_21d": float(r.rolling(21).sum().min()),
        "n_3sd": int((r < -3 * sd).sum()),
        "n_5sd": int((r < -5 * sd).sum()),
    }
    print(f"  observed: worst day {100 * obs['worst_1d']:.2f}%, "
          f"worst 21d {100 * obs['worst_21d']:.2f}%, "
          f"{obs['n_3sd']} days < -3sd, {obs['n_5sd']} days < -5sd")

    kernel = np.ones(21)
    sims, rows = {}, []
    for name in MODELS:
        w1 = np.empty(n_paths)
        w21 = np.empty(n_paths)
        c3 = np.empty(n_paths, dtype=int)
        c5 = np.empty(n_paths, dtype=int)
        done = 0
        block = 0
        while done < n_paths:
            k = min(chunk, n_paths - done)
            x = simulate(name, fits[name], k * n,
                         seed=seed + 1000 * block + MODELS.index(name))
            x = x.reshape(k, n)
            w1[done:done + k] = x.min(axis=1)
            for i in range(k):
                w21[done + i] = np.convolve(x[i], kernel, mode="valid").min()
            c3[done:done + k] = (x < -3 * sd).sum(axis=1)
            c5[done:done + k] = (x < -5 * sd).sum(axis=1)
            done += k
            block += 1
        sims[name] = {"worst_1d": w1, "worst_21d": w21}
        rows.append({
            "model": name,
            "median_sim_worst_1d_%": 100 * float(np.median(w1)),
            "P(worst day <= observed)": float((w1 <= obs["worst_1d"]).mean()),
            "median_sim_worst_21d_%": 100 * float(np.median(w21)),
            "P(worst 21d <= observed)": float((w21 <= obs["worst_21d"]).mean()),
            "median_sim_n_3sd": float(np.median(c3)),
            "P(n_3sd >= observed)": float((c3 >= obs["n_3sd"]).mean()),
            "median_sim_n_5sd": float(np.median(c5)),
            "P(n_5sd >= observed)": float((c5 >= obs["n_5sd"]).mean()),
        })
        print(f"  {name}: P(day as bad) = {rows[-1]['P(worst day <= observed)']:.4f}, "
              f"P(21d as bad) = {rows[-1]['P(worst 21d <= observed)']:.4f}")
    return pd.DataFrame(rows).set_index("model"), sims, obs


def plot_extremes(sims, obs, n_paths, save_dir):
    """Simulated worst-day and worst-21-day distributions vs history."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, obs_key, title in (
        (axes[0], "worst_1d", "worst_1d", "worst single day in 25 years"),
        (axes[1], "worst_21d", "worst_21d", "worst 21-day stretch in 25 years"),
    ):
        for name in MODELS:
            v = 100 * sims[name][key]
            lo = np.percentile(v, 0.5)
            grid = np.linspace(max(lo, -60), np.percentile(v, 99.5), 400)
            kde = stats.gaussian_kde(v)
            ax.plot(grid, kde(grid), color=MODEL_COLOURS[name], lw=1.5,
                    label=name)
        ax.axvline(100 * obs[obs_key], color="black", lw=1.4, ls="--")
        ax.text(100 * obs[obs_key], ax.get_ylim()[1] * 0.95,
                " observed", fontsize=8, va="top")
        ax.set_title(title)
        ax.set_xlabel("log-return (%)")
        ax.set_ylabel("density across simulated histories")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Does a 25-year history like ours fit inside each model? "
                 f"({n_paths:,} simulated histories per model)", y=0.99)
    fig.tight_layout()
    out = os.path.join(save_dir, "week8_extreme_sim.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# MODE 4: CLUSTERING — is the timing the real Black Swan?
# ════════════════════════════════════════════════════════════════════════════

def clustering_test(var_csv=os.path.join(_HERE, "..", "..", "week7", "data",
                                         "week7_var_series.csv"),
                    n_perm=20000, seed=42):
    """
    Conditional permutation test on the Week 7 hit sequences (99% VaR).
    Holding each model's TOTAL number of violations fixed, scatter them
    uniformly through the 5,787 backtest days and record the largest number
    landing in any 21-day window. The p-value is the share of permutations
    whose largest cluster reaches the observed one. This isolates timing
    from magnitude: the marginal (how many hits) is taken as given.
    """
    df = pd.read_csv(var_csv, index_col=0, parse_dates=True)
    rng = np.random.default_rng(seed)
    kernel = np.ones(21)
    models = [c.replace("hit990_", "") for c in df.columns
              if c.startswith("hit990_")]
    rows, perm_dists = [], {}
    for name in models:
        hits = df[f"hit990_{name}"].values.astype(int)
        obs_max = int(np.convolve(hits, kernel, mode="valid").max())
        when = df.index[np.argmax(np.convolve(hits, kernel, mode="valid"))]
        maxima = np.empty(n_perm, dtype=int)
        for b in range(n_perm):
            perm = rng.permutation(hits)
            maxima[b] = np.convolve(perm, kernel, mode="valid").max()
        pval = float((maxima >= obs_max).mean())
        perm_dists[name] = maxima
        rows.append({
            "model": name, "hits_990": int(hits.sum()),
            "obs_max_21d_cluster": obs_max,
            "cluster_window_start": when.date(),
            "iid_median_max": float(np.median(maxima)),
            "iid_q99_max": float(np.percentile(maxima, 99)),
            "p_value": pval,
        })
        print(f"  {name}: {hits.sum()} hits, biggest 21-day cluster {obs_max} "
              f"(around {when.date()}), iid median {np.median(maxima):.0f}, "
              f"p = {pval:.5f}")
    return pd.DataFrame(rows).set_index("model"), perm_dists


def plot_clustering(cl, perm_dists, save_dir):
    """Permutation distribution of the largest 21-day cluster vs observed."""
    models = list(cl.index)
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 4.2),
                             sharey=True)
    for ax, name in zip(np.atleast_1d(axes), models):
        maxima = perm_dists[name]
        top = max(int(maxima.max()), int(cl.loc[name, "obs_max_21d_cluster"])) + 2
        bins = np.arange(0.5, top + 0.5)
        ax.hist(maxima, bins=bins, density=True,
                color=MODEL_COLOURS.get(name, "grey"), alpha=0.65)
        obs = cl.loc[name, "obs_max_21d_cluster"]
        ax.axvline(obs, color="black", lw=1.5, ls="--")
        ax.text(obs - 0.4, ax.get_ylim()[1] * 0.55, f"observed = {obs}",
                fontsize=8, rotation=90, va="bottom")
        ax.set_title(f"{name}\np = {cl.loc[name, 'p_value']:.4f}", fontsize=10)
        ax.set_xlabel("largest 21-day cluster")
    np.atleast_1d(axes)[0].set_ylabel("permutation density")
    fig.suptitle("Largest 21-day cluster of 99% VaR violations: "
                 "observed vs the same hits scattered independently", y=0.99)
    fig.tight_layout()
    out = os.path.join(save_dir, "week8_clustering.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Week 8 Black Swan analysis")
    ap.add_argument("--mode", default="all",
                    choices=["all", "surprise", "oos", "extremes", "clustering"])
    ap.add_argument("--window", type=int, default=500,
                    help="pre-crisis estimation window (oos mode)")
    ap.add_argument("--paths", type=int, default=4000,
                    help="simulated 25-year histories per model (extremes)")
    ap.add_argument("--perms", type=int, default=20000,
                    help="permutations for the clustering test")
    ap.add_argument("--posterior_draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    ap.add_argument("--data_dir", default=os.path.join(_HERE, "..", "data"))
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    r = load_returns()

    stats_df = crisis_stats(r)
    stats_df.to_csv(os.path.join(args.data_dir, "week8_crisis_stats.csv"))
    print("\nCrisis descriptives")
    print(stats_df.round(2).to_string())

    fits = None
    surp = None
    if args.mode in ("all", "surprise", "oos", "extremes"):
        print("\nFitting the five models on the full sample...")
        fits = {name: fit_model(name, r.values) for name in MODELS}
        for name in MODELS:
            print(f"  {name}: loglik {fits[name]['loglik']:.1f}")

    if args.mode in ("all", "surprise"):
        print("\n=== MODE 1: full-sample surprise ===")
        surp = surprise_table(r, fits)
        surp.to_csv(os.path.join(args.data_dir, "week8_surprise_table.csv"))
        show = surp[[c for c in surp.columns if not c.startswith("p_")]]
        print(show.to_string(float_format=lambda v: f"{v:,.3g}"))
        plot_return_periods(surp, args.save_dir)

        worst_x, log10_rp = posterior_return_periods(
            r, args.data_dir, n_draws=args.posterior_draws, seed=args.seed)
        if log10_rp:
            pd.DataFrame({k: pd.Series(v).describe(
                percentiles=[0.03, 0.5, 0.97]) for k, v in log10_rp.items()}
            ).to_csv(os.path.join(args.data_dir, "week8_posterior_rp.csv"))
            plot_posterior_rp(log10_rp, worst_x, args.save_dir)

    if args.mode in ("all", "oos"):
        print(f"\n=== MODE 2: out-of-sample surprise (window {args.window}) ===")
        oos, _ = oos_table(r, window=args.window)
        oos.to_csv(os.path.join(args.data_dir, "week8_oos_table.csv"),
                   index=False)
        print(oos.drop(columns=["p_worst"]).to_string(
            index=False, float_format=lambda v: f"{v:,.3g}"))
        if surp is None:
            surp = surprise_table(r, fits)
        plot_oos_vs_full(surp, oos, args.save_dir)

    if args.mode in ("all", "extremes"):
        print(f"\n=== MODE 3: extreme-value simulation ({args.paths} paths) ===")
        ext, sims, obs = simulate_extremes(r, fits, n_paths=args.paths,
                                           seed=args.seed)
        ext.to_csv(os.path.join(args.data_dir, "week8_extremes_table.csv"))
        print(ext.round(4).to_string())
        plot_extremes(sims, obs, args.paths, args.save_dir)

    if args.mode in ("all", "clustering"):
        print(f"\n=== MODE 4: violation clustering ({args.perms} permutations) ===")
        cl, perm_dists = clustering_test(n_perm=args.perms, seed=args.seed)
        cl.to_csv(os.path.join(args.data_dir, "week8_clustering.csv"))
        print(cl.round(4).to_string())
        plot_clustering(cl, perm_dists, args.save_dir)


if __name__ == "__main__":
    main()
