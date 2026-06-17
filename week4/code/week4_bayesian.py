"""
week4_bayesian.py — Bayesian estimation via NUTS (PyMC)
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Week 4 takes the marginals from Week 2/3 into a full Bayesian setting:
weakly-informative priors anchored on the Week 3 MLE point estimates, a prior
predictive check to confirm the priors imply plausible daily returns, then
NUTS sampling and convergence diagnostics.

Four models: Gaussian, Laplace, Student-t, NIG.  The Laplace stands in for the
Variance-Gamma family — it is the symmetric VG special case (θ = 0, ν = 1) and
has a closed-form density, so it needs no Bessel function.  NIG keeps its full
four-parameter density, implemented as a custom PyTensor log-density using the
modified Bessel function K₁ (pt.kve), with NUTS gradients flowing through it.

The prior predictive check is the decision step.  Rather than justifying the
priors by argument alone, we simulate returns from the prior and confirm they
look like real financial returns (mostly within a few percent, occasional
larger moves) instead of physically absurd ones (±50% routinely).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORS  (anchored on the Week 3 full-sample MLE)
────────
Gaussian N(μ, σ)
  μ      ~ Normal(0, 1e-3)      MLE +0.000223; drift is tiny and uncertain
  σ      ~ HalfNormal(0.02)     MLE 0.012234; positive, daily-vol scale

Laplace  (symmetric VG: θ = 0, ν = 1)
  μ      ~ Normal(0, 1e-3)      MLE +0.000602
  b      ~ HalfNormal(0.02)     MLE 0.008078 (Laplace scale)

Student-t  t(ν, μ, σ)
  μ      ~ Normal(0, 1e-3)      MLE +0.000656
  σ      ~ HalfNormal(0.02)     MLE 0.007077 (scale, not std)
  ν      ~ Gamma(2, 0.1)        weakly informative, mean 20, allows the data
                                to pull ν down toward its MLE of 2.648.
                                A shifted prior ν = 2 + Gamma(...) would force
                                finite variance (ν > 2); see --nu_floor.

NIG  (μ, α, β, δ)               full four-parameter Lévy density
  μ      ~ Normal(0, 1e-3)      MLE +0.001111
  γ      ~ Gamma(4, 4/52)       mean 52, sd 26; γ = √(α²−β²) (tail decay)
  β      ~ Normal(0, 15)        MLE −6.095 (asymmetry)
  δ      ~ HalfNormal(0.02)     MLE 0.007574 (scale)
  α      = √(γ² + β²)           derived, so α > |β| holds automatically.
                                MLE 52.341.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTIONS
─────────
load_returns(use_cache, cache_path)         → pd.Series
    Daily S&P 500 log-returns.  Caches to CSV after the first Yahoo pull so
    repeated sampling runs are reproducible and offline.

build_gaussian_model(r)                      → pm.Model
build_student_t_model(r, nu_floor)           → pm.Model
    PyMC models with the priors above.  nu_floor=2.0 switches the Student-t
    to the shifted ν = 2 + Gamma prior (finite-variance constraint).

prior_predictive_check(model, r, name, save_dir, draws) → dict
    Samples the prior predictive, reports the fraction of simulated daily
    returns beyond plausible bounds, and plots the prior-predictive return
    spread against the observed range.

run_nuts(model, draws, tune, chains, seed)   → az.InferenceData
summarise(idata, mle, name)                  → pd.DataFrame
    Posterior mean / sd / 94% HDI / R-hat / ESS, with the Week 3 MLE for
    side-by-side validation.

plot_trace(idata, name, save_dir)
    Trace + marginal posterior plots (the basic convergence eyeball; deeper
    diagnostics are Week 5).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run standalone:
    python week4_bayesian.py                       # both models, cached data
    python week4_bayesian.py --model student_t
    python week4_bayesian.py --nu_floor 2.0         # finite-variance Student-t
    python week4_bayesian.py --prior_only           # stop after prior checks
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats as _sps

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── data loading (with cache) ─────────────────────────────────────────────────

def _find_week2():
    """Locate week2_gaussian_student_mle.py for the shared fetch_returns."""
    candidates = [
        os.path.join(_HERE, "..", "..", "week2", "code"),  # repo_clone layout
        os.path.join(_HERE, "..", "Week2"),                # top-level layout
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "week2_gaussian_student_mle.py")):
            return c
    return None


def load_returns(use_cache=True,
                 cache_path=os.path.join(_HERE, "..", "data", "sp500_returns.csv")):
    """
    Daily S&P 500 log-returns (Jan 2000 – Dec 2024) as a dated pd.Series.

    Loads from the local CSV cache when present; otherwise pulls once from
    Yahoo via the Week 2 fetch_returns and writes the cache.  Caching keeps
    the Bayesian runs reproducible and avoids re-downloading on every call.
    """
    cache_path = os.path.abspath(cache_path)
    if use_cache and os.path.isfile(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = "logret"
        print(f"Loaded {len(s)} returns from cache: {cache_path}")
        return s

    w2 = _find_week2()
    if w2 is None:
        raise FileNotFoundError(
            "No cached returns and week2_gaussian_student_mle.py not found. "
            "Run a Week 2/3 script first, or place the CSV at " + cache_path
        )
    sys.path.insert(0, w2)
    from week2_gaussian_student_mle import fetch_returns

    print("Fetching returns from Yahoo Finance ...")
    s = fetch_returns()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    s.to_csv(cache_path)
    print(f"Cached {len(s)} returns -> {cache_path}")
    return s


# ── Week 3 MLE anchors (full sample), for prior-setting and validation ────────

# Display names for figure titles (proper capitalisation of the model labels)
DISPLAY = {
    "gaussian":  "Gaussian",
    "laplace":   "Laplace",
    "student_t": "Student-t",
    "nig":       "NIG",
}

MLE = {
    "gaussian":  {"mu": 0.000223, "sigma": 0.012234},
    "laplace":   {"mu": 0.000602, "b": 0.008078},
    "student_t": {"mu": 0.000656, "sigma": 0.007077, "nu": 2.648},
    "nig":       {"mu": 0.001111, "alpha": 52.341, "beta": -6.095, "delta": 0.007574},
}


# ── models ────────────────────────────────────────────────────────────────────

def build_gaussian_model(r):
    with pm.Model() as model:
        mu    = pm.Normal("mu", mu=0.0, sigma=1e-3)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        pm.Normal("returns", mu=mu, sigma=sigma, observed=r)
    return model


def build_laplace_model(r):
    """Laplace = symmetric VG special case (θ = 0, ν = 1). Built-in density."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1e-3)
        b  = pm.HalfNormal("b", sigma=0.02)
        pm.Laplace("returns", mu=mu, b=b, observed=r)
    return model


def _nig_logp(value, mu, alpha, beta, delta):
    """
    NIG(μ, α, β, δ) log-density in PyTensor (Barndorff-Nielsen 1997):

      f(x) = αδ · exp(δγ + β(x−μ)) · K₁(αq) / (π q),
      γ = √(α²−β²),  q = √(δ²+(x−μ)²)

    For tail stability use the scaled Bessel K₁(z) = kve(1,z)·exp(−z), so
    log K₁(z) = log kve(1,z) − z.  Matches scipy.stats.norminvgauss.logpdf
    to machine precision.
    """
    y     = value - mu
    q     = pt.sqrt(delta**2 + y**2)
    gamma = pt.sqrt(alpha**2 - beta**2)
    z     = alpha * q
    log_k1 = pt.log(pt.kve(1.0, z)) - z
    return (pt.log(alpha) + pt.log(delta) + delta * gamma
            + beta * y + log_k1 - np.log(np.pi) - pt.log(q))


def _nig_random(mu, alpha, beta, delta, rng=None, size=None):
    """Draw NIG samples via scipy's norminvgauss (a = αδ, b = βδ)."""
    return _sps.norminvgauss.rvs(a=alpha * delta, b=beta * delta,
                                 loc=mu, scale=delta, size=size, random_state=rng)


def build_nig_model(r):
    """
    NIG with the (γ, β) → α reparametrisation.  Sampling γ = √(α²−β²) > 0 and
    β ∈ ℝ, then setting α = √(γ²+β²), makes the constraint α > |β| automatic
    and avoids the awkward bounded geometry of sampling α and β directly.
    """
    with pm.Model() as model:
        mu    = pm.Normal("mu", mu=0.0, sigma=1e-3)
        gamma = pm.Gamma("gamma", alpha=4.0, beta=4.0 / 52.0)   # mean 52, sd 26
        beta  = pm.Normal("beta", mu=0.0, sigma=15.0)
        delta = pm.HalfNormal("delta", sigma=0.02)
        alpha = pm.Deterministic("alpha", pt.sqrt(gamma**2 + beta**2))
        pm.CustomDist("returns", mu, alpha, beta, delta,
                      logp=_nig_logp, random=_nig_random, observed=r)
    return model


def build_student_t_model(r, nu_floor=0.0):
    """
    Location-scale Student-t.  nu_floor=0 uses ν ~ Gamma(2, 0.1).
    nu_floor=2 uses the shifted prior ν = 2 + Gamma(2, 0.1), forcing finite
    variance (a defensible constraint when variance-based risk readings must
    stay valid).
    """
    with pm.Model() as model:
        mu    = pm.Normal("mu", mu=0.0, sigma=1e-3)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        if nu_floor and nu_floor > 0:
            nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.1)
            nu     = pm.Deterministic("nu", nu_floor + nu_raw)
        else:
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        pm.StudentT("returns", nu=nu, mu=mu, sigma=sigma, observed=r)
    return model


def make_model(name, r, nu_floor=0.0):
    if name == "gaussian":
        return build_gaussian_model(r)
    if name == "laplace":
        return build_laplace_model(r)
    if name == "student_t":
        return build_student_t_model(r, nu_floor=nu_floor)
    if name == "nig":
        return build_nig_model(r)
    raise ValueError(f"unknown model: {name}")


ALL_MODELS = ["gaussian", "laplace", "student_t", "nig"]


# ── prior predictive check (the decision step) ────────────────────────────────

def prior_predictive_check(model, r, name, save_dir, draws=500):
    """
    Simulate returns from the prior and judge whether they are plausible.

    Reports the fraction of simulated daily returns beyond ±25% (a move larger
    than anything in the 2000–2024 sample bar nothing — the worst actual day
    was about −12%).  A sane weakly-informative prior should put almost no mass
    there.  Saves a figure comparing the prior-predictive spread to the
    observed return range.
    """
    with model:
        pp = pm.sample_prior_predictive(draws=draws, random_seed=7)

    sim = np.asarray(pp.prior_predictive["returns"]).ravel()
    obs_max = float(np.max(np.abs(r)))
    frac_extreme = float(np.mean(np.abs(sim) > 0.25))
    q = np.quantile(sim, [0.001, 0.01, 0.5, 0.99, 0.999])

    print(f"\n── Prior predictive check: {name} ──")
    print(f"  observed |return| max : {obs_max:6.3%}")
    print(f"  prior-pred 0.1/99.9%  : [{q[0]:+.3%}, {q[4]:+.3%}]")
    print(f"  prior-pred 1/99%      : [{q[1]:+.3%}, {q[3]:+.3%}]")
    print(f"  frac |sim| > 25%      : {frac_extreme:.4%}")
    verdict = "PLAUSIBLE" if frac_extreme < 0.02 else "TOO WIDE — revisit priors"
    print(f"  verdict               : {verdict}")

    fig, ax = plt.subplots(figsize=(7, 4))
    clip = np.clip(sim, -0.30, 0.30)
    ax.hist(clip, bins=120, density=True, color="0.6",
            label="Prior predictive (clipped ±30%)")
    ax.axvline(-obs_max, color="crimson", ls="--", lw=1)
    ax.axvline(+obs_max, color="crimson", ls="--", lw=1,
               label=f"Observed |return| max ({obs_max:.1%})")
    ax.set_title(f"Prior predictive returns — {DISPLAY.get(name, name)}")
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(save_dir, f"week4_prior_predictive_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out}")

    return {"frac_extreme": frac_extreme, "obs_max": obs_max, "plausible": frac_extreme < 0.02}


# ── sampling and diagnostics ──────────────────────────────────────────────────

def run_nuts(model, draws=1000, tune=1000, chains=4, seed=42):
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1,
            target_accept=0.9, random_seed=seed, progressbar=True,
        )
    return idata


def summarise(idata, name):
    mle = MLE[name]
    var_names = [v for v in mle if v in idata.posterior.data_vars]
    s = az.summary(idata, var_names=var_names, ci_prob=0.94, ci_kind="hdi")
    # arviz 1.x names the interval columns hdi94_lb / hdi94_ub; detect them.
    lb = next(c for c in s.columns if c.endswith("_lb"))
    ub = next(c for c in s.columns if c.endswith("_ub"))
    cols = ["mean", "sd", lb, ub, "r_hat", "ess_bulk"]
    s = s[cols].rename(columns={lb: "hdi_lo", ub: "hdi_hi"})
    s["MLE (wk3)"] = [mle[v] for v in s.index]
    print(f"\n── Posterior summary: {name} ──")
    print(s.to_string())
    max_rhat = float(s["r_hat"].max())
    min_ess  = float(s["ess_bulk"].min())
    print(f"  max R-hat = {max_rhat:.3f}   min ESS = {min_ess:.0f}"
          f"   {'OK' if max_rhat < 1.01 else 'CHECK CONVERGENCE'}")
    return s


def plot_trace(idata, name, save_dir):
    """
    Trace + marginal posterior, built directly with matplotlib (the arviz 1.x
    plotting API returns a PlotCollection that is awkward to drive from here).
    Left column: per-chain posterior density.  Right column: per-chain trace.
    """
    var_names = [v for v in MLE[name] if v in idata.posterior.data_vars]
    post = idata.posterior
    n = len(var_names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.4 * n), squeeze=False)
    for i, v in enumerate(var_names):
        draws = post[v].values            # (chain, draw)
        for c in range(draws.shape[0]):
            chain = draws[c]
            axes[i, 0].hist(chain, bins=60, density=True, histtype="step", alpha=0.8)
            axes[i, 1].plot(chain, lw=0.5, alpha=0.7)
        axes[i, 0].axvline(MLE[name][v], color="k", ls="--", lw=1,
                           label="MLE (wk3)")
        axes[i, 0].set_ylabel(v)
        axes[i, 0].legend(fontsize=7)
        axes[i, 1].set_ylabel(v)
    axes[0, 0].set_title("Posterior density (by chain)")
    axes[0, 1].set_title("Trace (by chain)")
    axes[-1, 1].set_xlabel("Draw")
    fig.suptitle(f"NUTS trace — {DISPLAY.get(name, name)}", y=1.0)
    fig.tight_layout()
    out = os.path.join(save_dir, f"week4_trace_{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def save_posterior_csv(idata, name, data_dir):
    """
    Persist posterior draws as a tidy CSV (chain, draw, then each parameter).
    Avoids a NetCDF backend dependency and reloads trivially for Week 5.
    """
    post = idata.posterior
    var_names = [v for v in MLE[name] if v in post.data_vars]
    cols = {}
    for v in var_names:
        arr = post[v].values                       # (chain, draw)
        cols[v] = arr.ravel()
    n_chain, n_draw = post[var_names[0]].values.shape
    idx = pd.MultiIndex.from_product([range(n_chain), range(n_draw)],
                                     names=["chain", "draw"])
    df = pd.DataFrame(cols, index=idx).reset_index()
    out = os.path.abspath(os.path.join(data_dir, f"week4_posterior_{name}.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  posterior draws saved -> {out}")


# ── orchestration ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Week 4 Bayesian NUTS estimation")
    ap.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--nu_floor", type=float, default=0.0,
                    help="Student-t: 2.0 forces finite variance (nu = 2 + Gamma)")
    ap.add_argument("--prior_only", action="store_true",
                    help="Stop after the prior predictive check")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "..", "figures"))
    args = ap.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    r = load_returns(use_cache=not args.no_cache)
    r_vals = r.values.astype(float)

    models = ALL_MODELS if args.model == "all" else [args.model]
    for name in models:
        print("\n" + "=" * 64)
        print(f"MODEL: {name}")
        print("=" * 64)
        model = make_model(name, r_vals, nu_floor=args.nu_floor)

        prior_predictive_check(model, r_vals, name, save_dir)
        if args.prior_only:
            continue

        idata = run_nuts(model, draws=args.draws, tune=args.tune, chains=args.chains)
        summarise(idata, name)
        plot_trace(idata, name, save_dir)
        save_posterior_csv(idata, name, os.path.join(save_dir, "..", "data"))


if __name__ == "__main__":
    main()
