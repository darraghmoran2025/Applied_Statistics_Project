"""
week3_main.py — Week 3 Orchestrator
Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

Runs any combination of the three Week 3 analyses via the --mode flag.
Each mode maps directly to its dedicated module, which can also be run
independently.  This orchestrator exists so a single command can reproduce
the complete Week 3 deliverable.

Usage
─────
    # Run everything (default)
    python week3_main.py

    # Run a single component
    python week3_main.py --mode mle
    python week3_main.py --mode subperiod
    python week3_main.py --mode vix

    # Custom output directory and Monte Carlo size
    python week3_main.py --mode all --save_dir ../figures --n_sim 1000000

Modes
─────
mle        Four-model MLE on full sample; AIC/BIC table; VaR/ES comparison;
           density and QQ plots.  (week3_all_models_mle.py)

subperiod  Fit all four models to each of the four market shock windows;
           empirical shock statistics; AIC improvement over Gaussian;
           parameter and AIC bar charts.  (week3_subperiod.py)

vix        Build annual VG/NIG parameter panel; OLS regression on average
           annual VIX; scatter + regression line plots.
           (week3_vix_regression.py)

all        Run mle → subperiod → vix in sequence, sharing the downloaded
           return series.

Arguments
─────────
--mode     {mle, subperiod, vix, all}   default: all
--save_dir  Path to save all figures.   default: week3/figures/
--n_sim     Monte Carlo draws for VaR/ES and QQ.  default: 500000
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def _find_week2():
    """Locate week2_gaussian_student_mle.py in either repo layout."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "week2", "code"),
        os.path.join(here, "..", "Week2"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "week2_gaussian_student_mle.py")):
            return c
    raise FileNotFoundError(
        "week2_gaussian_student_mle.py not found. "
        "Looked in: " + ", ".join(candidates)
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _find_week2())
sys.path.insert(0, _HERE)

from week2_gaussian_student_mle import fetch_returns
from levy_models import default_save_dir

from week3_all_models_mle    import run_all_models_mle
from week3_subperiod         import run_subperiod_analysis
from week3_vix_regression    import run_vix_regression
from week3_leadup_regression import run_leadup_regression


def main():
    parser = argparse.ArgumentParser(
        description="Week 3: Lévy process MLE, sub-period analysis, VIX regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["mle", "subperiod", "leadup", "vix", "all"],
        default="all",
        help="Which analysis to run (default: all). 'leadup' is the lead-up "
             "regression on forward risk; 'vix' is the legacy cross-sectional "
             "parameter-on-VIX regression (kept standalone, not in 'all').",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Directory to save figures (default: week3/figures/)",
    )
    parser.add_argument(
        "--n_sim",
        type=int,
        default=500_000,
        help="Monte Carlo draws for VaR/ES and QQ plots (default: 500000)",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir(__file__)

    print("=" * 70)
    print("Week 3 — Beyond Black-Scholes: Lévy Process Analysis")
    print(f"Mode: {args.mode}   |   Save dir: {os.path.abspath(save_dir)}")
    print("=" * 70)

    # Download return series once; share across all modes
    print("\nDownloading S&P 500 returns (Jan 2000 – Dec 2024)…")
    r_series = fetch_returns()
    print(f"  {len(r_series):,} daily log-returns loaded.\n")

    run_mle        = args.mode in ("mle",       "all")
    run_subperiod  = args.mode in ("subperiod", "all")
    run_leadup     = args.mode in ("leadup",    "all")
    run_vix        = args.mode == "vix"   # legacy regression: standalone only

    fitted_models = None

    if run_mle:
        fitted_models = run_all_models_mle(r_series, n_sim=args.n_sim, save_dir=save_dir)
        print()

    if run_subperiod:
        run_subperiod_analysis(r_series, save_dir=save_dir)
        print()

    if run_leadup:
        run_leadup_regression(r_series, save_dir=save_dir)
        print()

    if run_vix:
        run_vix_regression(r_series, save_dir=save_dir)
        print()

    print("=" * 70)
    print("Week 3 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
