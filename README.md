# Beyond Black-Scholes: Fitting Levy Processes to Stock Returns

**BSc Financial Mathematics and Economics - Applied Statistics Research Project**

This project compares five distributional models for S&P 500 daily log-returns:
Gaussian, Laplace, Student-t, Variance-Gamma (VG), and Normal Inverse Gaussian (NIG).
Estimation is carried out by Maximum Likelihood (MLE) and Bayesian MCMC (PyMC/NUTS).
Risk measures (VaR, ES) are evaluated via rolling backtests using the Christoffersen test framework,
and the four crisis events in the sample are assessed against Taleb's Black Swan criteria.

## Repository structure

Each week is self-contained: code, figures, and write-up all live under the
corresponding week folder.

```
week1/
  Summer_Research_Proposal_DRAFT.pdf
  Week1_Literature_Review_final.pdf

week2/
  code/     - week2_gaussian_student_mle.py (standalone, runs end-to-end)
  figures/  - trace plot, density overlay, QQ plots, annual marginals
  writeup/  - Week2_Results.md

week3/
  code/     - levy_models.py, week3_all_models_mle.py, week3_subperiod.py,
              week3_leadup_regression.py, week3_vix_regression.py, week3_main.py
  figures/  - five-model density/QQ/risk, sub-period bars, lead-up regression plots
  writeup/  - Week3_Results.md, Week3_Regression.md

week4/
  code/     - week4_bayesian.py (Gaussian, Laplace, Student-t, NIG via NUTS)
  figures/  - prior predictive checks and NUTS traces for each model
  writeup/  - Week4_Bayesian.md

week5/
  code/     - week5_ppc.py (posterior predictive checks + convergence diagnostics)
  figures/  - PPC density fans, tail-risk bands, per-chain trace diagnostics
  writeup/  - Week5_PPC.md
  data/     - week5_ppc_stats.csv, week5_diagnostics.csv (regenerated; gitignored)

week6/
  code/     - week6_weekday.py, week6_open_close.py,
              week6_param_regressions.py, week6_earnings.py
  figures/  - weekday MLEs, overnight/intraday split, weekly vol,
              quarterly NIG fits, delta regressions, earnings profile
  writeup/  - Week6_Results.md
  data/     - cached OHLC and VIX, quarterly parameter panel (gitignored)

week7/
  code/     - week7_backtest.py (rolling VaR backtest, Christoffersen tests)
  figures/  - VaR forecasts with violations, Basel traffic light,
              cumulative violation curves
  writeup/  - Week7_Backtest.md
  data/     - daily forecast/hit series, test tables (gitignored)

week8/
  code/     - week8_black_swan.py (surprise / oos / extremes / clustering modes),
              generate_week8_doc.py
  figures/  - crisis return periods, posterior return periods, pre-crisis vs
              full-sample surprise, simulated extremes, violation clustering
  writeup/  - Week8_BlackSwan.md
  data/     - surprise, out-of-sample, extremes and clustering tables (gitignored)
```

The Week 9 final report (the standalone research paper) is still being written
and will be added to the repository when it is complete.

## Weekly progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Literature review and research proposal | Complete |
| 2 | Gaussian and Student-t MLE | Complete |
| 3 | Five-model MLE (incl. Laplace, VG, NIG); sub-period analysis; lead-up regression | Complete |
| 4 | Bayesian estimation (PyMC/NUTS) | Complete |
| 5 | Posterior predictive checks; diagnostics | Complete |
| 6 | Weekday and open/close structure; quarterly parameter regressions; earnings windows | Complete |
| 7 | Rolling VaR backtest (Christoffersen) | Complete |
| 8 | Black Swan analysis of the four crisis events | Complete |
| 9 | Final write-up | In progress |

## Week 1

Literature review covering the key motivation for moving beyond Black-Scholes,
and a research proposal outlining the modelling approach for the full project.
References: Madan, Carr & Chang (1998); Cont (2001); Christoffersen (1998).

## Week 2

Fits a Gaussian and location-scale Student-t to 6,287 daily S&P 500 log-returns
(2000-2024) by MLE. Key findings:

- nu-hat = 2.648 (95% CI: 2.43-2.87) - consistent with infinite kurtosis
- Student-t AIC is 1,803 units better than Gaussian after penalising the extra parameter
- 99% Expected Shortfall: -5.81% (Student-t) vs -3.24% (Gaussian), a 79.5% gap
- Sub-period analysis shows GFC and COVID produce nu-hat near 2.3-2.6;
  dot-com and rate-hike cycles return nu-hat near 6.5

### Running Week 2

```bash
pip install numpy pandas scipy yfinance matplotlib
python week2/code/week2_gaussian_student_mle.py
```

## Week 3

Extends the comparison to five distributional models on the same 6,287 daily
returns, adds a sub-period analysis across the four shock windows, and adds a
lead-up regression on forward risk. Write-ups: `Week3_Results.md` (the model
comparison) and `Week3_Regression.md` (the lead-up regression and the fitted
parameters versus VIX). Key findings:

- The Laplace (double-exponential), the symmetric VG special case (θ = 0, ν = 1),
  reaches ΔAIC −1,771 with only two parameters, 96% of the four-parameter NIG's
  gain over the Gaussian. Most of the improvement over the Gaussian comes from
  exponential rather than thin tails.
- NIG is the best fit overall (ΔAIC −1,851) and the only model not rejected by KS.
- 99% Expected Shortfall: −3.24% (Gaussian), −3.91% (Laplace), −4.28% (VG),
  −5.19% (NIG), −5.81% (Student-t).
- At the 97.5% ES level FRTB mandates, the Gaussian (−2.84%) sits 28% below
  the NIG (−3.94%). The shortfall is material at the confidence level the
  regulation uses, well before the deep 99% tail.
- Sub-period fits show the GFC and COVID differ in kind from the dot-com and
  rate-hike episodes: ν near 2.3–2.6 versus 6.5, NIG α near 18–26 versus 99–112.
- Lead-up regression on forward 21-day realised volatility: tail/VIX/drawdown
  factors add 11 points of R² (0.44 to 0.55) over a volatility-only baseline,
  with Newey-West HAC standard errors. Framed as retrospective risk attribution,
  not return forecasting.

### Running Week 3

```bash
pip install numpy pandas scipy yfinance matplotlib
python week3/code/week3_main.py            # mle + sub-period + lead-up regression
python week3/code/week3_main.py --mode mle # five-model comparison only
```

## Week 4

Refits four of the marginals (Gaussian, Laplace, Student-t, NIG) in a Bayesian
setting and samples their full posteriors with the No-U-Turn Sampler (NUTS) in
PyMC. The Laplace stands in for the Variance-Gamma family as its symmetric
special case (θ = 0, ν = 1). Write-up: `Week4_Bayesian.md`. Key points:

- Priors are weakly informative and anchored on the Week 3 MLE. Each one was
  accepted on the basis of a prior predictive check: returns simulated from the
  prior alone stay inside plausible bounds, with a negligible fraction of moves
  beyond the worst day in the sample.
- The NIG keeps its full four-parameter density, written as a custom PyTensor
  log-density using the modified Bessel function K1, so NUTS gradients flow
  through it without a latent mixing variable. The density matches scipy to
  machine precision.
- Every model converged cleanly: R-hat 1.00 across the board and effective
  sample size above 2,000, and the posteriors reproduce the MLEs.
- Student-t degrees of freedom: posterior mean 2.66, 94% HDI (2.46, 2.87). The
  whole interval sits above 2, so the heavy tails come with a finite variance.
- NIG asymmetry: posterior for β is negative across its 94% HDI (-8.75, -3.25),
  confirming the left skew as a stable feature rather than a point estimate.

### Running Week 4

```bash
pip install numpy pandas scipy yfinance matplotlib pymc arviz
python week4/code/week4_bayesian.py               # all four models, full sampling
python week4/code/week4_bayesian.py --prior_only  # prior predictive checks only
```

## Week 5

Validates the four Bayesian fits with posterior predictive checks and fuller
convergence diagnostics, working directly from the Week 4 posterior CSVs (no
re-sampling). Write-up: `Week5_PPC.md`. Key points:

- Each model generates 1,000 replicate return histories from its posterior;
  the observed data is scored against the replicate bands with two-sided
  posterior predictive p-values. Statistics cover the marginal distribution
  (moments, extremes, VaR/ES at 95%, 97.5% and 99%) plus one dependence
  statistic: the lag-1 autocorrelation of squared returns.
- The Gaussian fails everything beyond the centre of the distribution. The
  Laplace closes about half the tail gap. The Student-t covers the tail
  quantiles but its moments are too heavy to pin down. The NIG passes every
  marginal check, including skew and the FRTB 97.5% ES.
- All four models fail the volatility clustering statistic (observed 0.32
  vs replicate bands on zero). That is the limit of any static marginal and
  the reason for the Week 7 rolling backtest.
- Hand-computed diagnostics on the saved chains: split R-hat 1.00 throughout,
  bulk ESS > 2,000, tail ESS > 2,100, small MCSEs.

### Running Week 5

```bash
pip install numpy pandas scipy matplotlib
python week5/code/week5_ppc.py                 # all four models
python week5/code/week5_ppc.py --model nig     # single model
```

## Week 6

Turns to the calendar and cross-sectional structure of the same returns:
day-of-week effects, the market-open vs market-closed split, week-on-week
volatility, quarterly parameter regressions, and earnings seasons.
Write-up: `Week6_Results.md`. Key findings:

- The week has a real shape. The Mon / midweek / Fri grouping beats a
  pooled week decisively (LR = 42.9, p < 0.0001) and five separate days
  add nothing over it (p = 0.59). Monday, which carries the weekend gap,
  has the heaviest tail of the week (Student-t nu 2.17 vs Friday's 3.10).
- The open market earns 61% of the variance; the closed market earns 20%
  but takes the jumps: overnight nu 2.14 and kurtosis 36 against intraday
  nu 2.79 and kurtosis 5 (2015-2024, where Yahoo's opens are usable).
- Weekly realised volatility persists week on week (slope 0.72, R2 0.52),
  and weekly returns lighten to nu 3.38 from the daily 2.648, the
  aggregational Gaussianity of Cont (2001).
- Quarterly NIG delta tracks the VIX (R2 0.64 in logs, t = 9.7 across the
  67 well-identified quarters), and at quarterly frequency the skew
  parameters show the VIX relationship the annual regression lacked power
  to find. Tail-decay parameters stay flat against the VIX. A third of
  quarters sit on the NIG's Gaussian-limit ridge.
- Earnings seasons leave index volatility flat to the decimal (16.5 /
  16.4 / 16.5 percent before, during, after). Index tail risk is macro
  risk; single-name earnings surprises diversify away.

### Running Week 6

```bash
pip install numpy pandas scipy yfinance matplotlib
python week6/code/week6_weekday.py
python week6/code/week6_open_close.py
python week6/code/week6_param_regressions.py
python week6/code/week6_earnings.py
```

## Week 7

Puts the models to work out of sample: a rolling 500-day window, refitted
every 21 days, issues one-day-ahead VaR forecasts at 95%, 97.5% and 99% for
5,787 days (2002-2024), scored with the Christoffersen (1998) likelihood-ratio
framework, the Basel traffic light and an FRTB-style ES comparison.
Write-up: `Week7_Backtest.md`. Key findings:

- At 95% the Gaussian and NIG pass unconditional coverage; the Student-t is
  the worst model at that level (358 hits vs 289 expected) because its heavy
  tail (nu near 2.6) thins the shoulders where the 5% quantile lives.
- At 99% the Gaussian collapses: 152 violations against 58 expected, a factor
  of 2.6 (LR_uc = 107). The NIG is closest at 91 (factor 1.57), yet even it is
  rejected on coverage.
- Every model fails the independence test at every level (all p < 0.0001).
  Violations arrive in bursts (late 2008, March 2020), the out-of-sample
  counterpart of the Week 5 volatility-clustering failure.
- Basel traffic light: the Gaussian spends 31% of days in the red zone with a
  worst 250-day count of 32 violations; the NIG spends 14% with a worst of 15.
- FRTB ES check at 97.5%: realised losses on breach days are 33% deeper than
  the Gaussian's booked ES (ratio 1.33); Student-t 0.97, NIG 1.07.

### Running Week 7

```bash
pip install numpy pandas scipy matplotlib
python week7/code/week7_backtest.py
```

## Week 8

Asks whether the four crisis events (dot-com, GFC, COVID-19, the 2022 Fed
hikes) were Black Swans in Taleb's (2007) sense, treating "outside regular
expectation" as a model-relative quantity. Four measurements: implied return
periods of each crisis's worst day under the five full-sample fits (MLE and
Bayesian); the same question under pre-crisis fits that use only information
available at the time; a simulation check on whether 4,000 model-generated
25-year histories can reproduce the observed extremes; and a permutation test
on the timing of the Week 7 backtest violations.
Write-up: `Week8_BlackSwan.md`. Key findings:

- The worst COVID-19 day (-12.77%, 16 March 2020) is a once-per-5.6x10^22-years
  event under the Gaussian, once per 62,400 years under the Laplace, once per
  119 years under the NIG and once per decade under the Student-t. Days at
  least that bad occurred once in the 25-year sample. Under the Gaussian all
  four crises are Black Swans; under the Student-t none of them are.
- The exponential-tailed Laplace and VG, near-optimal by likelihood, misprice
  the extremes by three to four orders of magnitude. The Black Swan question
  is decided in the far tail, which the likelihood barely sees.
- Out of sample the Student-t is the robust model: fitted on two calm
  pre-crisis years it still prices the GFC and COVID worst days at 39 and 34
  years. The NIG loses three orders of magnitude on calm windows (its
  Gaussian-limit ridge), a capital-relevant fragility under rolling refits.
  The 2022 tightening was a white swan under every model.
- No iid model reproduces the observed worst month (-40% over 21 days), and
  none puts 11-12 of its 99% VaR violations inside one 21-day window as
  observed in March 2020 (permutation p < 1/20,000 for all four models).
  A better marginal cures the magnitude swan; the timing swan remains.

### Running Week 8

```bash
pip install numpy pandas scipy matplotlib
python week8/code/week8_black_swan.py                  # all four measurements
python week8/code/week8_black_swan.py --mode surprise  # or oos / extremes / clustering
```

## Week 9

The final write-up: a standalone research paper reorganising the whole project
by topic rather than by week, from the stylised facts through the model
hierarchy, estimation, in-sample results, parameter instability, the rolling
backtest and the Black Swan assessment to a closing scorecard. In progress;
it will be added to the repository when complete.

## Key references

- Madan, Carr & Chang (1998) - Variance Gamma process
- Cont (2001) - Stylised facts of financial returns
- Christoffersen (1998) - VaR backtesting framework
- Basel Committee on Banking Supervision (2013) - FRTB
- Taleb (2007) - The Black Swan
