# Beyond Black-Scholes: Fitting Levy Processes to Stock Returns

**BSc Financial Mathematics and Economics - Applied Statistics Research Project**

This project compares five distributional models for S&P 500 daily log-returns:
Gaussian, Laplace, Student-t, Variance-Gamma (VG), and Normal Inverse Gaussian (NIG).
Estimation is carried out by Maximum Likelihood (MLE) and Bayesian MCMC (PyMC/NUTS).
Risk measures (VaR, ES) are evaluated via rolling backtests using the Christoffersen test framework.

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

week5/      - Posterior predictive checks; convergence diagnostics  [upcoming]
week6/      - Rolling VaR backtest (Christoffersen)  [upcoming]
week7/      - Final write-up  [upcoming]
```

## Weekly progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Literature review and research proposal | Complete |
| 2 | Gaussian and Student-t MLE | Complete |
| 3 | Five-model MLE (incl. Laplace, VG, NIG); sub-period analysis; lead-up regression | Complete |
| 4 | Bayesian estimation (PyMC/NUTS) | Complete |
| 5 | Posterior predictive checks; diagnostics | Upcoming |
| 6 | Rolling VaR backtest (Christoffersen) | Upcoming |
| 7 | Final write-up | Upcoming |

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
- 99% Expected Shortfall: −3.24% (Gaussian), −3.91% (Laplace), −4.27% (VG),
  −5.19% (NIG), −5.81% (Student-t).
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

## Key references

- Madan, Carr & Chang (1998) - Variance Gamma process
- Cont (2001) - Stylised facts of financial returns
- Christoffersen (1998) - VaR backtesting framework
