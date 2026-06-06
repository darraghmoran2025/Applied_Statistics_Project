# Beyond Black-Scholes: Fitting Levy Processes to Stock Returns

**BSc Financial Mathematics and Economics - Applied Statistics Research Project**

This project compares four distributional models for S&P 500 daily log-returns:
Gaussian, Student-t, Variance-Gamma (VG), and Normal Inverse Gaussian (NIG).
Estimation is carried out by Maximum Likelihood (MLE) and Bayesian MCMC (PyMC/NUTS).
Risk measures (VaR, ES) are evaluated via rolling backtests using the Kupiec and
Christoffersen test framework.

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

week3/      - Variance-Gamma model  [upcoming]
week4/      - NIG model; AIC/BIC/KS comparison  [upcoming]
week5/      - Bayesian VG in PyMC  [upcoming]
week6/      - Posterior predictive checks; rolling backtest  [upcoming]
week7/      - Final write-up  [upcoming]
```

## Weekly progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Literature review and research proposal | Complete |
| 2 | Gaussian and Student-t MLE | Complete |
| 3 | Variance-Gamma model | Upcoming |
| 4 | NIG model; model comparison | Upcoming |
| 5 | Bayesian MCMC | Upcoming |
| 6 | Rolling VaR backtest + VIX regression | Upcoming |
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

## Key references

- Madan, Carr & Chang (1998) - Variance Gamma process
- Cont (2001) - Stylised facts of financial returns
- Christoffersen (1998) - VaR backtesting framework
