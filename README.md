# Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

**BSc Financial Mathematics and Economics — Applied Statistics Research Project**

This project compares four distributional models for S&P 500 daily log-returns:
Gaussian, Student-t, Variance-Gamma (VG), and Normal Inverse Gaussian (NIG).
Estimation is carried out by both Maximum Likelihood (MLE) and Bayesian MCMC (PyMC/NUTS).
Risk measures (VaR, ES) are evaluated via rolling backtests using the Kupiec and
Christoffersen test framework.

## Repository structure

Each week is self-contained: code, figures, and write-up all live under the
corresponding week folder.

```
week1/
  references/   — key papers and bibliography
  writeup/      — literature review and research proposal

week2/
  code/         — Python and R scripts (Gaussian & Student-t MLE)
  figures/      — diagnostic plots (density, QQ, trace, annual marginals)
  writeup/      — results write-up with parameter tables and risk analysis

week3/          — Variance-Gamma (VG) model  [upcoming]
week4/          — NIG model; AIC/BIC/KS comparison  [upcoming]
week5/          — Bayesian VG in PyMC  [upcoming]
week6/          — Posterior predictive checks; rolling backtest  [upcoming]
week7/          — Final write-up  [upcoming]
```

## Weekly progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Literature Review | Complete |
| 2 | Gaussian & Student-t MLE | Complete |
| 3 | VG & NIG MLE | Upcoming |
| 4 | Sub-period analysis | Upcoming |
| 5 | Bayesian MCMC | Upcoming |
| 6 | Rolling VaR backtest + VIX regression | Upcoming |
| 7 | Final write-up | Upcoming |

## Running Week 2

```bash
pip install numpy pandas scipy yfinance matplotlib
python week2/code/week2_gaussian_student_mle.py
```

For the R version:
```r
install.packages(c("quantmod", "MASS", "ggplot2", "gridExtra"))
source("week2/code/week2_gaussian_student_mle.R")
```

## Key references

- Madan, Carr & Chang (1998) — Variance Gamma process
- Cont (2001) — Stylised facts of financial returns
- Christoffersen (1998) — VaR backtesting framework
