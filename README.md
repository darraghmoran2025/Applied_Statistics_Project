# Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

**BSc Financial Mathematics and Economics — Applied Statistics Research Project**

This project compares four distributional models for S&P 500 daily log-returns:
Gaussian, Student-t, Variance-Gamma (VG), and Normal Inverse Gaussian (NIG).
Estimation is carried out by both Maximum Likelihood (MLE) and Bayesian MCMC (PyMC/NUTS).
Risk measures (VaR, ES) are evaluated via rolling backtests using the Kupiec and
Christoffersen test framework.

## Project structure

```
levy-project/
  data/           # Data fetching utilities (S&P 500, VIX)
  models/         # One module per distribution
  risk/           # VaR/ES computation and backtesting
  utils/          # Shared helpers: Hessian, plotting, stats
  scripts/        # Week-by-week runner scripts
  docs/           # Write-ups and figures
  website/        # GitHub Pages site
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

## Setup

```bash
pip install numpy pandas scipy yfinance matplotlib pymc arviz
```

R dependencies: `quantmod`, `MASS`, `ggplot2`, `gridExtra`

## Key references

- Madan, Carr & Chang (1998) — Variance Gamma process
- Cont (2001) — Stylised facts of financial returns
- Christoffersen (1998) — VaR backtesting framework
