# Week 3: Lead-up regression on forward volatility

This regression studies how the risk environment evolves. It models forward 21-day realised volatility on a set of factors observable at the time, using daily S&P 500 data from 2000 to 2024 (n = 6,207), and then examines which of those factors were elevated in the run-up to the four shock windows. The target is the dispersion of returns over the coming month, not their direction.

---

## 1. The model

The dependent variable is forward realised volatility over the next h = 21 trading days, about one calendar month. For each day t it is the square root of the summed squared daily log-returns over days t+1 to t+h, scaled by 252/h and annualised. Squared returns measure the magnitude of moves regardless of sign.

Every predictor is built only from information available up to and including day t. Two of them, rolling skewness and rolling excess kurtosis, are short-window empirical analogues of the VG and NIG asymmetry and tail-heaviness parameters estimated elsewhere in Week 3, so the model can ask whether the recent tail shape, not just the recent volatility, carries information about the risk to come.

**Table 1. The seven lead-up predictors. All are trailing quantities measured at day t.**

| Predictor | Definition |
|-----------|------------|
| Trailing 21d vol | Annualised standard deviation of returns over the last 21 days |
| VIX level | CBOE Volatility Index level (options-implied 30-day vol) |
| ΔVIX (5d) | 5-day change in the VIX |
| \|return\| (1d) | Absolute value of the most recent daily return |
| Rolling skew (21d) | Sample skewness of returns over the last 21 days |
| Rolling kurtosis (21d) | Excess kurtosis of returns over the last 21 days (tail proxy) |
| Drawdown (252d) | Current price relative to its trailing 252-day peak (≤ 0) |

The model is a single OLS regression on the full daily sample. Consecutive forward windows overlap, since day t and day t+1 share 20 of their 21 days, so the residuals are strongly autocorrelated; all standard errors are Newey-West HAC (Bartlett kernel, bandwidth 21, the overlap length). Coefficients are reported in standardised (z-score) form, so a coefficient of 0.40 means a one-standard-deviation rise in the factor goes with a 0.40-standard-deviation rise in forward volatility, holding the others fixed.

---

## 2. Results

A baseline using only trailing 21-day volatility explains forward volatility with R² = 0.441, as expected from volatility clustering. Adding the other six factors raises R² to 0.553. The extra 0.112, about eleven percentage points, comes from the implied-volatility, shock, tail and drawdown factors over and above plain volatility persistence. The risk environment is better described with tail-aware factors than with volatility alone, the same message the Lévy models give for the static distribution, now in a dynamic form.

**Table 2. Variance in forward 21-day realised volatility explained. n = 6,207.**

| Model | Predictors | R² | Adjusted R² |
|-------|-----------|----|-------------|
| Baseline | Trailing 21d vol only | 0.441 | 0.441 |
| Full | All seven lead-up factors | 0.553 | 0.552 |
| Increment | Tail / VIX / drawdown factors | +0.112 | n/a |

The VIX level is the strongest factor (+0.40, t = 4.2): options-implied volatility is the market's own forward-looking estimate of dispersion. The 5-day VIX change is also positive and significant (+0.12, t = 2.4), as is the latest absolute return (+0.05, t = 2.2). The drawdown and rolling-kurtosis loadings are the interesting ones. Drawdown is negative (−0.11, t = −1.8): the nearer the market sits to its trailing peak, the higher the volatility that tends to follow. Rolling excess kurtosis is also negative (−0.04, t = −2.1); once volatility and the VIX are in the model, a kurtosis spike tends to mark a jump that has already happened rather than risk still building. Rolling skewness is indistinguishable from zero.

**Table 3. Standardised coefficients of the full model, Newey-West HAC standard errors (bandwidth 21). The intercept is zero by construction and is omitted.**

| Factor | Std. coefficient | HAC SE | t-stat | p-value |
|--------|------------------|--------|--------|---------|
| VIX level | +0.399 | 0.094 | +4.23 | <0.001 |
| Trailing 21d vol | +0.203 | 0.105 | +1.94 | 0.053 |
| ΔVIX (5d) | +0.122 | 0.050 | +2.44 | 0.015 |
| \|return\| (1d) | +0.052 | 0.023 | +2.24 | 0.025 |
| Rolling skew (21d) | −0.025 | 0.026 | −0.97 | 0.333 |
| Rolling kurtosis (21d) | −0.040 | 0.019 | −2.09 | 0.037 |
| Drawdown (252d) | −0.115 | 0.063 | −1.82 | 0.069 |

![Figure 1. Standardised factor loadings with Newey-West HAC 95% intervals. The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.](../figures/week3_leadup_coefs.png)

*Figure 1. Standardised factor loadings with Newey-West HAC 95% intervals. The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.*

![Figure 2. Actual forward 21-day realised volatility (black) against in-sample fitted values (red), with the four shock windows shaded; in-sample R² = 0.55.](../figures/week3_leadup_fit.png)

*Figure 2. Actual forward 21-day realised volatility (black) against the in-sample fitted values (red), with the four shock windows shaded. The model tracks the broad evolution of risk, including the 2008 and 2020 spikes; in-sample R² = 0.55.*

---

## 3. What was elevated before each shock

For each shock window, Table 4 reports the average standardised level of each factor over the 21 trading days immediately before the window began. Because the factors are z-scored over the full sample, +0.7 means the factor stood 0.7 sample standard deviations above its typical level going into the episode, and a negative value means it was below.

The clearest pattern is drawdown, positive before all three episodes (+0.34, +0.68, +0.64): each shock began from a market sitting near its trailing peak rather than from an already depressed level. Beyond that the picture is mixed. The Global Financial Crisis was preceded by mildly elevated volatility and VIX (+0.39 and +0.30), consistent with a slow build-up. COVID-19 was the opposite: volatility, the VIX and skewness were all well below average in the weeks before the crash (−0.80, −0.70, −0.57), because that shock erupted from an unusually calm market.

**Table 4. Mean standardised factor level in the 21 trading days before each shock window (z-scores).**

| Shock window | Trail vol | VIX | ΔVIX | \|ret\| | Skew | Kurt | Drawdn |
|--------------|-----------|-----|------|---------|------|------|--------|
| GFC | +0.39 | +0.30 | −0.36 | −0.09 | +0.04 | −0.25 | +0.34 |
| COVID-19 | −0.80 | −0.70 | +0.25 | −0.26 | −0.57 | −0.47 | +0.68 |
| Fed rate hikes | +0.14 | +0.12 | −0.59 | +0.13 | −0.14 | −0.64 | +0.64 |

The dot-com crash is omitted because its start date (March 2000) falls inside the 252-day warm-up required by the drawdown factor, so a clean 21-day pre-window is not available in the sample.

![Figure 3. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.](../figures/week3_leadup_factors.png)

*Figure 3. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.*

---

## 4. Limitations

The estimates are in-sample. The predictors are correlated (most obviously trailing volatility and the VIX level), so the coefficients are partial associations rather than independent effects. The overlapping forward windows are the reason for the Newey-West standard errors, and they make the effective sample smaller than the nominal 6,207 observations. The 21-day horizon is a modelling choice; a shorter or longer one shifts the relative weight of the fast factors (the VIX change, the latest absolute return) and the slow ones (drawdown).
