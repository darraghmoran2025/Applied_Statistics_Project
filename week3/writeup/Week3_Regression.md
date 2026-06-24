# Week 3: Regression analyses

This write-up reports two regressions from Week 3. The first (Sections 1–3) models forward 21-day realised volatility on seven factors observable at day t, using daily S&P 500 data from 2000 to 2024 (n = 6,207), and examines which factors were elevated before the four shock windows. The target is the dispersion of returns over the coming month, not their direction. The second (Section 4) regresses the fitted VG and NIG parameters on the annual VIX.

---

## 1. The model

The dependent variable is forward realised volatility over the next 21 trading days, annualised. Every predictor is built from information available only up to day t. Rolling skewness and rolling excess kurtosis are short-window empirical analogues of the VG and NIG shape parameters, asking whether recent tail shape carries information about forward risk beyond recent volatility.

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

The model is OLS on the full daily sample. Consecutive forward windows overlap (day t and t+1 share 20 of 21 days), so residuals are strongly autocorrelated; all standard errors are Newey-West HAC (Bartlett kernel, bandwidth 21). Coefficients are reported in standardised form: a coefficient of 0.40 means a one-standard-deviation rise in a factor goes with a 0.40-standard-deviation rise in forward volatility.

Before running any regression it is worth showing why the VIX belongs in it. Figure 1 plots the VIX against trailing 21-day realised volatility, both annualised, for every day in the sample. The two track each other closely, with Pearson r = 0.88 and Spearman ρ = 0.84 (n = 6,207). Options-implied and recently realised volatility are largely the same quantity measured two ways, which is why the VIX carries real information about dispersion and is the natural lead predictor. The fitted slope is close to one, and most points sit a little below the 45-degree line, the usual variance risk premium: implied volatility tends to run slightly above the volatility that is later realised.

![Figure 1. VIX against trailing 21-day realised volatility (both annualised), with OLS fit and 45-degree line.](../figures/week3_vix_scatter.png)

*Figure 1. The VIX against trailing 21-day realised volatility, both annualised percent, one point per trading day (n = 6,207). Pearson r = 0.88, Spearman ρ = 0.84. The strong positive association is the empirical justification for using the VIX as the lead predictor of forward volatility.*

---

## 2. Results

A volatility-only baseline explains forward volatility with R² = 0.441, as expected from volatility clustering. Adding the six remaining factors raises R² to 0.553, an increment of 0.112 from the implied-volatility, tail and drawdown factors.

**Table 2. Variance in forward 21-day realised volatility explained. n = 6,207.**

| Model | Predictors | R² | Adjusted R² |
|-------|-----------|----|-------------|
| Baseline | Trailing 21d vol only | 0.441 | 0.441 |
| Full | All seven lead-up factors | 0.553 | 0.552 |
| Increment | Tail / VIX / drawdown factors | +0.112 |  |

VIX level is the strongest factor (+0.40, t = 4.2): options-implied volatility is the market's own forward-looking dispersion estimate. ΔVIX (+0.12, t = 2.4) and absolute return (+0.05, t = 2.2) are also positive and significant. Drawdown is negative (−0.11, t = −1.8): the nearer the market is to its trailing peak, the higher the forward volatility tends to be. Rolling excess kurtosis is also negative (−0.04, t = −2.1); once volatility and VIX are controlled for, a kurtosis spike marks a jump that has already happened rather than risk still building. Rolling skewness is indistinguishable from zero.

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

![Figure 2. Forest plot of standardised factor loadings with Newey-West HAC 95% intervals. The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.](../figures/week3_leadup_coefs.png)

*Figure 2. Forest plot of the standardised factor loadings: each point is a coefficient estimate and the whisker its Newey-West HAC 95% interval, with factors ordered by effect size and a filled circle marking intervals that exclude zero (diamonds mark those that span it). The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.*

![Figure 3. Left: actual forward 21-day realised volatility (black) against in-sample fitted values (red), with the four shock windows shaded. Right: fitted against actual, with the 45-degree line; in-sample R² = 0.55.](../figures/week3_leadup_fit.png)

*Figure 3. Left: actual forward 21-day realised volatility (black) against the in-sample fitted values (red), with the four shock windows shaded. Right: the same fit as fitted-against-actual, which removes the time axis altogether; points cluster on the 45-degree line, falling below it at the highest volatilities where the model under-predicts. In-sample R² = 0.55.*

The two curves sit on top of each other across the calm bulk of the sample but separate at the sharpest spikes. This is structural: 

- The actual (black) is forward-looking. The forward realised volatility for day t 
- The fitted (red) is built from trailing and contemporaneous factors, trailing 21-day volatility, the VIX level, the absolute return, all measured at t. 
- So at a jump the black leaps immediately and the red only catches up about a horizon later, once trailing volatility and the VIX have risen. 

The right panel makes the same point without a time axis: agreement is tight in the dense low-to-mid-volatility cloud and the largest actual volatilities sit below the 45-degree line, the days the model under-predicts.

---

## 3. What was elevated before each shock

Table 4 reports the average standardised level of each factor over the 21 trading days immediately before each shock window. Because factors are z-scored over the full sample, +0.7 means the factor stood 0.7 standard deviations above its full-sample mean going into the episode.

Drawdown is positive before all four episodes (+0.17, +0.34, +0.68, +0.64): each shock began from a market near its trailing peak. The dot-com and GFC run-ups carried mildly elevated volatility and VIX (+0.58/+0.45 and +0.39/+0.30). COVID-19 was the opposite: volatility, VIX and skewness were all well below average in the weeks before the crash (−0.80, −0.70, −0.57), because that shock erupted from an unusually calm market.

**Table 4. Mean standardised factor level in the 21 trading days before each shock window (z-scores).**

| Shock window | Trail vol | VIX | ΔVIX | \|ret\| | Skew | Kurt | Drawdn |
|--------------|-----------|-----|------|---------|------|------|--------|
| Dot-com crash | +0.58 | +0.45 | +0.03 | +0.23 | −0.61 | −0.15 | +0.17 |
| GFC | +0.39 | +0.30 | −0.36 | −0.09 | +0.04 | −0.25 | +0.34 |
| COVID-19 | −0.80 | −0.70 | +0.25 | −0.26 | −0.57 | −0.47 | +0.68 |
| Fed rate hikes | +0.14 | +0.12 | −0.59 | +0.13 | −0.14 | −0.64 | +0.64 |

The dot-com crash starts in March 2000, inside the 252-day drawdown warm-up of the 2000–2024 sample, so its lead-up window is otherwise undefined. To report all four shocks consistently, the dot-com row alone is computed on a panel extended with pre-2000 price history solely to warm the trailing-peak drawdown; every factor is still standardised against the 2000–2024 sample, so the other three rows are unchanged. This extension is used only for this table and is not carried into the rest of the project.

![Figure 4. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.](../figures/week3_leadup_factors.png)

*Figure 4. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.*

---

## 4. Fitted Lévy parameters versus the VIX

Each calendar year from 2000 to 2024 is fitted separately, giving 25 annual estimates of each VG and NIG parameter, each then regressed on that year's average VIX (n = 25).

**Table 5. The fitted parameters and how each relates to the VIX.**

| Parameter | What it controls | Relationship with the VIX |
|-----------|------------------|---------------------------|
| VG σ (scale) | Width of the diffusion component | Strong and positive (R² = 0.91, p < 0.001): scale rises almost one-for-one with the VIX |
| VG θ (asymmetry) | Skew; negative means a heavier left tail | None detectable (R² = 0.01, p = 0.57) |
| VG ν (variance rate) | Tail heaviness from the random time-change | None detectable (R² = 0.06, p = 0.24) |
| NIG α (tail heaviness) | Larger means lighter tails | Weak and negative (R² = 0.14, p = 0.07): tails tend to get heavier as the VIX rises |
| NIG β (asymmetry) | Skew; negative means a heavier left tail | None detectable (R² = 0.10, p = 0.12) |
| NIG δ (scale) | Overall spread | None detectable (R² = 0.00, p = 0.93) |

The VIX tracks distribution scale very closely (VG σ, R² = 0.91) but carries almost no information about tail shape or asymmetry. Every tail and skew parameter is statistically flat against the VIX; only NIG α shows a marginal signal. Knowing the VIX tells you how wide the distribution will be, but little about how heavy or lopsided its tails are. That distinction matters for ES-based capital, since ES is driven by tail shape rather than scale.

![Figure 5. Annual VG and NIG parameter estimates against average annual VIX, with OLS regression lines.](../figures/week3_vix_regression.png)

*Figure 5. Annual VG and NIG parameter estimates (2000 to 2024) against that year's average VIX, with OLS regression lines; each panel shows its R² and p-value. Only VG σ (top left) has a strong relationship with the VIX; the tail and asymmetry parameters are essentially flat.*

---

## 5. Limitations

These results are in-sample. The predictors in the lead-up regression are correlated (most obviously trailing volatility and VIX), so coefficients are partial associations rather than independent effects. Overlapping forward windows make the effective sample smaller than n = 6,207; Newey-West SEs correct for this but not for the correlation between predictors. The 21-day horizon is a modelling choice: a shorter or longer window shifts the relative weight of fast factors (VIX change, absolute return) and slow ones (drawdown).
