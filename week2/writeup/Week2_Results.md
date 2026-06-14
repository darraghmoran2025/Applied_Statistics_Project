# Week 2 results: Gaussian and Student-t MLE

## 1. Overview

This week I fitted two distributions to 6,287 daily log-returns on the S&P 500 (January 2000 to December 2024): the Gaussian as the standard baseline, and the Student-t as an upgrade that allows for heavier tails. The goal was to find out whether the Gaussian is actually adequate for daily returns, and to put a number on how wrong its risk estimates are if it isn't.

---

## 2. The data

Figure 1 shows the full return series. I chose a 25-year window to cover four distinct market stress periods: the dot-com crash (2000–2002), the Global Financial Crisis (2007–2009), the COVID-19 shock (February–June 2020), and the Fed rate hike cycle (2022–2023). These are shaded in the figure.

The volatility is uneven. The 2012–2019 period looks almost flat against the GFC and COVID spikes. The single worst day in the sample, roughly −12% on 16 March 2020, barely looks like it belongs to the same series as a typical day in 2017.

![Figure 1. S&P 500 daily log-returns (2000–2024) with the four market shock periods shaded.](../figures/week2_trace.png)

*Figure 1. S&P 500 daily log-returns (2000–2024) with the four market shock periods shaded.*

---

## 3. Parameter estimates

I fitted both models by Maximum Likelihood Estimation (MLE). Tables 1 and 2 report the results.

**Table 1. Gaussian MLE results.**

| Parameter | Estimate | Standard Error | t-statistic |
|-----------|----------|----------------|-------------|
| μ (daily drift) | +0.000223 | 0.000154 | 1.45 |
| σ (scale) | 0.012234 | 0.000109 | — |
| Log-likelihood | 18,764.3 | — | — |
| AIC | −37,524.5 | — | — |
| BIC | −37,511.0 | — | — |

The Gaussian results are unsurprising. The daily drift of +0.022% is indistinguishable from zero (t-statistic 1.45), which is what you'd expect in an efficient market. Annualised volatility comes out at 19.4%, in line with long-run S&P 500 figures.

**Table 2. Student-t MLE results.**

| Parameter | Estimate | Standard Error | 95% CI |
|-----------|----------|----------------|--------|
| ν (degrees of freedom) | 2.648 | 0.110 | (2.43, 2.87) |
| μ (location) | +0.000656 | 0.000110 | — |
| σ (scale) | 0.007077 | 0.000128 | — |
| Log-likelihood | 19,666.7 | — | — |
| AIC | −39,327.5 | — | — |
| BIC | −39,307.2 | — | — |

The key result is ν̂ = 2.648. The confidence interval stays above 2 at the lower end, so the variance is technically finite. But 2.648 is below 3, which means skewness is formally undefined, and well below 4, which means kurtosis is formally undefined too. The data is consistent with a distribution of infinite kurtosis.

Worth noting: the Student-t σ of 0.0071 is not the standard deviation. The actual standard deviation is σ × √(ν / (ν − 2)), which works out to about 1.43% per day, higher than the Gaussian's 1.22%. The Student-t packs its spread into the tails rather than the centre.

---

## 4. Model fit

Figure 2 overlays both fitted PDFs on a histogram of the actual returns. The Gaussian is too flat at the centre and too spread across the moderate loss range. The Student-t fits the sharp central peak much better, and its tails sit higher at the extremes.

![Figure 2. Return histogram with Gaussian MLE (blue) and Student-t MLE (red) PDFs overlaid.](../figures/week2_density.png)

*Figure 2. Return histogram with Gaussian MLE (blue) and Student-t MLE (red) PDFs overlaid.*

The QQ plots make the difference clearest. A QQ plot sorts the data and compares each value to what the theoretical distribution predicts at that rank. Points on the diagonal mean a perfect fit.

Figure 3 shows the Gaussian QQ. The S-shape is the signature of fat tails: the actual worst daily loss plots at around −10 standardised units, while the Gaussian predicts roughly −3.5 at that rank. The worst day was more than three times as extreme as the model allows.

![Figure 3. QQ plot against N(0,1). The S-shape confirms fat tails in both directions.](../figures/week2_qq_gaussian.png)

*Figure 3. QQ plot against N(0,1). The S-shape confirms fat tails in both directions.*

Figure 4 shows the Student-t QQ. The fit is dramatically better across the central 99% of the data. The remaining departures are a handful of extreme observations at each end: 9/11, the Lehman collapse, the March 2020 crash. These look more like structural breaks than draws from a stable distribution.

![Figure 4. QQ plot against t(ν = 2.648). Near-perfect fit across the central 99%; residual departures correspond to major crisis events.](../figures/week2_qq_student_t.png)

*Figure 4. QQ plot against t(ν = 2.648). Near-perfect fit across the central 99%; residual departures correspond to major crisis events.*

**Table 3. Kolmogorov-Smirnov goodness-of-fit on standardised residuals.**

| Model | KS statistic (D) | p-value\* | n |
|-------|-----------------|----------|---|
| Gaussian | 0.094 | <0.001 | 6,287 |
| Student-t | 0.018 | 0.030 | 6,287 |

The KS statistic D backs up the QQ plots. The Student-t D of 0.018 is five times smaller than the Gaussian's 0.094. Both technically reject at n = 6,287, but the gap between them is what matters.

**Table 4. Model comparison by AIC and BIC. Lower is better.**

| Model | Parameters | Log-likelihood | AIC | BIC |
|-------|------------|---------------|-----|-----|
| Gaussian | 2 | 18,764.3 | −37,524.5 | −37,511.0 |
| Student-t | 3 | 19,666.7 | −39,327.5 | −39,307.2 |

The AIC difference is 1,803 units. AIC already penalises the Student-t for its extra parameter, so this isn't just a reward for complexity. Adding ν is worth 1,803 units of genuine fit improvement.

---

## 5. Risk measures

VaR tells you the loss threshold exceeded on the worst α% of days. ES tells you the average loss on those days.

**Table 5. Daily VaR and ES. Negative values = losses.**

| Confidence | VaR (Gaussian) | ES (Gaussian) | VaR (Student-t) | ES (Student-t) |
|------------|---------------|--------------|----------------|---------------|
| 95% | −1.990% | −2.501% | −1.694% | −3.000% |
| 99% | −2.824% | −3.238% | −3.515% | −5.813% |

At 95%, the Gaussian VaR looks more conservative than the Student-t (−1.99% vs −1.69%). That's because the Student-t's fitted σ is much smaller (0.71% vs 1.22%), which at moderate quantile depths outweighs the heavier-tail effect. But even at 95%, ES tells the opposite story: Student-t ES of −3.00% is 20% worse than the Gaussian's −2.50%. Once inside the tail, the heavier distribution wins.

At 99%, there is no crossover. Student-t ES is −5.81% against the Gaussian's −3.24%. A portfolio using Gaussian ES to set capital would hold roughly 44% too little (3.238 / 5.813 = 0.557). That's the number this whole week was building toward.

---

## 6. How crises change the distribution

When you fit each model separately to the four shock windows, the variation in tail behaviour is interesting.

**Table 6. Sub-period MLE results. ν̂ is fitted independently to each window.**

| Period | n | Gaussian σ (ann.) | Student-t ν̂ | Student-t σ (ann.) |
|--------|---|------------------|------------|-------------------|
| Full sample (2000–2024) | 6,287 | 19.4% | 2.648 | 11.2% |
| Dot-com crash | 671 | 23.5% | 6.526 | 19.7% |
| GFC | 378 | 38.3% | 2.607 | 23.0% |
| COVID-19 | 104 | 50.5% | 2.285 | 28.4% |
| Fed rate hikes | 501 | 19.5% | 6.525 | 16.3% |

The ν̂ values are where it gets interesting. The GFC and COVID periods return ν̂ around 2.3–2.6, extremely close to the boundary where variance becomes infinite. The dot-com crash and rate hike cycle return ν̂ around 6.5, close enough to the Gaussian that the difference barely matters. Evidence suggests crises are structurally different.

A counterintuitive feature of the crisis windows is that excess kurtosis falls even as variance rises sharply. In a sustained crisis, extreme daily moves are no longer outliers relative to the elevated variance, nearly every day is volatile, so the distribution spreads broadly rather than concentrating mass near the centre with a thin spike. The excess kurtosis that drives ν̂ down in the full sample comes largely from the contrast between tranquil and turbulent regimes; within a crisis window that contrast disappears, and the within-period distribution is wide but not especially peaked.

Figure 5 shows the annual return distributions for 2000–2024, with crisis years highlighted and annualised volatility labelled in each panel. 2008 (σ = 41%) and 2020 (σ = 35%) stand out clearly. The 2013–2019 stretch looks narrow by comparison, with σ as low as 7% in 2017. The dot-com years are elevated but noticeably more moderate than the GFC, which matches the higher ν̂.

![Figure 5. Annual return distributions, 2000–2024. Crisis years are highlighted; annualised volatility is shown in each panel.](../figures/week2_marginals_by_year.png)

*Figure 5. Annual return distributions, 2000–2024. Crisis years are highlighted; annualised volatility is shown in each panel.*

The full-sample ν̂ of 2.648 is effectively an average across all these regimes. In calm years it overstates tail risk; in GFC conditions it may still understate it.

---

## 7. Five-year block marginals

Figures 6–10 break the 25-year sample into five 5-year blocks, each showing five annual panels side by side. The legend adds two entries relative to Figure 5: an x = 0 reference line (grey dashed) marking zero return, and the year mean (black dotted). Shock-window background shading follows the same colour scheme as Figure 1. The inset annotation in each panel gives annualised σ and ν.

![Figure 6. Annual return distributions, 2000–2004. Dot-com crash years (2000–2002) are shaded.](../figures/week2_marginals_2000_2004.png)

*Figure 6. Annual return distributions, 2000–2004. Dot-com crash years (2000–2002) are shaded.*

![Figure 7. Annual return distributions, 2005–2009. GFC (2007–2009) panels are shaded; ν drops to 2.6 in 2008.](../figures/week2_marginals_2005_2009.png)

*Figure 7. Annual return distributions, 2005–2009. GFC (2007–2009) panels are shaded; ν drops to 2.6 in 2008.*

![Figure 8. Annual return distributions, 2010–2014. Post-crisis recovery; annualised volatility declines across the block.](../figures/week2_marginals_2010_2014.png)

*Figure 8. Annual return distributions, 2010–2014. Post-crisis recovery; annualised volatility declines across the block.*

![Figure 9. Annual return distributions, 2015–2019. 2017 shows the narrowest spread in the full sample (σ = 7%).](../figures/week2_marginals_2015_2019.png)

*Figure 9. Annual return distributions, 2015–2019. 2017 shows the narrowest spread in the full sample (σ = 7%).*

![Figure 10. Annual return distributions, 2020–2024. The 2020 COVID panel produces σ = 35% and ν near 2.3.](../figures/week2_marginals_2020_2024.png)

*Figure 10. Annual return distributions, 2020–2024. The 2020 COVID panel produces σ = 35% and ν near 2.3.*
