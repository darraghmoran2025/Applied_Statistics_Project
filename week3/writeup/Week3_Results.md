# Week 3 results: Laplace, Variance-Gamma and NIG MLE

## 1. Overview

This week I extended the comparison to five distributional models on the same 6,287 daily S&P 500 log-returns used in Week 2: Laplace (double-exponential), Variance-Gamma (VG), and Normal Inverse Gaussian (NIG) alongside the Gaussian and Student-t. The Laplace is a two-parameter benchmark (the symmetric VG special case, θ = 0, ν = 1) and with the same parameter count as the Gaussian it captures 96% of the AIC gain the four-parameter NIG achieves. Most of the improvement over the Gaussian comes from exponential rather than thin tails. NIG is the best-fitting model and the only one not rejected by KS; its 99% ES of −5.19% sits between the Gaussian (−3.24%) and the Student-t (−5.81%). The sub-period analysis shows the GFC and COVID were structurally different crises from the dot-com crash and the rate-hike cycle.

---

## 2. Parameter estimates

**Table 1. Gaussian MLE (Week 2).**

| Parameter | Estimate | Standard Error | t-statistic |
|-----------|----------|----------------|-------------|
| μ (daily drift) | +0.000223 | 0.000154 | 1.45 |
| σ (scale) | 0.012234 | 0.000109 |  |
| Log-likelihood | 18,764.3 |  |  |
| AIC | −37,524.5 |  |  |
| BIC | −37,511.0 |  |  |

**Table 2. Laplace MLE (symmetric VG special case: θ = 0, ν = 1).**

| Parameter | Estimate | Standard Error |
|-----------|----------|----------------|
| μ (location) | +0.000602 | 0.000102 |
| b (scale) | 0.008078 | 0.000102 |
| Log-likelihood | 19,649.9 |  |
| AIC | −39,295.7 |  |
| BIC | −39,282.2 |  |

Its log-likelihood beats the Gaussian by 885.6 and falls only 15.6 short of the full VG despite fixing θ = 0 and ν = 1; the exponential-tail gain alone dwarfs the benefit of freeing those two parameters.

**Table 3. Student-t MLE (Week 2).**

| Parameter | Estimate | Standard Error | 95% CI |
|-----------|----------|----------------|--------|
| ν (degrees of freedom) | 2.648 | 0.110 | (2.43, 2.87) |
| μ (location) | +0.000656 | 0.000110 |  |
| σ (scale) | 0.007077 | 0.000128 |  |
| Log-likelihood | 19,666.7 |  |  |
| AIC | −39,327.5 |  |  |
| BIC | −39,307.2 |  |  |

**Table 4. Variance-Gamma MLE.**

| Parameter | Estimate | Standard Error |
|-----------|----------|----------------|
| σ (scale) | 0.011629 | 0.000163 |
| θ (asymmetry) | −0.000660 | 0.000149 |
| ν (variance rate) | 1.17431 | 0.043159 |
| μ (location) | +0.000883 | 0.000023 |
| Log-likelihood | 19,665.5 |  |
| AIC | −39,322.9 |  |
| BIC | −39,296.0 |  |

VG's log-likelihood nearly matches the Student-t's despite two extra parameters, giving slightly worse AIC. Negative θ confirms left skew; ν = 1.174 confirms the Gamma time-change is active.

**Table 5. Normal Inverse Gaussian MLE.**

| Parameter | Estimate | Standard Error |
|-----------|----------|----------------|
| α (tail-heaviness) | 52.341 | 2.786 |
| β (asymmetry) | −6.095 | 1.498 |
| δ (scale) | 0.007574 | 0.000217 |
| μ (location) | +0.001111 | 0.000150 |
| Log-likelihood | 19,691.7 |  |
| AIC | −39,375.3 |  |
| BIC | −39,348.4 |  |

NIG achieves the highest log-likelihood and an AIC improvement of 1,851 units over the Gaussian. Negative β confirms left skew; α falls to 17–26 in the GFC and COVID (Section 5): far heavier tails than the full-sample 52.

---

## 3. Model fit

Figure 1 overlays all five PDFs on the return histogram; the four non-Gaussian curves look similar to the eye, with Laplace and Student-t nearly indistinguishable at this scale. Figure 2's QQ plots show the difference more precisely: NIG and VG sit closest to the diagonal throughout, while the Laplace's tails are slightly too light at the most extreme points.

![Figure 1. Return histogram with all five fitted PDFs. Gaussian (blue), Laplace (purple, dashed), Student-t (red), VG (green), NIG (orange).](../figures/week3_density_all_models.png)

*Figure 1. Return histogram with all five fitted PDFs. Gaussian (blue), Laplace (purple, dashed), Student-t (red), VG (green), NIG (orange).*

![Figure 2. QQ plots for all five models. The Gaussian S-shape (top left) is severe. NIG (bottom row) sits closest to the diagonal.](../figures/week3_qq_all_models.png)

*Figure 2. QQ plots for all five models. The Gaussian S-shape (top left) is severe. NIG (bottom row) sits closest to the diagonal.*

**Table 6. Kolmogorov-Smirnov goodness-of-fit.**

| Model | KS statistic (D) | p-value | n |
|-------|-----------------|---------|---|
| Gaussian | 0.0938 | <0.001 | 6,287 |
| Laplace | 0.0180 | 0.034 | 6,287 |
| Student-t | 0.0182 | 0.030 | 6,287 |
| VG | 0.0129 | 0.250 | 6,287 |
| NIG | 0.0113 | 0.399 | 6,287 |

NIG is the only model not rejected by KS; VG passes too. Laplace and Student-t tie on D (0.0180 vs 0.0182) and are borderline. KS for VG and NIG uses the two-sample test against 1,000,000 simulated draws.

**Table 7. Model comparison. Lower AIC/BIC is better.**

| Model | Parameters | Log-lik | AIC | BIC | ΔAIC vs Gaussian |
|-------|------------|---------|-----|-----|------------------|
| Gaussian | 2 | 18,764.3 | −37,524.5 | −37,511.0 |  |
| Laplace | 2 | 19,649.9 | −39,295.7 | −39,282.2 | −1,771.2 |
| Student-t | 3 | 19,666.7 | −39,327.5 | −39,307.2 | −1,802.9 |
| VG | 4 | 19,665.5 | −39,322.9 | −39,296.0 | −1,798.4 |
| NIG | 4 | 19,691.7 | −39,375.3 | −39,348.4 | −1,850.8 |

The Laplace reaches ΔAIC −1,771 with two parameters (96% of NIG's gain), confirming exponential tails account for most of the improvement over the Gaussian. VG is slightly worse than the three-parameter Student-t on both AIC and BIC; NIG gains 26 log-likelihood units over VG, meaning the Inverse Gaussian mixing distribution captures something the Gamma cannot.

---

## 4. Risk measures

**Table 8. Daily VaR and ES for all five models. Negative values = losses.**

| Confidence | VaR (Gauss.) | ES (Gauss.) | VaR (Lap) | ES (Lap) | VaR (t) | ES (t) | VaR (VG) | ES (VG) | VaR (NIG) | ES (NIG) |
|------------|-------------|------------|-----------|----------|---------|--------|----------|---------|-----------|----------|
| 95% | −1.990% | −2.501% | −1.800% | −2.608% | −1.694% | −3.000% | −1.915% | −2.820% | −1.880% | −3.075% |
| 97.5% | −2.376% | −2.838% | −2.360% | −3.168% | −2.366% | −4.020% | −2.542% | −3.447% | −2.646% | −3.941% |
| 99% | −2.824% | −3.238% | −3.100% | −3.908% | −3.515% | −5.813% | −3.363% | −4.283% | −3.769% | −5.187% |

At 99% ES the five models span from −3.24% (Gaussian) to −5.81% (Student-t). The Laplace at −3.91% already exceeds the Gaussian by 21% from exponential tails alone. The Student-t's outlier reading reflects its symmetric constraint at ν = 2.648, which inflates the right tail to match the left. VG and NIG, with asymmetry parameters, sit at −4.28% and −5.19%. Using the Gaussian ES to set capital leaves a 37% shortfall against NIG. Under FRTB (BCBS 2013), that shortfall is material.

The 97.5% row is the one FRTB mandates: the Basel III/IV internal-models framework sets regulatory capital from ES at 97.5% confidence. At that level the Gaussian ES of −2.84% sits 28% below the NIG's −3.94% and 29% below the Student-t's −4.02%. The Gaussian shortfall is already material at the confidence level the regulation uses, well before the deep 99% tail.

![Figure 3. VaR and ES at 95%, 97.5% and 99% for all five models. Hatched bars are ES.](../figures/week3_risk_comparison.png)

*Figure 3. VaR and ES at 95%, 97.5% (the FRTB ES confidence level) and 99% for all five models. Hatched bars are ES.*

---

## 5. How crises change the distribution

**Table 9. Sub-period parameter estimates. Each window fitted independently.**

| Period | n | Gauss. σ (ann.) | t ν | VG ν | VG θ | NIG α | NIG β |
|--------|---|-----------------|-----|------|------|-------|-------|
| Full sample | 6,287 | 19.4% | 2.648 | 1.174 | −0.00066 | 52.3 | −6.09 |
| Dot-com crash | 671 | 23.5% | 6.526 | 0.397 | +0.00190 | 98.6 | +10.04 |
| GFC | 378 | 38.3% | 2.607 | 1.250 | −0.00297 | 25.9 | −2.74 |
| COVID-19 | 104 | 50.5% | 2.285 | 1.910 | −0.00495 | 17.8 | −4.17 |
| Fed rate hikes | 501 | 19.5% | 6.525 | 0.493 | −0.00002 | 111.9 | −3.85 |

The GFC and COVID return Student-t ν around 2.3–2.6, near the variance singularity at ν = 2, and NIG α falls to 17–26. The dot-com crash and rate-hike cycle show ν around 6.5 and NIG α near 99–112, near-Gaussian. The two crisis types differ in kind: the dot-com was a slow drawdown; the GFC and COVID were clusters of extreme single-day moves, invisible in volatility alone but clear in the tail parameters.

**Table 10. AIC improvement over Gaussian by period. Positive = Lévy model preferred.**

| Period | ΔAIC (Student-t) | ΔAIC (VG) | ΔAIC (NIG) |
|--------|------------------|-----------|------------|
| Full sample | 1,802.9 | 1,798.4 | 1,850.8 |
| Dot-com crash | 22.7 | 22.3 | 23.0 |
| GFC | 72.7 | 78.7 | 76.4 |
| COVID-19 | 19.7 | 23.1 | 20.9 |
| Fed rate hikes | 16.0 | 16.2 | 15.1 |

The GFC improvement is 72–79 units; calm periods are 15–23. Lévy models earn their keep exactly when markets are worst.

![Figure 4. Left: annualised scale by period and model. Right: tail parameter ν for Student-t and VG. The dashed line is the ν = 2 variance singularity.](../figures/week3_subperiod_params.png)

*Figure 4. Left: annualised scale by period and model. Right: tail parameter ν for Student-t and VG. The dashed line is the ν = 2 variance singularity.*

![Figure 5. AIC improvement of each Lévy model over the Gaussian by period.](../figures/week3_subperiod_aic.png)

*Figure 5. AIC improvement of each Lévy model over the Gaussian by period.*
