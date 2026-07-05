# Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

*Master writeup, Weeks 1 to 7 combined. Compiled from the committed weekly markdown sources; render to Word with `generate_master_doc.py`.*

## Abstract

Daily stock returns are not Gaussian, and the difference is expensive. In this project I fit five distributional models to 6,287 daily S&P 500 log-returns from January 2000 to December 2024: the Gaussian benchmark that underlies Black-Scholes, the Laplace, the Student-t, and two Lévy models built by Brownian subordination, the Variance-Gamma (Madan, Carr and Chang, 1998) and the Normal Inverse Gaussian. I estimated everything twice, by maximum likelihood and again in a Bayesian setting with the No-U-Turn Sampler, so the headline parameters come with full posteriors rather than point estimates. The fitted Student-t degrees of freedom is 2.648, with a 94% credible interval of (2.46, 2.87) sitting entirely above the variance singularity at 2: the tails are heavy, and the variance is finite. At the 99% level the Gaussian understates Expected Shortfall by 79.5% against the Student-t and by 37% against the NIG, the only model that survived every goodness-of-fit and posterior predictive check I ran. A rolling out-of-sample backtest of one-day Value-at-Risk over 5,787 days, scored with the Christoffersen (1998) conditional coverage framework, confirms both halves of the story: the Gaussian produces 2.6 times the nominal number of 99% violations while the NIG cuts the excess to 1.6, yet every model, at every level, fails the independence test, because no static distribution can time the volatility clustering documented by Cont (2001). Choosing a heavier marginal therefore improves the measurement of tail risk by amounts that matter under the FRTB metrics (BCBS, 2013); what it cannot buy is the timing of that risk, which belongs to dynamics.

---

# Week 1: Literature Review

*Applied Statistical Research in Financial Market Risk Modelling*
*Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns*

## 1. Introduction

Since the early 1970s, quantitative finance has run on one big assumption: daily stock returns are normally distributed. Black-Scholes is built on it. Most of classical financial econometrics depends on it. The problem is that it is wrong in ways that matter. Real returns have fatter tails and sharper peaks than any Gaussian, and they skew negative, crashes hit harder and more often than the equivalent rally.

This review makes the case for replacing the Gaussian baseline with Lévy processes, specifically the Variance-Gamma (VG) process (Madan, Carr and Chang, 1998). It also covers the empirical evidence motivating the switch (Cont, 2001) and the backtesting framework used to assess whether a better distributional fit actually improves risk estimates in practice (Christoffersen, 1998).

## 2. Empirical Failures of the Gaussian Model

### 2.1 Stylised Facts of Financial Returns

Cont (2001) is the standard reference for what financial return data actually looks like. He surveys equities, foreign exchange, and commodities and finds the same statistical patterns recurring across all of them. The paper is worth reading carefully, not because the individual facts are surprising, but because Cont is rigorous about which properties are truly universal and which are asset-class specific.

The facts most relevant to this project are:

- **Excess kurtosis.** Daily log-returns have far more mass in the tails and centre than a Gaussian. Cont (2001) reports excess kurtosis of around 15.95 for S&P 500 futures at the five-minute scale, with higher figures for foreign exchange. Extreme moves happen much more often than the normal distribution predicts.
- **Gain/loss asymmetry.** Large falls in equity prices are more common than equivalently large rises. Cont (2001) notes this is specific to equities; foreign exchange markets show no such directional asymmetry. That is worth keeping in mind when interpreting skewness estimates.
- **Heavy tails.** Using extreme value theory, Cont (2001) puts the tail index between two and five for most markets studied. This rules out the normal distribution and also rules out stable laws with infinite variance. The distribution has finite variance, but the tails are heavy enough that kurtosis is large.
- **Volatility clustering.** Raw returns show no autocorrelation, but squared and absolute returns do, and the autocorrelation decays slowly. This is the main mechanism by which VaR violations cluster in practice. This is what Christoffersen (1998) builds his independence test to detect.
- **Aggregational Gaussianity.** At longer horizons, returns look increasingly normal. The non-Gaussian behaviour is a daily phenomenon, which is why this project focuses on daily data.

One of Cont's more useful observations is structural: to fit all of the above properties, a parametric model needs at least four parameters, location, scale, tail decay, and asymmetry. That rules out the Gaussian (two parameters) and the Student-t (three parameters). The four-parameter Lévy models are the natural next step.

### 2.2 Evidence from Major Stress Episodes

The S&P 500 dataset covers the dot-com crash (2000–2002), the Global Financial Crisis (2007–2009), the COVID shock (March 2020), and the Fed rate-hike cycle (2022–2023). In March 2020 alone, several daily moves exceeded five historical standard deviations. Under a Gaussian model, each one of those is essentially impossible. Under a well-fitted Lévy model, they are unlikely but not negligible, and that difference is exactly what this project is trying to quantify.

## 3. Lévy Processes: Theoretical Framework

### 3.1 Definition and the Lévy–Khintchine Theorem

A Lévy process has three properties: independent increments, stationary increments, and càdlàg sample paths. Brownian motion is the simplest example, but the class is much broader. The Lévy–Khintchine theorem gives the full picture. Any Lévy process is characterised by a drift b, a Gaussian variance σ², and a Lévy measure ν on ℝ\{0}. The Lévy measure is where the interesting behaviour lives: it controls jump intensity and magnitude. Set ν to zero and you get Brownian motion with drift. Make it non-trivial and the process jumps, with frequency and size governed by ν. That is the financial point: jumps capture sudden large moves that continuous diffusion cannot produce by construction.

### 3.2 Subordination: Building Lévy Models from Brownian Motion

Madan, Carr and Chang (1998) construct the VG process by subordination: Brownian motion is evaluated not at calendar time but at a random time change T(t) that is itself a non-decreasing Lévy process. The result inherits independent and stationary increments from both components. Intuitively, the clock runs fast during volatile periods and slow during quiet ones. Madan, Carr and Chang (1998) tie this directly to economics: each unit of calendar time has a random effective length drawn from a gamma distribution, capturing variation in market activity. More practically, subordination yields closed-form density functions that can be fitted by maximum likelihood without simulation.

## 4. The Variance-Gamma Model

### 4.1 Construction and Parameters

The VG process was introduced by Madan and Seneta (1990) and given its definitive form in Madan, Carr and Chang (1998), "The Variance Gamma Process and Option Pricing." The construction subordinates Brownian motion with drift to a Gamma process. Let W(t) be standard Brownian motion, θ a drift, σ a volatility, and γ(t; 1, ν) a Gamma process with unit mean rate and variance rate ν. The VG process is:

X(t; σ, ν, θ) = θγ(t; 1, ν) + σW(γ(t; 1, ν))

The model has three parameters beyond the mean return. σ is volatility. θ drives skewness: negative θ gives the left skew seen in equities. ν controls tail heaviness. This satisfies Cont's structural requirement exactly.

The parameter interpretations are precise. Madan, Carr and Chang (1998) derive the central moments explicitly. The third central moment is proportional to (2θ³ν² + 3σ²θν)t, so θ = 0 implies zero skewness and the sign of θ determines the direction of skew. When θ = 0, ν is directly the annualised percentage excess kurtosis: ν is quoted per unit of annual time, so at the daily horizon the kurtosis is 3(1 + 365ν). All moments are finite, which means VaR and Expected Shortfall are well-defined under the model.

Unlike Brownian motion, the VG process has no continuous martingale component: it is a pure jump process of finite variation. It can be written as the difference of two independent increasing gamma processes, one for upward moves and one for downward. The relative sizes of those two processes determine the skew. The VG process also inherits from the gamma process an infinite arrival rate of small jumps, which is what makes it behave like Brownian motion at fine scales while still allowing occasional large discontinuities. Madan, Carr and Chang (1998) argue this makes the VG more realistic than Poisson jump-diffusion models, which require a finite jump rate and therefore need a separate diffusion component.

### 4.2 Estimation and Empirical Results

Madan, Carr and Chang (1998) estimated the VG model on S&P 500 data from January 1992 to September 1994 by maximum likelihood. The density involves a modified Bessel function of the second kind, available in standard numerical libraries, so MLE is a straightforward L-BFGS-B optimisation. The results are useful for setting expectations.

On the statistical process, the lognormal model is strongly rejected against the symmetric VG with a χ²(1) statistic of 83.94. The skewness parameter θ is statistically insignificant, the S&P 500 daily return distribution is approximately symmetric. Excess kurtosis is the real story: the estimated ν = 0.002 (in annual units) corresponds to a daily kurtosis of 3(1 + 0.002 × 365) ≈ 5.19, against 3 for the Gaussian. Adding the full asymmetric VG on top of the symmetric one makes no improvement for the statistical process.

The risk-neutral estimation is different. Inverting the VG option pricing formula across 143 weeks of S&P 500 futures options gives an average θ = −0.14, strong negative skew, and much higher kurtosis than in the statistical process. Both are consistent with risk aversion: investors pay a premium for protection against large negative jumps. The sign and magnitude of the risk-neutral skew is a direct function of risk aversion in the equilibrium framework.

The option pricing results also matter. On orthogonality tests of pricing errors, the full VG achieves an adjusted R² of 0.001, against 0.161 for Black-Scholes and 0.171 for the symmetric VG. VG pricing errors are essentially unpredictable from moneyness or maturity. The Black-Scholes model is rejected in favour of the VG in 91.6% of the 143 weeks at the 1% level.

## 5. Risk Estimation and VaR Backtesting

### 5.1 Value at Risk and Expected Shortfall

VaR at confidence level α is the loss not exceeded on (1 − α) of trading days. Expected Shortfall is the expected loss on days that do exceed VaR. ES has replaced VaR as the primary regulatory metric under Basel III/IV at 97.5% confidence. For the Gaussian, both are in closed form. For the VG, they require numerical integration over the fitted density, which is tractable given the closed-form density derived in Madan, Carr and Chang (1998). The question this project will answer concretely: by how many basis points does the Gaussian underestimate 99th-percentile daily loss relative to the VG?

### 5.2 Christoffersen's Conditional Coverage Framework

Christoffersen (1998), "Evaluating Interval Forecasts," built the standard framework for testing whether a VaR model is actually well-calibrated. The paper's starting point is a criticism of existing practice: most of the literature tests only whether violations occur at the right average rate, and ignores whether those violations cluster in time. A model can pass the average-rate test while systematically underestimating risk during volatile periods. That is precisely when accurate risk estimates matter most.

The paper introduces three likelihood ratio tests. The unconditional coverage test (LR_uc) asks whether E[I_t] = p, where I_t indicates a violation on day t. This is asymptotically χ²(1). The independence test (LR_ind) models the violation sequence as a first-order Markov chain and tests whether a violation today is more likely given a violation yesterday. Also χ²(1). The conditional coverage test (LR_cc = LR_uc + LR_ind) combines both, distributed χ²(2). A model that gets the average rate right but clusters its failures passes the UC test and fails the independence test.

The empirical application in Christoffersen (1998) uses daily exchange rate data and compares three forecasts: static empirical quantiles, the J.P. Morgan RiskMetrics IGARCH model, and a GARCH(1,1) with Student-t innovations. Static forecasts consistently pass the UC test and fail the independence test because they cannot adapt to volatility clustering. The GARCH-t model passes the independence test across most coverage rates. The lesson is not that Gaussian models fail on average, but that they fail when it matters, in runs during stress periods.

In this project, a rolling estimation window computes one-day-ahead 99% VaR under each model at each step across the 2000–2024 sample. The UC and CC tests are then applied to the violation sequences. Given that the dataset includes four distinct crisis sub-periods, the independence test should be particularly informative: a model with a Gaussian-shaped tail will produce violation clusters that are hard to miss.

## 6. Synthesis

These three papers give this project its structure. Cont (2001) establishes what the data looks like and sets a concrete requirement: four-parameter distributions with separate control over location, scale, tail decay, and asymmetry. Madan, Carr and Chang (1998) show that the VG process meets those requirements, derive the density formulas needed for MLE, and validate the model on S&P 500 data, finding that the statistical process is approximately symmetric but that excess kurtosis is highly significant, with daily kurtosis around 5.19. The Black-Scholes model is rejected in over 90% of weeks in their option pricing tests. Christoffersen (1998) shows how to test whether a better distributional fit actually translates into better risk estimates, and why the independence of violations matters as much as the average violation rate.

What this project adds is a side-by-side comparison of the Gaussian and Student-t baselines against the VG model on a single 25-year dataset from 2000 to 2024, one that contains an unusually dense collection of stress events. The sub-period breakdowns and rolling backtest together give a fuller picture of where each model holds and where it breaks than any of the original papers attempt.

## Bibliography

Christoffersen, P. F. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.

Cont, R. (2001). Empirical Properties of Asset Returns: Stylised Facts and Statistical Issues. *Quantitative Finance*, 1(2), 223–236.

Madan, D. B., Carr, P. P. and Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *European Finance Review*, 2(1), 79–105.

---

# Week 2 results: Gaussian and Student-t MLE

## 1. Overview

This week I fitted two distributions to 6,287 daily log-returns on the S&P 500 (January 2000 to December 2024): the Gaussian as the standard baseline, and the Student-t as an upgrade that allows for heavier tails. The goal was to find out whether the Gaussian is actually adequate for daily returns, and to put a number on how wrong its risk estimates are if it isn't.

---

## 2. The data

Figure 1 shows the full return series. I chose a 25-year window to cover four distinct market stress periods: the dot-com crash (2000–2002), the Global Financial Crisis (2007–2009), the COVID-19 shock (February–June 2020), and the Fed rate hike cycle (2022–2023). These are shaded in the figure.

The volatility is uneven. The 2012–2019 period looks almost flat against the GFC and COVID spikes. The single worst day in the sample, roughly −12% on 16 March 2020, barely looks like it belongs to the same series as a typical day in 2017.

![Figure 1. S&P 500 daily log-returns (2000–2024) with the four market shock periods shaded.](week2/figures/week2_trace.png)

*Figure 1. S&P 500 daily log-returns (2000–2024) with the four market shock periods shaded.*

---

## 3. Parameter estimates

I fitted both models by Maximum Likelihood Estimation (MLE). Tables 1 and 2 report the results.

**Table 1. Gaussian MLE results.**

| Parameter | Estimate | Standard Error | t-statistic |
|-----------|----------|----------------|-------------|
| μ (daily drift) | +0.000223 | 0.000154 | 1.45 |
| σ (scale) | 0.012234 | 0.000109 |  |
| Log-likelihood | 18,764.3 |  |  |
| AIC | −37,524.5 |  |  |
| BIC | −37,511.0 |  |  |

The Gaussian results are unsurprising. The daily drift of +0.022% is indistinguishable from zero (t-statistic 1.45), which is what you'd expect in an efficient market. Annualised volatility comes out at 19.4%, in line with long-run S&P 500 figures.

**Table 2. Student-t MLE results.**

| Parameter | Estimate | Standard Error | 95% CI |
|-----------|----------|----------------|--------|
| ν (degrees of freedom) | 2.648 | 0.110 | (2.43, 2.87) |
| μ (location) | +0.000656 | 0.000110 |  |
| σ (scale) | 0.007077 | 0.000128 |  |
| Log-likelihood | 19,666.7 |  |  |
| AIC | −39,327.5 |  |  |
| BIC | −39,307.2 |  |  |

The key result is ν̂ = 2.648. The confidence interval stays above 2 at the lower end, so the variance is technically finite. But 2.648 is below 3, which means skewness is formally undefined, and well below 4, which means kurtosis is formally undefined too. The data is consistent with a distribution of infinite kurtosis.

Worth noting: the Student-t σ of 0.0071 is not the standard deviation. The actual standard deviation is σ × √(ν / (ν − 2)), which works out to about 1.43% per day, higher than the Gaussian's 1.22%. The Student-t packs its spread into the tails rather than the centre.

---

## 4. Model fit

Figure 2 overlays both fitted PDFs on a histogram of the actual returns. The Gaussian is too flat at the centre and too spread across the moderate loss range. The Student-t fits the sharp central peak much better, and its tails sit higher at the extremes.

![Figure 2. Return histogram with Gaussian MLE (blue) and Student-t MLE (red) PDFs overlaid.](week2/figures/week2_density.png)

*Figure 2. Return histogram with Gaussian MLE (blue) and Student-t MLE (red) PDFs overlaid.*

The QQ plots make the difference clearest. A QQ plot sorts the data and compares each value to what the theoretical distribution predicts at that rank. Points on the diagonal mean a perfect fit.

Figure 3 shows the Gaussian QQ. The S-shape is the signature of fat tails: the actual worst daily loss plots at around −10 standardised units, while the Gaussian predicts roughly −3.5 at that rank. The worst day was more than three times as extreme as the model allows.

![Figure 3. QQ plot against N(0,1). The S-shape confirms fat tails in both directions.](week2/figures/week2_qq_gaussian.png)

*Figure 3. QQ plot against N(0,1). The S-shape confirms fat tails in both directions.*

Figure 4 shows the Student-t QQ. The fit is dramatically better across the central 99% of the data. The remaining departures are a handful of extreme observations at each end: 9/11, the Lehman collapse, the March 2020 crash. These look more like structural breaks than draws from a stable distribution.

![Figure 4. QQ plot against t(ν = 2.648). Near-perfect fit across the central 99%; residual departures correspond to major crisis events.](week2/figures/week2_qq_student_t.png)

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

![Figure 5. Annual return distributions, 2000–2024. Crisis years are highlighted; annualised volatility is shown in each panel.](week2/figures/week2_marginals_by_year.png)

*Figure 5. Annual return distributions, 2000–2024. Crisis years are highlighted; annualised volatility is shown in each panel.*

The full-sample ν̂ of 2.648 is effectively an average across all these regimes. In calm years it overstates tail risk; in GFC conditions it may still understate it.

---

## 7. Five-year block marginals

Figures 6–10 break the 25-year sample into five 5-year blocks, each showing five annual panels side by side. The legend adds two entries relative to Figure 5: an x = 0 reference line (grey dashed) marking zero return, and the year mean (black dotted). Shock-window background shading follows the same colour scheme as Figure 1. The inset annotation in each panel gives annualised σ and ν.

![Figure 6. Annual return distributions, 2000–2004. Dot-com crash years (2000–2002) are shaded.](week2/figures/week2_marginals_2000_2004.png)

*Figure 6. Annual return distributions, 2000–2004. Dot-com crash years (2000–2002) are shaded.*

![Figure 7. Annual return distributions, 2005–2009. GFC (2007–2009) panels are shaded; ν drops to 2.6 in 2008.](week2/figures/week2_marginals_2005_2009.png)

*Figure 7. Annual return distributions, 2005–2009. GFC (2007–2009) panels are shaded; ν drops to 2.6 in 2008.*

![Figure 8. Annual return distributions, 2010–2014. Post-crisis recovery; annualised volatility declines across the block.](week2/figures/week2_marginals_2010_2014.png)

*Figure 8. Annual return distributions, 2010–2014. Post-crisis recovery; annualised volatility declines across the block.*

![Figure 9. Annual return distributions, 2015–2019. 2017 shows the narrowest spread in the full sample (σ = 7%).](week2/figures/week2_marginals_2015_2019.png)

*Figure 9. Annual return distributions, 2015–2019. 2017 shows the narrowest spread in the full sample (σ = 7%).*

![Figure 10. Annual return distributions, 2020–2024. The 2020 COVID panel produces σ = 35% and ν near 2.3.](week2/figures/week2_marginals_2020_2024.png)

*Figure 10. Annual return distributions, 2020–2024. The 2020 COVID panel produces σ = 35% and ν near 2.3.*

---

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

![Figure 1. Return histogram with all five fitted PDFs. Gaussian (blue), Laplace (purple, dashed), Student-t (red), VG (green), NIG (orange).](week3/figures/week3_density_all_models.png)

*Figure 1. Return histogram with all five fitted PDFs. Gaussian (blue), Laplace (purple, dashed), Student-t (red), VG (green), NIG (orange).*

![Figure 2. QQ plots for all five models. The Gaussian S-shape (top left) is severe. NIG (bottom row) sits closest to the diagonal.](week3/figures/week3_qq_all_models.png)

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

![Figure 3. VaR and ES at 95%, 97.5% and 99% for all five models. Hatched bars are ES.](week3/figures/week3_risk_comparison.png)

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

![Figure 4. Left: annualised scale by period and model. Right: tail parameter ν for Student-t and VG. The dashed line is the ν = 2 variance singularity.](week3/figures/week3_subperiod_params.png)

*Figure 4. Left: annualised scale by period and model. Right: tail parameter ν for Student-t and VG. The dashed line is the ν = 2 variance singularity.*

![Figure 5. AIC improvement of each Lévy model over the Gaussian by period.](week3/figures/week3_subperiod_aic.png)

*Figure 5. AIC improvement of each Lévy model over the Gaussian by period.*

---

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

![Figure 1. VIX against trailing 21-day realised volatility (both annualised), with OLS fit and 45-degree line.](week3/figures/week3_vix_scatter.png)

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

VIX level is the strongest factor (+0.40, t = 4.2): options-implied volatility is the market's own forward-looking dispersion estimate. ΔVIX (+0.12, t = 2.4) and absolute return (+0.05, t = 2.2) are also positive and significant. Drawdown carries a negative coefficient (−0.11, t = −1.8). The variable is at most zero and equals zero at the trailing peak, so the negative sign means the deeper the market sits below its peak, the higher forward volatility tends to be. This is the familiar leverage effect. Holding the other factors fixed, a market at its peak predicts lower forward volatility, which sits oddly beside the Section 3 finding that all four shocks began near a peak. There is no contradiction: Section 3 describes where those particular episodes started, while the coefficient is the average relationship across all 6,207 days. Rolling excess kurtosis is also negative (−0.04, t = −2.1); once volatility and VIX are controlled for, a kurtosis spike marks a jump that has already happened rather than risk still building. Rolling skewness is indistinguishable from zero.

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

![Figure 2. Forest plot of standardised factor loadings with Newey-West HAC 95% intervals. The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.](week3/figures/week3_leadup_coefs.png)

*Figure 2. Forest plot of the standardised factor loadings: each point is a coefficient estimate and the whisker its Newey-West HAC 95% interval, with factors ordered by effect size and a filled circle marking intervals that exclude zero (diamonds mark those that span it). The VIX level dominates; drawdown and rolling kurtosis carry negative loadings.*

![Figure 3. Left: actual forward 21-day realised volatility (black) against in-sample fitted values (red), with the four shock windows shaded. Right: fitted against actual, with the 45-degree line; in-sample R² = 0.55.](week3/figures/week3_leadup_fit.png)

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

![Figure 4. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.](week3/figures/week3_leadup_factors.png)

*Figure 4. Standardised trajectories of four representative factors (trailing volatility, rolling kurtosis, the 5-day VIX change, and drawdown), with the four shock windows shaded.*

---

## 4. Fitted Lévy parameters versus the VIX

Each calendar year from 2000 to 2024 is fitted separately, giving 25 annual estimates of each VG and NIG parameter, each then regressed on that year's average VIX (n = 25).

**Table 5. The fitted parameters and how each relates to the VIX.**

| Parameter | What it controls | Relationship with the VIX |
|-----------|------------------|---------------------------|
| VG σ (scale) | Width of the diffusion component | Strong and positive (R² = 0.90, p < 0.001): scale rises almost one-for-one with the VIX |
| VG θ (asymmetry) | Skew; negative means a heavier left tail | None detectable (R² = 0.01, p = 0.60) |
| VG ν (variance rate) | Tail heaviness from the random time-change | None detectable (R² = 0.06, p = 0.24) |
| NIG α (tail heaviness) | Larger means lighter tails | Weak and negative (R² = 0.12, p = 0.09): tails tend to get heavier as the VIX rises |
| NIG β (asymmetry) | Skew; negative means a heavier left tail | None detectable (R² = 0.08, p = 0.16) |
| NIG δ (scale) | Overall spread | None detectable (R² = 0.00, p = 0.95) |

The VIX tracks distribution scale very closely (VG σ, R² = 0.90) but carries almost no information about tail shape or asymmetry. Every tail and skew parameter is statistically flat against the VIX; only NIG α shows a marginal signal. Knowing the VIX tells you how wide the distribution will be, but little about how heavy or lopsided its tails are. That distinction matters for ES-based capital, since ES is driven by tail shape rather than scale.

![Figure 5. Annual VG and NIG parameter estimates against average annual VIX, with OLS regression lines.](week3/figures/week3_vix_regression.png)

*Figure 5. Annual VG and NIG parameter estimates (2000 to 2024) against that year's average VIX, with OLS regression lines; each panel shows its R² and p-value. Only VG σ (top left) has a strong relationship with the VIX; the tail and asymmetry parameters are essentially flat.*

---

## 5. Limitations

These results are in-sample. The predictors in the lead-up regression are correlated (most obviously trailing volatility and VIX), so coefficients are partial associations rather than independent effects. Overlapping forward windows make the effective sample smaller than n = 6,207; Newey-West SEs correct for this but not for the correlation between predictors. The 21-day horizon is a modelling choice: a shorter or longer window shifts the relative weight of fast factors (VIX change, absolute return) and slow ones (drawdown).

Two further caveats apply to the Section 4 regressions. The dependent variables are themselves estimates: each annual fit uses only around 250 observations, so the tail parameters in particular are noisy, and that noise inflates the residual variance and weakens any relationship the regression could find. The flat tail-parameter results are therefore partly a power problem. And with six parameters each tested against the VIX at n = 25, one marginal p-value (NIG α, p = 0.09) is about what chance alone would produce, so I read it as suggestive at most.

---

# Week 4: Bayesian estimation with NUTS

## 1. Bayesian Refitting

Weeks 2 and 3 fitted every model by maximum likelihood. That gives one best value for each parameter and a standard error read off the curvature of the likelihood. 

For this week I refitted four models in a Bayesian setting and drew their full posteriors with the No-U-Turn Sampler (NUTS) in PyMC. My goal here is to know whether the posterior for ν stays on the finite-variance side of 2 or whether it leaks below.

The four models are the Gaussian, the Laplace, the Student-t and the NIG. The Laplace stands in for the Variance-Gamma family here. It is the symmetric VG with θ = 0 and ν = 1, and unlike the full VG it has a closed-form density, so it carries the heavy-tail idea without dragging in any special functions. The NIG keeps its full four-parameter density.

---

## 2. Choosing the priors

A Bayesian fit needs a prior on each parameter, I made my selections based on their level of informativeness. That is, weakly informative priors. This is due to the loose 6,287 daily returns, but tight enough to rule out values that are not sensible. I started each prior from Week 3 MLE and then widened it well beyond the plausible range.

**Table 1. Priors and the reasoning behind each.**

| Parameter | Prior | Reasoning |
|-----------|-------|-----------|
| μ (all models) | Normal(0, 0.001) | Daily drift is tiny and barely identified; centred at zero, wide enough to cover any realistic value |
| σ, b, δ (scales) | HalfNormal(0.02) | Must be positive; 0.02 is generous next to a daily volatility near one percent |
| ν (Student-t) | Gamma(2, 0.1) | Mean 20, with a long right tail. It leans toward light tails, so any pull down toward ν ≈ 2.6 has to come from the data |
| γ (NIG) | Gamma(4, 4/52) | Mean 52, standard deviation 26; sets the tail decay and fixes α through α = √(γ² + β²) |
| β (NIG) | Normal(0, 15) | Asymmetry, free to take either sign |

The ν prior is the one I thought about most. It would have been easy to put a prior that only allows ν > 2 and guarantees a finite variance. I deliberately did not make that the default. Forcing ν above 2 would answer the exact question I am asking before the data gets a vote. Instead I used a prior that, if anything, prefers larger and lighter-tailed values, and let the likelihood pull ν down on its own. I kept the finite-variance version as an option in the code for anyone who needs that constraint, but it is not what produced these results.

The NIG needed a different kind of care. Its parameters obey α > |β|, and sampling two numbers under an inequality like that is awkward and tends to trip the sampler up at the boundary. I rewrote the model in terms of γ = √(α² − β²) and β, then recovered α = √(γ² + β²) afterwards. Now the inequality holds automatically and NUTS gets a clean, unconstrained space to explore.

---

## 3. Checking the priors before sampling

A prior can read sensibly in a table and imply correct results once ran through the model, the most accurate way to decide is through simulation. Before any sampling I drew returns straight from each prior and looked at what came out. This prior predictive check is how I actually decided the priors were acceptable, not the table above.

Figures 1 to 4 show the four. The shared feature is the one I was after: almost all the simulated mass sits inside the worst move the market actually produced, marked by the red lines at ±12.8%, with only thin tails past it. The differences between them matter too. The Gaussian prior is the tightest, its 99.9% range inside ±9.7%. The Student-t and Laplace reach further, to roughly ±12.4% and ±16.7%, because their priors permit heavier tails. The NIG is the only lopsided one, its left tail stretching to about −18.6% against +16.0% on the right, which is the asymmetry its β parameter exists to capture. So the priors are not just harmless. They already carry the tail character each model is meant to bring, before the data has said a word.

![Figure 1. Prior predictive returns for the Gaussian.](week4/figures/week4_prior_predictive_gaussian.png)

*Figure 1. Prior predictive returns for the Gaussian. The grey histogram is daily returns drawn from the prior alone, before the data is seen; the red lines mark the largest move in the actual sample (±12.8%). This is the tightest of the four.*

![Figure 2. Prior predictive returns for the Laplace.](week4/figures/week4_prior_predictive_laplace.png)

*Figure 2. Prior predictive returns for the Laplace. The symmetric tails reach further than the Gaussian, out to about ±16.7% at the 99.9% level.*

![Figure 3. Prior predictive returns for the Student-t.](week4/figures/week4_prior_predictive_student_t.png)

*Figure 3. Prior predictive returns for the Student-t. Tails sit between the Gaussian and the Laplace, reaching about ±12.4%.*

![Figure 4. Prior predictive returns for the NIG.](week4/figures/week4_prior_predictive_nig.png)

*Figure 4. Prior predictive returns for the NIG. The one asymmetric prior: its left tail stretches further than its right, the skew the β parameter allows.*

The fraction of simulated days beyond ±25%, a move bigger than anything in the sample, was 0.0001% for the Gaussian, 0.02% for the Student-t, 0.03% for the Laplace and 0.08% for the NIG. All four are negligible. Had any come back fat with absurd moves, I would have tightened the scale or tail prior and tried again. None did, so I sampled.

---

## 4. Sampling and convergence

I ran four chains for each model, with 1,000 tuning steps and 1,000 kept draws per chain. NUTS works by following the gradient of the log-density, which is straightforward for the Gaussian, Laplace and Student-t. The NIG was the real task. Its density contains a modified Bessel function, K₁, and the sampler needs the derivative of that. PyTensor supplies it, so I could code the NIG density directly instead of introducing the latent time-change variable the distribution is built from. I checked my version of the density against the one in scipy and they matched, which gave me confidence the gradient it was differentiating was the right one.

Every model converged cleanly. R-hat sat at 1.00 across the board, and the effective sample size never fell below about 2,000. Figures 5 to 8 show the traces. In each one the chains sit on top of each other on the left, and the raw draws on the right form a flat, stationary band with no drift and no chain wandering off, which is what good mixing looks like. The Gaussian and Laplace settle fastest, as their two-parameter posteriors should. The Student-t panel is the one I watch most closely, because it shows the ν chains circling steadily around 2.66 rather than creeping toward the ν = 2 boundary. The NIG, with four parameters and a Bessel function in its density, mixes just as well as the rest.

![Figure 5. NUTS trace for the Gaussian.](week4/figures/week4_trace_gaussian.png)

*Figure 5. NUTS diagnostics for the Gaussian. Left: the posterior for each parameter, one line per chain, with the Week 3 MLE dashed. Right: the raw chains.*

![Figure 6. NUTS trace for the Laplace.](week4/figures/week4_trace_laplace.png)

*Figure 6. NUTS diagnostics for the Laplace. Two parameters, fast and clean convergence.*

![Figure 7. NUTS trace for the Student-t.](week4/figures/week4_trace_student_t.png)

*Figure 7. NUTS diagnostics for the Student-t. The ν chains (bottom) circle around 2.66 and stay clear of the ν = 2 boundary.*

![Figure 8. NUTS trace for the NIG, the hardest of the four.](week4/figures/week4_trace_nig.png)

*Figure 8. NUTS diagnostics for the NIG. Four parameters and a custom Bessel-function density, and the chains still land on top of each other with flat, stationary traces.*

---

## 5. What the posteriors say

The posteriors come out almost exactly where the MLE did. With this much data and priors this loose, that is what should happen, and seeing it is reassuring rather than dull: it tells me the sampler and the MLE agree.

**Table 2. Posterior summaries against the Week 3 MLE. HDI (Highest Density Interval) is the 94% highest-density interval.**

| Model | Parameter | Posterior mean | 94% HDI | Week 3 MLE |
|-------|-----------|----------------|---------|------------|
| Gaussian | μ | +0.000215 | (−0.00007, +0.00049) | +0.000223 |
| | σ | 0.012233 | (0.01202, 0.01244) | 0.012234 |
| Laplace | μ | +0.000599 | (+0.00040, +0.00080) | +0.000602 |
| | b | 0.008081 | (0.00790, 0.00827) | 0.008078 |
| Student-t | μ | +0.000647 | (+0.00043, +0.00084) | +0.000656 |
| | σ | 0.007084 | (0.00686, 0.00733) | 0.007077 |
| | ν | 2.658 | (2.46, 2.87) | 2.648 |
| NIG | μ | +0.001080 | (+0.00081, +0.00135) | +0.001111 |
| | α | 52.35 | (47.1, 57.7) | 52.341 |
| | β | −5.86 | (−8.75, −3.25) | −6.095 |
| | δ | 0.007575 | (0.00717, 0.00798) | 0.007574 |

Two results are worth dwelling on.

The first is the Student-t ν. Its 94% interval runs from about 2.46 to 2.87, and the whole of it sits above 2. That is the question from Section 1 answered. The finite variance is not a fragile property of one point estimate that happens to land just north of the singularity. It holds across the entire credible range. The tails are genuinely heavy, and the variance is genuinely finite, both at once.

The second is the NIG asymmetry β. Every part of its interval is negative, from roughly −8.75 to −3.25. Week 3 reported a negative point estimate and read it as left skew. The posterior says that skew is a stable feature of the fit rather than something that could round away to zero. Losses sit in a heavier tail than gains, and the data is fairly firm about it.

---

## 6. Year-by-year VG and NIG fits

Everything so far, here and in Weeks 2 and 3, fits one set of parameters to the whole 2000–2024 sample. I refitted the Variance-Gamma and NIG densities by maximum likelihood one calendar year at a time, twenty-five fits each, and read the parameters as a time series.

Two choices shape these fits. First, the location is fixed at μ = 0. Within a single year there are only about 250 returns, the daily drift is tiny and barely identified, and leaving it free just adds a noisy nuisance parameter; pinning it to zero lets the scale, tail and asymmetry parameters carry the year's shape, and whatever small mean exists is absorbed by the asymmetry parameter (θ for VG, β for NIG), which is where it belongs. Second, the daily scale parameters σ and δ are reported annualised (× √252), so a year's scale reads on the same footing as the VIX and the Gaussian σ. The densities, the α > |β| reparametrisation and the standard errors are reused from the Week 3 code, so these yearly fits are numerically of a piece with the full-sample ones.

![Figure 9. Year-by-year Variance-Gamma parameters (μ = 0): annualised scale σ, asymmetry θ, and variance rate ν, with shock windows shaded and ±1 SE bars.](week4/figures/week4_yearly_vg_params.png)

*Figure 9. Year-by-year Variance-Gamma fit (μ = 0). Top to bottom: the annualised scale σ, the asymmetry θ, and the variance rate ν (tail heaviness). Shaded bands are the four shock windows; bars are ±1 standard error.*

The VG scale moves with the crises. σ peaks in 2008 at 39.8% annualised and again in 2020 at 31.3%, the GFC and COVID years, and bottoms out in the calm of 2017 at 6.5%. The variance rate ν, which sets the tail heaviness, tends to run higher through the turbulent 2008 to 2020 stretch than in the quiet mid-2000s or the 2021 to 2023 years, though it is a good deal noisier than the scale.

![Figure 10. Year-by-year NIG parameters (μ = 0): tail α and scale δ on log axes, asymmetry β with ±1 SE bars; shock windows shaded.](week4/figures/week4_yearly_nig_params.png)

*Figure 10. Year-by-year Normal-Inverse-Gaussian fit (μ = 0). Tail-heaviness α and annualised scale δ are on log axes; asymmetry β is linear with ±1 SE bars. In the very calm years 2004, 2005 and 2023 the NIG runs into its Gaussian limit, where α and δ both grow large and are only weakly identified individually (the SEs there are enormous, so the bars are omitted on the log panels); their product stays finite and the tails are simply close to normal.*

The NIG says the same thing through its tail. In exactly 2008 and 2020, is the heaviest tails of the whole sample land on the two largest crises. In the calmest years the NIG drifts toward the Gaussian. The asymmetry β is most negative, most left-skewed, in 2001 to 2002, in 2008 and in 2022, the dot-com unwind, the GFC and the rate-hike drawdown. That matches the full-sample posterior in Section 5, which put β firmly below zero.

The two models agree. Tail heaviness and scale are not fixed features of the market; they rise sharply in crises and settle back in calm years, and the negative skew bunches into the stressed periods. 

Figures 9 and 10 track the parameters; Figures 11 to 15 show the fitted densities they produce, one panel per year against that year's empirical distribution, in the same five-year-block layout as the Week 2 marginals. Each panel shades its background with the shock window the year belongs to, overlays the fitted VG (green) and NIG (orange dashed) densities on the empirical kernel estimate, and prints the year's annualised scale together with the two tail parameters. The crisis years stand out by eye: 2008 and 2020 have the sharpest peaks and the longest tails, the calm years are visibly closer to normal, and the two Lévy densities sit almost on top of each other throughout, which is why the choice between them turns on tail risk rather than overall shape.

![Figure 11. Per-year fitted VG and NIG densities, 2000–2004.](week4/figures/week4_marginals_2000_2004.png)

*Figure 11. Per-year empirical distribution (kernel estimate) with fitted VG and NIG densities (μ = 0), 2000–2004. Panel backgrounds mark the shock windows; each panel prints the annualised scale σ, the VG ν, and the NIG α.*

![Figure 12. Per-year fitted VG and NIG densities, 2005–2009.](week4/figures/week4_marginals_2005_2009.png)

*Figure 12. As Figure 11, for 2005–2009. The GFC years 2008–2009 carry the sharpest peaks and heaviest tails of the block.*

![Figure 13. Per-year fitted VG and NIG densities, 2010–2014.](week4/figures/week4_marginals_2010_2014.png)

*Figure 13. As Figure 11, for 2010–2014.*

![Figure 14. Per-year fitted VG and NIG densities, 2015–2019.](week4/figures/week4_marginals_2015_2019.png)

*Figure 14. As Figure 11, for 2015–2019.*

![Figure 15. Per-year fitted VG and NIG densities, 2020–2024.](week4/figures/week4_marginals_2020_2024.png)

*Figure 15. As Figure 11, for 2020–2024. The COVID year 2020 has the most extreme peak-and-tail shape of the whole sample.*

---

# Week 5: Posterior predictive checks and convergence diagnostics

## 1. How the check works

I have a posterior over parameters. For each draw from that posterior I generate a complete replicate dataset the same size as the real one, 6,287 returns, from that draw's likelihood. Do this for a thousand draws and I have a thousand fake histories, each one a fair sample of what the fitted model thinks the market should produce. Then I compare the real data against that cloud.

I did this in numpy and scipy rather than rebuilding the models in PyMC, because Week 4 already saved every posterior draw to CSV. Reading those back and sampling the likelihood at each one gives the same thing without re-running the sampler.

Value at risk and expected shortfall at 95%, 97.5% and 99% are the risk numbers the project actually reports, computed on each replicate exactly as they are on the real series; 97.5% is included because it is the ES confidence level FRTB mandates for regulatory capital. For each statistic I get its spread across the thousand replicates, and I score the fit with a two-sided posterior predictive p-value: the probability that a replicate is at least as extreme as the observed value. A p near 0.5 means the model produces data with that feature comfortably. A p near 0 or 1 means the real data sits out past almost every replicate, and the model cannot reach it. I flag anything below 0.05 or above 0.95.

I also included one statistic knowing every model will fail it: the lag-1 autocorrelation of squared returns, the standard measure of volatility clustering (Cont, 2001). All four models are fitted as independent draws from a fixed distribution, so their replicates cannot cluster by construction. The real series does. Every other statistic in the table is a property of the histogram, which is what these models were built to describe; this one tests the dynamics, and it shows where the static approach stops working. The Week 3 lead-up regression made the same point from the other direction: trailing volatility alone explains 44% of forward volatility, and that is exactly the dependence an iid model throws away.

---

## 2. Gaussian

Figure 1 shows the Gaussian's replicate densities in blue with the real data in black, and the right panel zooms into the left tail on a log scale. The Gaussian was fitted to a daily volatility near 1.2% and it reproduces that. The tail panel is where it falls apart. The real density still has visible mass out past −5%, and the Gaussian replicates have effectively none.

The statistics say the same thing. Observed excess kurtosis is 10.4; the Gaussian replicates run between −0.11 and +0.12, as a normal distribution must. The real worst day is −12.8%; the worst day the Gaussian manages across a thousand replicate histories sits around −5%. Both p-values are below 1/1,000; no replicate came close. The risk numbers carry the same verdict in the units the project cares about: observed 99% expected shortfall is −5.07%, and the Gaussian replicates put it near −3.24%; at the FRTB level the observed 97.5% expected shortfall is −3.79% against replicates near −2.83%. The 79.5% shortfall gap from Week 2 is not an artefact of one estimator. It is the Gaussian being structurally unable to generate the losses the market actually delivered.

At 95% value at risk the Gaussian is not too thin but slightly too fat: observed −1.88% against a replicate −1.99%. That is the flip side of forcing a single bell curve onto a peaked, heavy-tailed sample. To cover the fat middle the fitted variance has to stretch, which pushes the ordinary 95% quantile out a little too far. It is only when you go further into the tail, to 99% and to expected shortfall, that the missing mass shows up and the Gaussian collapses.

![Figure 1. Posterior predictive density for the Gaussian.](week5/figures/week5_ppc_gaussian.png)

*Figure 1. Posterior predictive check for the Gaussian. Left: replicate return densities (blue) against the observed data (black). Right: the left tail on a log scale, where the observed mass past −5% has no counterpart in the replicates.*

---

## 3. Laplace

The Laplace is my Variance-Gamma model. Figure 2 is a clear step up from the Gaussian. Its tails reach much further, and its 95% value at risk now matches the data (p = 0.13).

It still fails the deep tail. Observed excess kurtosis is 10.4 against replicates near 2.9, which is exactly the kurtosis a Laplace has by construction. Its 99% expected shortfall lands near −3.89% against the observed −5.07%, better than the Gaussian's −3.24%.

![Figure 2. Posterior predictive density for the Laplace.](week5/figures/week5_ppc_laplace.png)

*Figure 2. Posterior predictive check for the Laplace. The exponential tails reach much further than the Gaussian's but still fall short of the observed mass in the deep left tail.*

---

## 4. Student-t

The Student-t is the first model where it gets the deep tail correct. Its 99% value at risk matches the data almost exactly (p = 0.94), and its 99% expected shortfall covers the observed −5.07% comfortably.

Observe the excess kurtosis in Figure 3 and Table 1. The replicate median is about 49, and the 94% band runs from 14 all the way to 1,200. The observed 10.4 sits below almost all of it. The posterior for ν sits around 2.66, and a Student-t only has finite kurtosis when ν exceeds 4.

For a genuinely heavy-tailed model, moment-based checks like kurtosis are close to meaningless, because the moment they measure does not exist in the population. The quantile-based checks, value at risk and expected shortfall, stay well defined no matter how heavy the tail, and those are the ones the Student-t passes. The same caution applies to its skew "pass" (p = 0.81). The observed −0.39 falls inside the band only because the replicate skew ranges from −10.7 to +11.8: a symmetric model whose third moment barely exists produces sample skews so erratic that almost any value would pass. That pass is degenerate, not evidence that the Student-t captures the asymmetry.

![Figure 3. Posterior predictive density for the Student-t.](week5/figures/week5_ppc_student_t.png)

*Figure 3. Posterior predictive check for the Student-t. The tails now cover the observed data, but with ν near 2.66 the replicate tails are so heavy that the densest part of the figure is dominated by occasional extreme replicates.*

---

## 5. NIG

The NIG passes every marginal check. Each distributional statistic I tested falls inside its replicate band: standard deviation (p = 0.82), excess kurtosis (p = 0.21), skew (p = 0.56), the minimum and maximum, and value at risk and expected shortfall at all three levels, including the FRTB 97.5% (observed ES −3.79% against a replicate median of −3.93%). Its 99% expected shortfall is −5.17% against the observed −5.07%. Figure 4 shows replicate densities sitting on the data through the body and into both tails, and unlike the Student-t its kurtosis is finite and stable.

The skew result is the one I find most satisfying. The replicate skew is centred at −0.54 against the observed −0.39, and the band is tight enough for the pass to mean something (unlike the Student-t's). The model is not just heavy-tailed in a symmetric way.

The NIG fails one statistic, the same one every model fails: the lag-1 autocorrelation of squared returns. The observed value is 0.32 and the replicate bands of all four models sit on zero, because independent draws cannot cluster. That says nothing about the NIG relative to the others. A fixed distribution can get the size of tail risk right, which the NIG demonstrably does, but it cannot say when that risk arrives; the arrival times are the volatility clustering the iid assumption discards.

![Figure 4. Posterior predictive density for the NIG.](week5/figures/week5_ppc_nig.png)

*Figure 4. Posterior predictive check for the NIG. Replicate densities track the observed data through the body and into both tails, with the heavier left tail reproduced rather than assumed.*

Figure 5 puts the risk numbers side by side. For each of value at risk and expected shortfall at 95%, 97.5% and 99% it draws each model's replicate band against the observed value as a dashed line. The Gaussian band sits clear of the dashed line in the 97.5% and 99% panels, the Laplace closes about half the distance, and the Student-t and NIG bands straddle it. The picture is the Week 2 shortfall gap drawn in posterior-predictive form, with the NIG the only model whose band covers the truth on all six.

![Figure 5. Posterior predictive tail risk for all four models.](week5/figures/week5_ppc_risk.png)

*Figure 5. Posterior predictive value at risk and expected shortfall at 95%, 97.5% (the FRTB ES level) and 99%. Each marker is a model's replicate median with its 94% band; the dashed line is the observed value. The Gaussian misses badly beyond 95%; the NIG covers all six.*

**Table 1. Posterior predictive p-values by model. Values near 0.5 indicate the model reproduces that feature; values flagged with † sit beyond 0.05 or 0.95, where the observed data lies outside almost all replicates. ACF²(1) is the lag-1 autocorrelation of squared returns, the volatility clustering statistic that no iid model can reproduce.**

| Statistic | Gaussian | Laplace | Student-t | NIG |
|-----------|----------|---------|-----------|-----|
| Std deviation | 0.99 † | 0.00 † | 0.05 | 0.82 |
| Excess kurtosis | 0.00 † | 0.00 † | 0.00 † | 0.21 |
| Skew | 0.00 † | 0.00 † | 0.81 | 0.56 |
| Minimum | 0.00 † | 0.00 † | 0.22 | 0.36 |
| Maximum | 0.00 † | 0.01 † | 0.05 | 0.28 |
| ACF²(1) | 0.00 † | 0.00 † | 0.00 † | 0.00 † |
| VaR 95% | 0.01 † | 0.13 | 0.00 † | 0.93 |
| ES 95% | 0.00 † | 0.00 † | 0.92 | 0.55 |
| VaR 97.5% | 0.00 † | 0.03 † | 0.08 | 0.37 |
| ES 97.5% | 0.00 † | 0.00 † | 0.52 | 0.50 |
| VaR 99% | 0.00 † | 0.00 † | 0.94 | 0.16 |
| ES 99% | 0.00 † | 0.00 † | 0.23 | 0.73 |

---

## 6. Convergence diagnostics

Because the posterior predictive check relies on the posterior draws being trustworthy, I revisited the draws themselves with more scrutiny than the Week 4 trace plots provided. Working directly from the saved chains rather than through a library, I computed four diagnostics: split R-hat, which compares within-chain to between-chain variance and should equal 1.00 once the chains agree; bulk effective sample size, which indicates how many independent draws the correlated chain is effectively worth for estimating the posterior's centre; tail effective sample size, the same concept applied to the 5% and 95% indicators so that it reflects the quantiles; and the Monte Carlo standard error on each posterior mean.

Table 2 has the numbers. Every R-hat rounds to 1.00. Bulk ESS never drops below ~2,000, and tail ESS stays above 2,100 across the board. The tail figure is the one that actually matters here. The quantities I'm reporting are tail quantiles, and a posterior mean is only as good as the effective sample behind its tail. Monte Carlo standard errors are small next to the posterior spread. Figure 6 shows the chains for all four models: flat, stationary, sitting right on top of each other, which is just the table again but easier to read.

**Table 2. Convergence diagnostics, computed directly from the saved chains.**

| Model | Parameter | R-hat | Bulk ESS | Tail ESS | MCSE |
|-------|-----------|-------|----------|----------|------|
| Gaussian | μ | 1.00 | 3,973 | 2,626 | 2.4e-06 |
| | σ | 1.00 | 3,323 | 2,300 | 2.0e-06 |
| Laplace | μ | 1.00 | 3,844 | 2,531 | 1.7e-06 |
| | b | 1.00 | 3,670 | 2,944 | 1.7e-06 |
| Student-t | μ | 1.00 | 2,574 | 2,505 | 2.2e-06 |
| | σ | 1.00 | 2,520 | 2,232 | 2.5e-06 |
| | ν | 1.00 | 2,326 | 2,328 | 2.3e-03 |
| NIG | μ | 1.00 | 2,251 | 2,428 | 3.1e-06 |
| | α | 1.00 | 2,074 | 2,138 | 6.3e-02 |
| | β | 1.00 | 2,089 | 2,315 | 3.2e-02 |
| | δ | 1.00 | 2,085 | 2,430 | 4.8e-06 |

![Figure 6. Trace diagnostics for all four models.](week5/figures/week5_diagnostics_nig.png)

*Figure 6. Per-chain traces for the NIG.*

---

## 7. Where this leaves the project

The NIG reproduces every marginal feature of the data: the body, both tails, the skew, and the risk numbers at all three confidence levels including the FRTB 97.5%. The Student-t gets the tail quantiles right but has moments too heavy to pin down. The Laplace closes part of the Gaussian's gap but stops at exponential tails. The Gaussian fails the moment the test moves past the centre of the distribution. This confirms the Week 2 shortfall by a different route.

What no static model reproduces is the clustering. All four fail the squared-return autocorrelation check, because independent draws cannot produce the market's long calm stretches broken by turbulent ones. So the summary has two parts: the NIG gets the size of tail risk right, and no fixed marginal gets the timing. Whether re-estimating the parameters through time is enough to fix the timing is the question for the Week 7 rolling backtest. Christoffersen's independence test catches the clustered VaR violations a fixed model produces, and the rolling window gives each model its chance to pass.

---

# Week 6: The calendar structure of returns

## 1. Overview

This week I looked at the same 6,287 daily S&P 500 log-returns from a different angle: the calendar. Instead of asking what distribution fits the returns, I asked when the risk actually shows up. Four questions, following the week's plan:

1. Does the day of the week matter, and is the right way to cut the week Monday / midweek / Friday, or is the week just one block?
2. How does volatility split between the market being open (intraday) and closed (overnight), and what does the week-open (Monday) to week-close (Friday) picture look like inside each of our four crises?
3. Can I explain the fitted Lévy parameters with a regression, one parameter at a time, starting with the NIG δ?
4. Do quarterly earnings seasons move index volatility?

Short answers: yes, Monday / midweek / Friday is exactly the right cut; the market is only open for three fifths of its variance and almost none of its jumps; δ is mostly a VIX story and the skew parameters turn out to follow the VIX too once there is enough data; and earnings seasons do nothing at the index level, which turned out to be the most interesting null of the project so far.

---

## 2. The shape of the week

I fitted the Gaussian and the Student-t to each weekday separately. One thing to keep in mind: a Monday return runs from Friday's close to Monday's close, so the whole weekend is inside it.

**Table 1. Per-weekday MLEs, 2000-2024. Mean in basis points per day, σ annualised.**

| Day | n | Mean (bp) | σ (ann.) | Student-t ν | SE(ν) | Excess kurtosis |
|-----|---|-----------|----------|-------------|-------|-----------------|
| Monday | 1,178 | +0.1 | 21.4% | 2.17 | 0.18 | 17.5 |
| Tuesday | 1,289 | +5.5 | 19.4% | 2.88 | 0.27 | 8.7 |
| Wednesday | 1,290 | +1.6 | 18.9% | 2.70 | 0.25 | 6.9 |
| Thursday | 1,268 | +3.4 | 19.6% | 2.61 | 0.24 | 8.3 |
| Friday | 1,262 | +0.3 | 17.8% | 3.10 | 0.35 | 5.3 |

The means are all statistically zero (each one has a standard error of about 3.4 bp), which is what an efficient market should give. The interesting rows are the second moments. Monday is the most volatile day and has by far the heaviest tail: ν = 2.17, close enough to the ν = 2 boundary that its confidence interval brushes against infinite variance. Friday is the calmest and lightest day on every measure.

The plan asked me to compare two definitions of the week, and the likelihood-ratio tests settle it:

**Table 2. Gaussian likelihood-ratio tests between week definitions.**

| Comparison | LR | df | p |
|------------|----|----|---|
| Mon / midweek / Fri vs one pooled week | 42.9 | 4 | < 0.0001 |
| Five separate days vs one pooled week | 45.7 | 8 | < 0.0001 |
| Five separate days vs Mon / midweek / Fri | 2.8 | 4 | 0.594 |

Treating the week as one block is firmly rejected. But splitting it all the way into five days is no better than the three-group version. So Monday / midweek / Friday is the right resolution: Tuesday, Wednesday and Thursday are statistically the same day, and the week's real structure is its open, its middle and its close.

Grouped that way over the full sample, the pattern is a clean staircase. Volatility falls from 21.4% (Monday) to 19.3% (midweek) to 17.8% (Friday). Excess kurtosis falls from 17.5 to 8.0 to 5.3. The tail parameter rises from ν = 2.17 to 2.73 to 3.10. Every measure says the same thing: the week opens risky and heavy-tailed, and calms down as it goes.

![Figure 1. Volatility and tail parameter by weekday.](week6/figures/week6_weekday_params.png)

*Figure 1. Left: annualised Gaussian volatility by weekday with 95% intervals. Right: Student-t ν by weekday against the full-sample 2.648.*

---

## 3. The week inside each crisis

Next I ran the same week-open (Monday) to week-close (Friday) lens through each of the four crisis windows, measuring volatility, variance and the fat-tail measures in each. One warning before the table: the crisis windows are short, so the per-group samples get small. COVID has only about 20 Mondays and 20 Fridays, so I report volatility, variance and kurtosis there but not a Student-t fit, which would need more data than that to mean anything.

**Table 3. Week open vs midweek vs week close, full sample and per crisis. Variance is the daily variance in %². ν is omitted where n < 50.**

| Window | Group | n | σ (ann.) | Daily var (%²) | Excess kurt | Worst day | ν |
|--------|-------|---|----------|----------------|-------------|-----------|---|
| Full sample | Monday | 1,178 | 21.4% | 1.81 | 17.5 | −12.8% | 2.17 |
| | Midweek | 3,847 | 19.3% | 1.48 | 8.0 | −10.0% | 2.73 |
| | Friday | 1,262 | 17.8% | 1.25 | 5.3 | −6.0% | 3.10 |
| Dot-com crash | Monday | 127 | 23.1% | 2.12 | 2.0 | −5.0% | 4.78 |
| | Midweek | 409 | 23.7% | 2.23 | 0.9 | −4.2% | 7.43 |
| | Friday | 135 | 23.1% | 2.11 | 1.5 | −6.0% | 7.10 |
| GFC | Monday | 73 | 44.8% | 7.98 | 4.8 | −9.4% | 2.01 |
| | Midweek | 229 | 38.8% | 5.99 | 2.5 | −9.5% | 2.79 |
| | Friday | 76 | 28.2% | 3.16 | 1.1 | −4.3% | 7.59 |
| COVID-19 | Monday | 20 | 65.9% | 17.2 | 2.3 | −12.8% | |
| | Midweek | 64 | 46.7% | 8.65 | 2.1 | −10.0% | 2.42 |
| | Friday | 20 | 43.9% | 7.63 | 3.0 | −4.4% | |
| Fed rate hikes | Monday | 90 | 16.7% | 1.10 | 2.8 | −4.0% | 2.64 |
| | Midweek | 309 | 19.6% | 1.53 | 1.4 | −4.4% | 8.10 |
| | Friday | 102 | 21.1% | 1.77 | 0.2 | −3.7% | 15.5 |

The four crises do not treat the week the same way, and the differences line up with what Week 3 found about the crises themselves.

The GFC is the extreme case. Monday volatility was 44.8% against Friday's 28.2%, so the variance of a GFC Monday was two and a half times the variance of a GFC Friday. The Monday tail parameter is ν = 2.01, sitting exactly on the infinite-variance boundary, while GFC Fridays fit at ν = 7.6, which is nearly Gaussian. The crisis was hitting hardest right as the week opened, after the weekend's bank failures and rescue announcements had piled up with no trading to absorb them.

COVID shows the same shape even more sharply in volatility terms: 65.9% on Mondays against 43.9% on Fridays, and the single worst day of the whole 25-year sample, the −12.8% of 16 March 2020, was a Monday. With only 20 observations per group I would not lean on the kurtosis numbers, but the volatility gap is too big to be an accident of sampling.

The dot-com crash is completely flat: 23.1%, 23.7%, 23.1% across the three groups, with mild tails throughout. That fits its character from Week 3. It was a long grind lower, not a sequence of weekend shocks, so it had no reason to care what day it was.

The Fed rate-hike window is the surprise: the pattern reverses. Mondays were the calm days (16.7%) and Fridays the risky ones (21.1%). My reading is that this crisis ran on scheduled announcements rather than weekend surprises, and the big macro releases, CPI and the payrolls report, land on weekday mornings, with payrolls on Fridays. When the risk arrives by calendar appointment, the weekend stops mattering and the week's shape flips. The tail numbers back this up: those risky Fridays fit at ν = 15.5 with excess kurtosis of just 0.2, so the announcement volatility was big but nearly Gaussian. Scheduled news makes the market wide, not fat-tailed.

So the full-sample Monday effect in Section 2 is really a crisis effect. It is driven by the GFC and COVID, absent in the dot-com years, and reversed in 2022-23.

![Figure 2. The week inside each crisis.](week6/figures/week6_crisis_weekday.png)

*Figure 2. Annualised volatility (left) and excess kurtosis (right) for Monday, midweek and Friday, over the full sample and inside each crisis window.*

---

## 4. Market open vs market closed within the day

The other meaning of open versus closed is within a single day. The close-to-close return splits exactly into an overnight part (yesterday's close to today's open, while the market is shut) and an intraday part (open to close, while it is trading):

r_close-to-close = r_overnight + r_intraday

One data problem first. Yahoo's ^GSPC open prices are fake in the early sample. A real index almost never opens exactly where it closed, since futures move overnight, so when the open equals the previous close it means the vendor had no true open and just copied the prior close into the column. I flag a day as stale when |ln(O_t / C_{t-1})| < 10^-8. The script recomputes this audit every time it runs, and the fractions come out at 96.3% of days in 2000-2004, 31.4% in 2005-2009, 12.4% in 2010-2014, and 0.08% from 2015 on. Running the overnight and intraday split on the early data would therefore give an overnight return of exactly zero on nearly every day of 2000-2004. That zero is a vendor artifact, and it would sit as a huge spike at the origin in every distributional estimate this section makes.

There is no repairing the bad opens; the true opening levels were simply never recorded in this data. So this section restricts itself to 2015-2024, where the opens are real. That leaves n = 2,515 trading days, plenty for estimation, and the leftover 0.08% is 2 days whose overnight return happens to be exactly zero, which does no harm. Nothing else in the project is touched: this is the only analysis that reads the open column, and everything else runs on closing prices, which are the official end-of-day index levels and reliable over the whole 25 years. The cost is scope. Everything below describes 2015-2024, and pushing the split further back would take a better data source for the early opens rather than a different method.

**Table 4. Return components, 2015-2024.**

| Component | σ (ann.) | Variance share | Student-t ν | Excess kurtosis | Skew |
|-----------|----------|----------------|-------------|-----------------|------|
| Close-to-close | 17.9% | 100% | 2.68 | 15.7 | −0.81 |
| Overnight (closed) | 8.0% | 20.0% | 2.14 | 36.1 | −1.76 |
| Intraday (open) | 14.0% | 61.0% | 2.79 | 5.2 | −0.41 |

The two shares add to 81%; the missing 19% is twice the covariance, because the components correlate at +0.27 and overnight moves tend to continue into the day.

The split is lopsided in a way I find really satisfying. The open market carries three times the closed market's variance, but the closed market carries the tails. Overnight returns fit at ν = 2.14 with excess kurtosis of 36 and skew of −1.8. Intraday returns are comparatively tame: ν = 2.79, kurtosis 5. The reason seems clear enough: while the market is shut, news piles up and nothing can be traded against it, so it all lands at once at the open. That is a jump. And the whole project has been about jumps, so it is nice to find out what time of day they happen: mostly while nobody can trade.

![Figure 3. Overnight vs intraday densities.](week6/figures/week6_open_close_density.png)

*Figure 3. Overnight and intraday return densities, 2015-2024, on a linear and a log scale. The overnight density is narrower in the middle but crosses over in the tails.*

The two calendar effects also meet exactly where they should. Monday's overnight component runs at about 10% annualised against 7.3 to 7.8% for the other days. That extra piece is the weekend gap from Sections 2 and 3, now located in the specific session where it accrues.

![Figure 4. Volatility by weekday and session.](week6/figures/week6_open_close_weekday.png)

*Figure 4. Annualised volatility by weekday, split into overnight and intraday. The weekend lives in Monday's overnight bar.*

---

## 5. Week-on-week volatility

For the week-on-week measure I built weekly realised volatility (the root of each week's summed squared daily returns, annualised) over the full 2000-2024 sample, 1,302 weeks.

Volatility carries over strongly from one week to the next: vol_w = 4.30 + 0.72 × vol_(w−1), with R² = 0.52. A calm week is usually followed by a calm week and a wild one by a wild one. This is the weekly version of the volatility clustering that the Week 5 posterior predictive check showed none of our static models can produce.

Weekly returns also let me check aggregational Gaussianity, one of the stylised facts from the Week 1 literature review. The Student-t fitted to weekly returns gives ν = 3.38 (SE 0.35) against the daily 2.648. One step of aggregation already lightens the tail by a measurable amount, though weekly returns are still nowhere near Gaussian.

![Figure 5. Weekly realised volatility.](week6/figures/week6_weekly_vol.png)

*Figure 5. Left: weekly realised volatility, 2000-2024, with the four crisis windows shaded. Right: each week's volatility against the previous week's, slope 0.72, R² 0.52.*

---

## 6. Quarterly parameter regressions

The central item of the plan: fit the Lévy models through time and regress each parameter separately. Yearly fits only give 25 data points, so I refitted VG(σ, θ, ν) and NIG(α, β, δ) with μ = 0 one calendar quarter at a time. That gives 100 quarters of roughly 63 returns each, using the same zero-mean machinery as the Week 4 yearly fits.

One thing has to be dealt with before any regression. In calm quarters the NIG cannot tell itself apart from a Gaussian: α and δ both blow up together, and only their ratio (which sets the variance) is pinned down by the data. The individual values of α and δ in those quarters are meaningless. This happens in 33 of the 100 quarters (I flagged any quarter with α above 500), so the δ regressions run on the 67 quarters where δ actually means something. The fact that a third of all quarters look Gaussian is a finding in itself: it repeats the Week 3 message that heavy tails are something markets do in episodes, not all the time.

**Table 5. One regression per parameter. Newey-West standard errors, 4 lags. The full specification adds realised volatility, drawdown and the parameter's own lag to the VIX.**

| Parameter | R² (VIX only) | VIX t-stat | R² (full) | AR(1) slope | AR(1) R² |
|-----------|---------------|------------|-----------|-------------|----------|
| NIG δ (ann., 67 quarters) | 0.50 | +3.0 | 0.60 | +0.40 | 0.16 |
| NIG log α | 0.02 | −1.3 | 0.24 | +0.26 | 0.07 |
| NIG β | 0.17 | −3.2 | 0.28 | −0.01 | 0.00 |
| VG σ (ann.) | 0.82 | +19.9 | 1.00 | +0.52 | 0.27 |
| VG θ | 0.19 | −5.1 | 0.65 | +0.04 | 0.00 |
| VG ν | 0.04 | −2.1 | 0.22 | +0.19 | 0.04 |

The δ result is the one the plan asked for first. Against the VIX alone, δ gives R² = 0.50 in levels. In logs, which stops a few crisis quarters from dominating the fit, it reaches R² = 0.64 with a t-statistic of 9.7. The NIG's scale parameter is, to a decent approximation, a re-reading of the VIX. The VG σ says the same even louder: R² = 0.82 with 100 quarters, matching the 0.90 that Week 3 found with 25 years. (The R² of 1.00 in σ's full specification is not impressive, it is circular: quarterly realised volatility and the fitted VG scale are almost the same number, so one explains the other perfectly. I kept the row only for completeness.)

The genuinely new result is the skew. The Week 3 annual regression found no relationship between the VIX and either asymmetry parameter, with p-values of 0.60 and 0.16 on 25 points. With 100 quarters both show up clearly: VG θ at t = −5.1 and NIG β at t = −3.2. Higher-VIX quarters are more left-skewed quarters. The annual nulls were a sample-size problem, which the Week 3 limitations section suspected at the time. The tail-decay parameters, on the other hand, stay flat against the VIX at any frequency (log α at R² = 0.02, VG ν at 0.04). So the sharper version of Week 3's conclusion is: the VIX prices the scale of the distribution and, seen quarterly, its skew, but it says nothing about how fast the tails decay.

The plan's "two parameters as a function of themselves" is the AR(1) column: each parameter regressed on its own previous value. The scale parameters persist (log δ has an AR slope of 0.58 with R² = 0.34, VG σ sits at 0.52). The skew and tail parameters have AR slopes of essentially zero. So the scale of the market moves slowly and remembers itself from quarter to quarter, while asymmetry and tail weight belong to specific episodes and reset.

![Figure 6. Quarterly NIG parameters through time.](week6/figures/week6_quarterly_nig.png)

*Figure 6. Quarterly NIG δ (top) and α (bottom), both on log axes, crisis windows shaded. Grey crosses are the 33 weakly identified quarters where only the ratio of δ to α means anything. Among the identified quarters, α bottoms out near 7 in 2020Q1, the heaviest quarterly tail in the sample.*

![Figure 7. The δ regressions.](week6/figures/week6_delta_regressions.png)

*Figure 7. Left: quarterly NIG δ against that quarter's average VIX, well-identified quarters only. Right: δ against its own lag, the AR(1).*

---

## 7. Earnings seasons

The index has no earnings dates of its own, so I used the aggregate reporting calendar as a proxy: each earnings season is the month starting on the 15th of January, April, July and October, when most S&P 500 companies report. That covers 34% of trading days. This is a proxy, and a study with company-level reporting dates would be a different and bigger exercise.

**Table 6. In season vs out of season, 2000-2024.**

| Window | n | σ (ann.) | Student-t ν | Excess kurtosis |
|--------|---|----------|-------------|-----------------|
| In season | 2,161 | 19.1% | 2.87 | 7.4 |
| Out of season | 4,126 | 19.6% | 2.55 | 11.8 |

I also ran an event-study profile: average volatility in the 21 trading days before each season, during it, and in the 21 days after, across 99 seasons. The answer is flat to the decimal: 16.5%, 16.4%, 16.5%, with paired t-statistics all under 0.15.

I expected at least something here, so the flatness took me a moment to accept. But it makes sense. Earnings risk is company-specific, and inside a 500-name index the individual surprises average away almost completely. If anything the in-season distribution has lighter tails (ν = 2.87 against 2.55), because the shocks that actually move the index tail, March 2020, August 2011, the 2008 cascade, mostly happened outside reporting windows. The conclusion I take: index-level tail risk is macro risk, not earnings risk.

![Figure 8. Volatility around earnings seasons.](week6/figures/week6_earnings_profile.png)

*Figure 8. Average annualised volatility before, during and after the 99 earnings seasons, with 95% intervals.*

---

## 8. Limitations

The crisis-window weekday cuts run on small samples, down to 20 Mondays for COVID, so those rows lean on volatility and variance rather than fitted tail parameters, and the kurtosis figures there are indicative at best. The overnight/intraday split rests on ten years of usable open prices and describes the post-2015 market only. The quarterly NIG fits are noisy for the tail and skew parameters at 63 observations a quarter, and the ridge treatment means the δ results describe normal-to-turbulent quarters, not calm ones. The regressions are contemporaneous attribution in the Week 3 sense, not forecasts, and the VIX and realised volatility are strongly correlated, so their coefficients in the full specifications are partial associations. The earnings windows are a calendar proxy at index level; a constituent-level event study could still find effects the index averages away.

---

# Week 7: The rolling VaR backtest

## 1. Overview

Everything up to now has been in-sample: fit a distribution to the full history, read the risk numbers off the fitted tail. This week I made the models work for a living. I refitted each one on a rolling 500-day window and asked it, every day from January 2002 to December 2024, for tomorrow's Value-at-Risk. That gave me 5,787 genuine out-of-sample forecasts per model per confidence level, and a hit sequence of days where the realised return fell below the forecast. Christoffersen's (1998) framework then asks the two questions that matter about that sequence: are there the right *number* of violations, and do they arrive *independently*? A model can only be called correct if the answer to both is yes.

This also puts the last of the four references to work. Christoffersen (1998) has been sitting in my bibliography since Week 1 waiting for exactly this exercise.

The models are the four from the Bayesian arc: Gaussian, Laplace, Student-t and NIG, with the Laplace again standing in for the Variance-Gamma family as its symmetric special case. I refitted every 21 trading days, roughly the monthly update cycle a risk desk would run, and held the forecasts fixed between refits.

The headline is a clean split. Rolling refits repair the *number* of violations at the 95% level for the Gaussian and the NIG, and the NIG comes closest at 99%. But every model, at every level, fails the independence test outright. The violations do not arrive as a trickle; they arrive as bursts in late 2008 and March 2020. A rolling window drags the whole distribution towards yesterday's volatility, but a month too late. The missing ingredient is not a heavier tail. It is conditional volatility, and no static marginal, however well chosen, can supply it.

---

## 2. Design

I chose a window of 500 trading days, about two years: long enough for a stable four-parameter NIG fit, short enough to adapt across regimes. I refitted every 21 trading days and held the parameters in between. Each day's forecast is one-day-ahead VaR at 95%, 97.5% and 99%, read off the fitted distribution's quantile (closed form for the Gaussian, Laplace and Student-t; root-finding on the CDF for the NIG). A violation is a realised return strictly below the forecast.

One numerical fight worth recording. On calm windows the rolling NIG regularly lands on its Gaussian-limit ridge (very large α with big standard errors), the same weakly identified regime my quarterly fits found in Week 6. scipy's generic NIG quantile refused to converge there, so I solved the quantile myself: bracket the root using the NIG's own mean and standard deviation, then run Brent's method on the CDF. On the ridge the quantile smoothly approaches the Gaussian one, which is the behaviour I wanted.

I ran three likelihood-ratio tests per model and level, following Christoffersen (1998):

- **LR_uc** (Kupiec, unconditional coverage): is the violation rate equal to the nominal 5%, 2.5% or 1%? Chi-squared, 1 df.
- **LR_ind** (independence): is the chance of a violation today unaffected by whether yesterday was a violation, against a first-order Markov alternative? Chi-squared, 1 df.
- **LR_cc** (conditional coverage): the sum of the two, chi-squared with 2 df. This is the joint test the paper is about.

---

## 3. Coverage: how many violations

**Table 1. Rolling backtest, 5,787 out-of-sample days (Jan 2002 to Dec 2024). Expected violations are n x (1 - level).**

| Model | Level | Expected | Hits | Rate | LR_uc | p_uc | LR_ind | p_ind | LR_cc | p_cc |
|-------|-------|----------|------|------|-------|------|--------|-------|-------|------|
| Gaussian | 95% | 289 | 308 | 5.32% | 1.24 | 0.265 | 47.2 | <0.0001 | 48.5 | <0.0001 |
| Gaussian | 97.5% | 145 | 223 | 3.85% | 37.4 | <0.0001 | 39.7 | <0.0001 | 77.1 | <0.0001 |
| Gaussian | 99% | 58 | 152 | 2.63% | 106.9 | <0.0001 | 36.1 | <0.0001 | 142.9 | <0.0001 |
| Laplace | 95% | 289 | 323 | 5.58% | 3.98 | 0.046 | 35.7 | <0.0001 | 39.6 | <0.0001 |
| Laplace | 97.5% | 145 | 193 | 3.34% | 15.0 | 0.0001 | 34.6 | <0.0001 | 49.6 | <0.0001 |
| Laplace | 99% | 58 | 110 | 1.90% | 37.5 | <0.0001 | 28.0 | <0.0001 | 65.5 | <0.0001 |
| Student-t | 95% | 289 | 358 | 6.19% | 16.0 | 0.0001 | 39.0 | <0.0001 | 55.0 | <0.0001 |
| Student-t | 97.5% | 145 | 218 | 3.77% | 33.1 | <0.0001 | 35.9 | <0.0001 | 69.0 | <0.0001 |
| Student-t | 99% | 58 | 104 | 1.80% | 30.0 | <0.0001 | 18.6 | <0.0001 | 48.7 | <0.0001 |
| NIG | 95% | 289 | 313 | 5.41% | 1.98 | 0.159 | 37.5 | <0.0001 | 39.5 | <0.0001 |
| NIG | 97.5% | 145 | 177 | 3.06% | 6.92 | 0.009 | 31.9 | <0.0001 | 38.8 | <0.0001 |
| NIG | 99% | 58 | 91 | 1.57% | 16.3 | 0.0001 | 19.3 | <0.0001 | 35.6 | <0.0001 |

Three things stood out to me.

At 95% the Gaussian is fine and the Student-t is the worst model in the table. That inversion is worth dwelling on. The full-sample Student-t carries ν = 2.648, a tail heavy enough to flirt with infinite kurtosis. A distribution with that much mass far out in the tail has to take it from somewhere, and it takes it from the shoulders: its 5% quantile sits *closer to zero* than the Gaussian's for the same data. So at the everyday 95% level the t under-covers (358 hits against 289 expected, rejected at p = 0.0001) while the thin-tailed Gaussian, which spends its probability exactly where the 5% quantile lives, passes comfortably (p = 0.27). Heavy tails are a statement about extremes, and they carry a cost at moderate quantiles.

At 99% the ordering flips and the Gaussian collapses. It produces 152 violations where 58 were expected, a factor of 2.6, with LR_uc = 106.9. This is the out-of-sample counterpart of the 79.5% ES gap from Week 2: the Gaussian tail is simply too short where it matters. The NIG does best (91 hits, a factor of 1.57), then Student-t (104) and Laplace (110). Even the NIG is rejected on pure coverage at 99%; a static tail fitted to the last two years is still too short when the regime breaks.

And every cell of the LR_ind column is a rejection. More on that next.

![Figure 1. 99% VaR forecasts and violations per model.](week7/figures/week7_var_series.png)

*Figure 1. Rolling one-day 99% VaR forecasts (coloured line) against realised returns (grey), with violations marked. The crosses bunch in the GFC and COVID windows for every model.*

---

## 4. Independence: when the violations arrive

The independence statistics are large everywhere: LR_ind between 18.6 and 47.2, all with p-values that are zero to four decimal places. The mechanism is visible in the raw transition counts. At the 95% level, a day after a Gaussian violation had a 48-in-308 chance of being another violation, a repeat rate of about 16% against an unconditional 5.3%. The models are not wrong at random times; they are wrong in runs.

Figure 2 shows the same fact cumulatively. Violations accrue as staircases: flat for years, then a vertical jump in late 2008 and again in March 2020. The dashed line is what a correct model's staircase should look like.

![Figure 2. Cumulative violations against the nominal rate.](week7/figures/week7_cumulative_hits.png)

*Figure 2. Cumulative violations at 95% (left) and 99% (right) against the nominal accrual (dashed). The GFC and COVID verticals are the independence failure made visible.*

This is the out-of-sample confirmation of my Week 5 posterior predictive result. There, all four models failed the one dependence statistic in the battery (the lag-1 autocorrelation of squared returns, observed 0.32 against replicate bands centred on zero) while the NIG passed every marginal check. Here the same split reappears with real forecasts: the NIG gets the closest to the right number of violations, and still fails their timing, because a 500-day window updated monthly cannot chase volatility that doubles inside a week. Cont's (2001) volatility clustering is exactly the stylised fact these iid models were built without, and the backtest sends the bill.

---

## 5. The Basel traffic light

Basel's backtesting regime keys the capital multiplier off the count of 99% violations in the trailing 250 days: green up to 4, yellow from 5 to 9, red at 10 or more, where the internal model is presumed broken.

**Table 2. Share of out-of-sample days each model spends in each Basel zone (99% VaR, trailing 250 days).**

| Model | Green | Yellow | Red | Worst 250-day count |
|-------|-------|--------|-----|---------------------|
| Gaussian | 53.2% | 16.0% | 30.8% | 32 |
| Laplace | 63.8% | 17.5% | 18.7% | 28 |
| Student-t | 65.3% | 16.7% | 17.9% | 22 |
| NIG | 66.6% | 19.3% | 14.1% | 15 |

A Gaussian desk spends almost a third of the 23 years in the red zone, and its worst 250-day window contains 32 violations, more than three times the red threshold and a level at which a regulator would simply withdraw model approval. The Lévy tails halve the red-zone time, and the NIG's worst window (15) is half the Gaussian's. But no model stays out of the red: every one of them breaches the threshold through the GFC and again around COVID, which is the traffic-light version of the independence failure.

![Figure 3. Rolling 250-day violation counts against the Basel zones.](week7/figures/week7_basel_traffic.png)

*Figure 3. Trailing 250-day count of 99% violations per model over the Basel green, yellow and red bands.*

---

## 6. Expected Shortfall at 97.5%: what the models promised

FRTB (BCBS, 2013) replaces 99% VaR with 97.5% ES as the capital metric, so my last check compares, on the days each model's 97.5% VaR was breached, the average realised loss with the average ES the model had forecast for those same days. A ratio above 1 means the tail bit harder than promised.

**Table 3. Realised versus forecast losses on 97.5% breach days.**

| Model | Breach days | Mean realised | Mean forecast ES | Ratio |
|-------|-------------|---------------|------------------|-------|
| Gaussian | 223 | −2.99% | −2.25% | 1.33 |
| Laplace | 193 | −3.16% | −2.63% | 1.20 |
| Student-t | 218 | −3.02% | −3.12% | 0.97 |
| NIG | 177 | −3.18% | −2.96% | 1.07 |

The Gaussian falls short twice over: too many breach days (223 against 145 expected) *and* losses 33% deeper than the ES it booked on those days. Under FRTB that is precisely the number a capital charge is built on. The Student-t's promised shortfall is actually adequate (ratio 0.97), the same phenomenon as its 95% failure viewed from the other end: the heavy ν buys deep-tail honesty at the price of shoulder coverage. The NIG is 7% short, much the best balance of breach count and depth, which matches its Week 5 status as the only model to pass the marginal FRTB check.

---

## 7. Where this leaves the project

The backtest completes an argument I have been building since Week 2:

1. The full-sample tail is heavy (ν = 2.648) and the Gaussian understates 99% ES by 79.5% in-sample (Week 2). Out of sample, that shows up as 2.6 times too many 99% violations and realised breach losses 33% beyond the booked ES.
2. Choosing a better marginal genuinely helps. The NIG cuts the excess 99% violations from a factor 2.6 to 1.57, spends half as much time in the Basel red zone, and prices its own breaches to within 7%.
3. No marginal is enough. Every model fails Christoffersen's independence test at every level, because iid models, however heavy their tails, put the violations in the wrong places. The failure is not the size of the tail but its timing, and the timing is volatility clustering.

Point 3 is the honest limit of the model class I have been working with, established formally in-sample in Week 5 and now out-of-sample here. It also hands me the closing argument for the final write-up: the Lévy marginals fix what is fixable at the level of a static distribution, and what remains is, by construction, dynamics.

---

---

# Conclusions

## 1. The consolidated scorecard

I have now examined every model four ways: in-sample fit by maximum likelihood (Weeks 2 and 3), full Bayesian posteriors (Week 4), posterior predictive checks against the observed data (Week 5), and a rolling out-of-sample VaR backtest (Week 7). Table 1 puts the whole project on one page. The Variance-Gamma's Bayesian and backtest columns are carried by the Laplace, its symmetric special case (θ = 0, ν = 1), which is how I represented the VG family from Week 4 onward.

**Table 1. The five models across the full project. ΔAIC is against the Gaussian (more negative is better); KS is the Kolmogorov-Smirnov test on the full sample; PPC summarises the Week 5 posterior predictive checks; the backtest columns are the Week 7 rolling one-day 99% VaR results over 5,787 out-of-sample days (58 violations expected) and the 97.5% ES shortfall ratio (realised loss over promised ES on breach days; 1 is honest).**

| | Gaussian | Laplace | Student-t | VG | NIG |
|---|---|---|---|---|---|
| Parameters | 2 | 2 | 3 | 4 | 4 |
| ΔAIC vs Gaussian | 0 | −1,771 | −1,803 | −1,798 | −1,851 |
| KS test | rejected | borderline | borderline | passes | passes |
| 99% ES (in-sample) | −3.24% | −3.91% | −5.81% | −4.28% | −5.19% |
| PPC verdict | fails beyond the centre | fails deep tail | tail quantiles pass; moments undefined | (via Laplace) | passes all marginal checks |
| Backtest 99% violations | 152 (×2.6) | 110 (×1.9) | 104 (×1.8) | (via Laplace) | 91 (×1.6) |
| Basel red-zone share | 30.8% | 18.7% | 17.9% | (via Laplace) | 14.1% |
| ES shortfall ratio (97.5%) | 1.33 | 1.20 | 0.97 | (via Laplace) | 1.07 |
| Independence test | fails | fails | fails | (via Laplace) | fails |

## 2. What the project found

The Gaussian assumption fails where it is most expensive. The full-sample Student-t degrees of freedom is ν = 2.648, and the Bayesian posterior places the entire 94% interval at (2.46, 2.87): heavy tails with a finite variance, as a stable population property rather than a point estimate. The price of ignoring this is concentrated in the tail the regulation cares about. In-sample, the Gaussian understates 99% Expected Shortfall by 79.5% against the Student-t; out-of-sample it produces 2.6 times the nominal number of 99% VaR violations, spends 31% of two decades in the Basel red zone, and books an ES one third smaller than the losses realised on its own breach days.

Most of the gain is exponential tails; the last part is the NIG's. The two-parameter Laplace, with the same parameter count as the Gaussian, captures 96% of the best model's AIC improvement: simply replacing thin tails with exponential ones does most of the work. The NIG earns the rest with genuine structure. It is the only model to pass the KS test and every posterior predictive check including skew and the FRTB 97.5% ES, its posterior asymmetry β is negative across its whole credible interval, and out of sample it comes closest to the nominal violation count at every level. Between them sits the anomaly I found most instructive: the Student-t, at the everyday 95% level, is the *worst* forecaster in the table, because a tail heavy enough to cover crashes thins the shoulders where the 5% quantile lives. Tail heaviness is a budget, not a free improvement.

No static distribution passes the timing test. Every model fails Christoffersen's independence test at every confidence level, in-sample (the Week 5 volatility clustering check) and out-of-sample (Week 7) alike. Violations arrive in bursts, in late 2008 and March 2020, exactly as Cont's (2001) volatility clustering says they must and exactly as Christoffersen (1998) found for the static forecasts in his original application. A rolling window softens this but cannot fix it: refitting monthly on two years of data always arrives about a month late to a regime change. What stays wrong after the marginal is fixed is the arrival pattern of the violations, and that is a property of dynamics, which an independent-increments model discards by construction.

## 3. Limitations

There are five caveats I want to state plainly.

- One index, one frequency. Everything rests on daily log-returns of a single equity index. Cont (2001) documents that some stylised facts (the gain/loss asymmetry in particular) are equity-specific, so the parameter values, and especially the skew findings, should not be assumed to transfer to other asset classes. Weekly aggregation already lightens the tail (ν rises from 2.65 to 3.4 in Week 6), so conclusions are horizon-specific too.
- The model class is iid by design. I deliberately set out to test what a static marginal distribution can and cannot do, and the independence failures are the honest boundary of that design. The models do not condition on volatility state, and the backtest results quantify the cost of that choice rather than escape it.
- Sub-sample fits are sometimes weakly identified. In calm windows the NIG runs into its Gaussian-limit ridge, where α and δ grow large with enormous standard errors and only their product is pinned down. Roughly a third of the Week 6 quarterly fits sit on that ridge, and the rolling backtest windows regularly do as well; quantile forecasts remain well behaved there, but individual parameter readings from calm periods should not be over-interpreted.
- Backtest design choices matter. The 500-day window and 21-day refit cycle are conventional but not innocent: a shorter window adapts faster at the cost of noisier four-parameter fits, and results at the margins would shift. The Basel traffic-light figures are descriptive; I did not implement the full regulatory apparatus around the multiplier.
- The Bayesian arc represents the VG family by its Laplace special case. The full four-parameter VG was fitted by maximum likelihood only; its posterior and backtest behaviour are inferred from its family rather than sampled directly. Given that the full-sample VG sits between the Laplace and the NIG on every in-sample measure, this is unlikely to change any ranking, but it is a corner I cut knowingly.

## 4. Closing remark

I set out to ask whether heavy-tailed Lévy models materially improve tail risk measurement under the Basel metrics. They do, by amounts that would change a capital number: roughly a third more Expected Shortfall at the FRTB's own confidence level, and half the regulatory red-zone time out of sample. The same battery of tests also showed me what a better marginal cannot buy: the arrival times of the violations, which no independent-increments model can place. That division of labour, measured on one dataset with one consistent toolkit, is what this project contributes.

---
