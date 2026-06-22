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

The parameter interpretations are precise. Madan, Carr and Chang (1998) derive the central moments explicitly. The third central moment is proportional to (2θ³ν² + 3σ²θν)t, so θ = 0 implies zero skewness and the sign of θ determines the direction of skew. When θ = 0, ν is directly the percentage excess kurtosis. All moments are finite, which means VaR and Expected Shortfall are well-defined under the model.

Unlike Brownian motion, the VG process has no continuous martingale component: it is a pure jump process of finite variation. It can be written as the difference of two independent increasing gamma processes, one for upward moves and one for downward. The relative sizes of those two processes determine the skew. The VG process also inherits from the gamma process an infinite arrival rate of small jumps, which is what makes it behave like Brownian motion at fine scales while still allowing occasional large discontinuities. Madan, Carr and Chang (1998) argue this makes the VG more realistic than Poisson jump-diffusion models, which require a finite jump rate and therefore need a separate diffusion component.

### 4.2 Estimation and Empirical Results

Madan, Carr and Chang (1998) estimated the VG model on S&P 500 data from January 1992 to September 1994 by maximum likelihood. The density involves a modified Bessel function of the second kind, available in standard numerical libraries, so MLE is a straightforward L-BFGS-B optimisation. The results are useful for setting expectations.

On the statistical process, the lognormal model is strongly rejected against the symmetric VG with a χ²(1) statistic of 83.94. The skewness parameter θ is statistically insignificant, the S&P 500 daily return distribution is approximately symmetric. Excess kurtosis is the real story: estimated ν = 0.002 corresponds to a daily kurtosis of roughly 5.19, against 3 for the Gaussian. Adding the full asymmetric VG on top of the symmetric one makes no improvement for the statistical process.

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
