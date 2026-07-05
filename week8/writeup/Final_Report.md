# Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns

*Darragh Moran. BSc Financial Mathematics and Economics, Applied Statistics Research Project, July 2026.*

## Abstract

Daily stock returns are not Gaussian, and the difference is expensive. In this project I fit five distributional models to 6,287 daily S&P 500 log-returns from January 2000 to December 2024: the Gaussian benchmark that underlies Black-Scholes, the Laplace, the Student-t, and two Lévy models built by Brownian subordination, the Variance-Gamma (Madan, Carr and Chang, 1998) and the Normal Inverse Gaussian. I estimated everything twice, by maximum likelihood and again in a Bayesian setting with the No-U-Turn Sampler, so the headline parameters come with full posteriors rather than point estimates. The fitted Student-t degrees of freedom is 2.648, with a 94% credible interval of (2.46, 2.87) sitting entirely above the variance singularity at 2: the tails are heavy, and the variance is finite. At the 99% level the Gaussian understates Expected Shortfall by 79.5% against the Student-t, and using the Gaussian number to set capital against the NIG, the only model that survived every goodness-of-fit and posterior predictive check I ran, leaves a 37% shortfall. A rolling out-of-sample backtest of one-day Value-at-Risk over 5,787 days, scored with the Christoffersen (1998) conditional coverage framework, confirms both halves of the story: the Gaussian produces 2.6 times the nominal number of 99% violations while the NIG cuts the excess to 1.6, yet every model, at every level, fails the independence test, because no static distribution can time the volatility clustering documented by Cont (2001). Choosing a heavier marginal therefore improves the measurement of tail risk by amounts that matter under the FRTB metrics (BCBS, 2013); what it cannot buy is the timing of that risk, which belongs to dynamics.

---

## 1. Introduction

Since the early 1970s, most of quantitative finance has run on one large assumption: daily returns are normally distributed. Black-Scholes is built on it, and much of classical risk management inherited it. The problem is that the assumption fails in exactly the region a risk manager is paid to care about. Real daily returns have a sharper peak, far heavier tails, and a persistent negative skew: crashes hit harder and more often than the equivalent rallies. Cont (2001) collected these regularities into a list of stylised facts that any serious model of returns has to reproduce, and made a structural observation I use as a design rule throughout this project: matching the location, scale, tail decay and asymmetry of real returns takes at least four parameters. The Gaussian has two.

The question I set out to answer is practical rather than philosophical. Regulators moved on from 99% Value-at-Risk to 97.5% Expected Shortfall as the market-risk capital metric under the Fundamental Review of the Trading Book (BCBS, 2013), and ES is a tail expectation, so it is far more sensitive to distributional shape than a quantile is. If the Gaussian understates the tail, it understates capital. By how much? And do the Lévy alternatives, which were built precisely to fix the tail, actually earn their extra parameters when tested properly?

My model set is a hierarchy of five. The Gaussian is the two-parameter baseline. The Laplace, also two parameters, swaps thin tails for exponential ones and turns out to be the most informative cheap upgrade in the project. The Student-t adds a third parameter that directly indexes tail heaviness. The Variance-Gamma (VG) of Madan, Carr and Chang (1998) and the Normal Inverse Gaussian (NIG) are the two four-parameter Lévy models, each built by running Brownian motion on a random clock, and each meeting Cont's four-parameter requirement exactly.

I tested this hierarchy four ways, and the paper follows that sequence. First, in-sample fit: maximum likelihood on the full 25-year sample, judged by information criteria and goodness-of-fit tests (Section 5). Second, a full Bayesian re-estimation with the No-U-Turn Sampler, which turns the important point estimates into posterior distributions and lets me make probability statements about the quantities that matter, such as whether the variance is finite (Sections 4.2 and 5.4). Third, checks on where the parameters move: across crises, across calendar structure, and against the VIX (Section 6). Fourth, and most demanding, an out-of-sample rolling backtest in which every model had to forecast tomorrow's VaR every day for 23 years and have the forecasts scored with the Christoffersen (1998) conditional coverage tests (Section 7).

The findings compress into three sentences. The Gaussian fails where failure is most expensive, understating 99% ES by 79.5% in-sample and producing 2.6 times the nominal number of 99% violations out of sample. Most of the improvement over the Gaussian comes from simply allowing exponential tails, but the NIG's extra structure is real: it is the only model that passed every distributional test I ran, in-sample and posterior predictive alike, and it came closest to nominal coverage out of sample. And no static distribution, however well shaped, passed the independence test, because violations arrive in bursts that only a dynamic volatility model could anticipate.

## 2. Data

The dataset is 6,287 daily log-returns on the S&P 500 index, r_t = log(S_t / S_{t−1}), from January 2000 to December 2024. I chose the 25-year window deliberately: it contains four distinct stress episodes, the dot-com crash (March 2000 to October 2002), the Global Financial Crisis (October 2007 to March 2009), the COVID-19 shock (February to June 2020) and the Fed rate-hike cycle (2022 to 2023). These four windows are shaded consistently in every figure and reappear throughout the analysis, because it turns out they are not four examples of the same thing.

![Figure 1. S&P 500 daily log-returns, 2000 to 2024, with the four shock windows shaded.](week2/figures/week2_trace.png)

*Figure 1. S&P 500 daily log-returns (2000 to 2024) with the four market shock windows shaded.*

Figure 1 shows the series. The unevenness is the first thing to notice: the calm of 2013 to 2019 barely looks like the same market as late 2008 or March 2020. The worst single day in the sample is a −12.8% log-return on 16 March 2020. Under a Gaussian fitted to this data, a move of that size is a twelve-standard-deviation event, which is a polite way of saying the model considers it impossible. The sample skewness is −0.39 and the sample excess kurtosis is 10.4, against 0 and 0 for a normal distribution.

Those numbers are the first two of Cont's (2001) stylised facts showing up on schedule: heavy tails and gain/loss asymmetry. The third, volatility clustering, is visible in Figure 1 as the banding of quiet and violent stretches, and it will matter enormously in Section 7. The fourth, aggregational Gaussianity, the tendency of returns to look more normal at longer horizons, appears in my weekly-frequency fits in Section 6.3. The whole project can be read as a test of how much of this list a static distribution can absorb, and the answer turns out to be: everything except the clustering.

## 3. The models

### 3.1 The hierarchy

Every model in the project is a distribution for the daily log-return. I fit them as independent and identically distributed draws, which is a deliberate and consequential choice: it isolates the question of distributional *shape* from the question of *dynamics*, and Section 7 measures the price of it.

The **Gaussian** N(μ, σ²) is the Black-Scholes benchmark and needs no introduction. Its tails decay like exp(−x²), which is the root of every failure documented below.

The **Student-t** t(ν, μ, σ) adds a degrees-of-freedom parameter ν that directly controls tail decay: the density falls off like a power law, |x|^{−(ν+1)}, and moments only exist up to order ν. This makes ν itself the object of interest. If the fitted ν lands below 4, kurtosis is infinite; below 3, skewness is undefined; below 2, even the variance diverges. Where ν falls relative to those thresholds is an empirical statement about how wild the market actually is.

The **Variance-Gamma** process of Madan, Carr and Chang (1998) is built by subordination: Brownian motion with drift θ and volatility σ is evaluated not at calendar time but at a random clock G(t), a Gamma process with unit mean rate and variance rate ν. Economically the clock is market activity: it runs fast in busy periods and slow in quiet ones. The resulting log-return distribution VG(μ, σ, θ, ν) has four parameters with clean jobs: μ locates, σ scales, θ signs the skew (negative for equity-style left skew), and ν sets the tail weight through the randomness of the clock. The VG is a pure-jump process of finite variation with an infinite arrival rate of small jumps, which Madan, Carr and Chang (1998) argue is exactly what lets it behave diffusively at fine scales while still producing occasional large moves. Its density involves a modified Bessel function of the second kind, K_λ, which shapes all the numerical work in Section 4.

The **Laplace** distribution earns its place in the hierarchy through the VG. Set θ = 0 and ν = 1 and the Gamma clock becomes an Exponential; the symmetric normal variance-mean mixture then collapses to a pair of back-to-back exponentials, which is the Laplace L(μ, b) with density exp(−|x−μ|/b)/(2b). It has the same parameter count as the Gaussian, exponential rather than Gaussian tails, an excess kurtosis of exactly 3, and a closed-form MLE (μ is the sample median, b the mean absolute deviation). It is the cheapest possible test of the proposition that the tails, not anything more exotic, are what the Gaussian gets wrong. From Section 5.4 onward the Laplace also stands in for the VG family in the Bayesian and backtesting work, a representation choice I return to in the limitations.

The **Normal Inverse Gaussian** NIG(μ, α, β, δ) is the second subordinated model: the same construction with an Inverse Gaussian clock instead of a Gamma one. Its density is

f(x) = αδ · exp(δγ + β(x−μ)) · K₁(α·q(x)) / (π·q(x)),  where q(x) = √(δ² + (x−μ)²) and γ = √(α² − β²).

The parameters again divide the labour: α controls tail decay (larger α means lighter tails), β the asymmetry subject to |β| < α, δ the scale and μ the location. The NIG's tails are "semi-heavy", heavier than exponential but lighter than any power law, and all its moments are finite, which turns out to matter when I try to check the fit through moments in Section 5.5.

### 3.2 What the hierarchy is for

The five models are ordered so that each comparison isolates one question. Gaussian against Laplace, at equal parameter count, asks whether the tail shape alone carries the improvement. Laplace against Student-t asks whether power-law tails beat exponential ones. Student-t against the four-parameter VG and NIG asks whether the asymmetry and the subordination structure add anything a symmetric heavy tail cannot. And VG against NIG asks whether the choice of random clock, Gamma against Inverse Gaussian, is detectable in 25 years of data. Every one of these questions gets a quantitative answer in Section 5.

## 4. Estimation

### 4.1 Maximum likelihood

The Gaussian and Laplace have closed-form MLEs. The Student-t is a routine three-parameter numerical optimisation. The two Lévy models need more care, and the care is mostly about the Bessel function.

Both the VG and NIG log-densities contain log K_λ(z), which underflows catastrophically in the tails if evaluated directly, precisely where a heavy-tail model does its work. I used the exponentially scaled Bessel function throughout: log K_λ(z) = log kve(λ, z) − z, which stays finite far into the tail. For the VG I optimised over (σ, θ, ν, μ) with L-BFGS-B from three starting points, keeping the best finite optimum. The NIG has the awkward constraint α > |β|, which I removed by reparametrising to ξ = β/α ∈ (−1, 1): box bounds on ξ then enforce the constraint automatically for any positive α, and the standard errors transform back to the natural parameters through the Jacobian. All standard errors come from the observed Fisher information, computed with a central-difference Hessian whose step sizes scale with each parameter's magnitude, since the parameters span five orders of magnitude.

Risk measures follow directly. For the Gaussian, Laplace and Student-t, VaR and ES at each level have closed forms; the Laplace ES is particularly tidy, sitting exactly one scale parameter b below the VaR because its exponential lower tail is memoryless. For the VG and NIG I computed VaR and ES by Monte Carlo from their mixture representations (a Gamma-Normal mixture for the VG, an Inverse-Gaussian-Normal mixture for the NIG) with one million draws, which makes the Monte Carlo error negligible at the levels I report.

### 4.2 Bayesian estimation

Maximum likelihood gives one best value per parameter and a curvature-based standard error. For the parameters this project actually cares about, that is not quite enough. The headline claim of Section 5.1 is that ν is near 2.6, and the interesting question is not the point estimate but how much posterior probability sits below the variance singularity at ν = 2. That is a question for a posterior, so I refitted four of the models (Gaussian, Laplace, Student-t and the full four-parameter NIG) in PyMC and sampled with the No-U-Turn Sampler.

The priors are weakly informative, anchored loosely on the MLE but widened well beyond the plausible range: μ ~ Normal(0, 0.001) for every model, HalfNormal(0.02) for every scale parameter, ν ~ Gamma(2, 0.1) for the Student-t, and for the NIG a reparametrisation to (γ, β) with γ ~ Gamma(4, 4/52) and β ~ Normal(0, 15), recovering α = √(γ² + β²) so the α > |β| constraint holds by construction. The ν prior deserves a sentence, because it embodies the one methodological decision I thought hardest about. It would have been easy to restrict ν > 2 and guarantee a finite variance. I did not, because that restriction answers my main question before the data gets a vote. The prior I used has mean 20 and if anything leans towards light tails; any pull down towards ν ≈ 2.6 has to come from the likelihood.

I accepted each prior only after a prior predictive check: simulate return histories from the prior alone and confirm they stay within plausible bounds. The fraction of simulated days beyond ±25%, a move larger than anything in the sample, was between 0.0001% and 0.08% across the four models. Had any prior produced absurd histories I would have tightened it; none did.

The NIG was the real implementation work. Its log-density contains K₁, and NUTS needs gradients through it. PyTensor's Bessel implementation carries correct analytic derivatives, so I wrote the NIG log-density directly as a custom distribution using the same scaled-Bessel trick as the MLE code, checked it against an independent reference implementation to within 10^{−15}, and let the sampler differentiate through it. This avoids the alternative of introducing 6,287 latent mixing variables, one per observation, which would have blown the dimension of the posterior up for no statistical gain.

Convergence was clean everywhere: four chains per model, split R-hat of 1.00 on every parameter, bulk effective sample sizes above 2,000, tail effective sample sizes above 2,100 (the tail figure is the one that matters here, since tail quantiles are what I report), and Monte Carlo standard errors small relative to posterior spread. Figure 5 in Section 5.4 shows the NIG traces, the hardest of the four; the chains sit on top of each other with no drift.

## 5. In-sample results

### 5.1 Parameter estimates

**Table 1. Full-sample MLEs, 6,287 daily returns. Standard errors in parentheses.**

| Model | Parameters (SE) | Log-lik |
|-------|-----------------|---------|
| Gaussian | μ = +0.000223 (0.000154), σ = 0.012234 (0.000109) | 18,764.3 |
| Laplace | μ = +0.000602 (0.000102), b = 0.008078 (0.000102) | 19,649.9 |
| Student-t | ν = 2.648 (0.110), μ = +0.000656 (0.000110), σ = 0.007077 (0.000128) | 19,666.7 |
| VG | σ = 0.011629 (0.000163), θ = −0.000660 (0.000149), ν = 1.174 (0.043), μ = +0.000883 (0.000023) | 19,665.5 |
| NIG | α = 52.34 (2.79), β = −6.10 (1.50), δ = 0.007574 (0.000217), μ = +0.001111 (0.000150) | 19,691.7 |

Three estimates carry the story. The Student-t ν of 2.648 sits below 4 and below 3: the data is consistent with infinite kurtosis and undefined skewness, though the 95% confidence interval (2.43, 2.87) keeps the variance finite. The VG θ and NIG β are both negative, the left skew of Cont's (2001) gain/loss asymmetry showing up as fitted parameters. And every non-Gaussian log-likelihood beats the Gaussian's by around nine hundred units, an almost absurd margin for one or two extra parameters.

One estimate deserves a cross-reference. Madan, Carr and Chang (1998) fitted the VG to S&P 500 data from 1992 to 1994 and found the statistical process nearly symmetric with modest kurtosis. My 2000 to 2024 sample gives a VG ν of 1.17 against their 0.002 in comparable units, and a clearly negative θ. Twenty-five years containing four crises is a very different market from the placid early 1990s, which is rather the point of the sample.

### 5.2 Which model fits

**Table 2. Model comparison. ΔAIC is against the Gaussian; lower is better. KS is Kolmogorov-Smirnov on the full sample.**

| Model | k | AIC | BIC | ΔAIC | KS D | KS p |
|-------|---|-----|-----|------|------|------|
| Gaussian | 2 | −37,524.5 | −37,511.0 | 0 | 0.0938 | <0.001 |
| Laplace | 2 | −39,295.7 | −39,282.2 | −1,771.2 | 0.0180 | 0.034 |
| Student-t | 3 | −39,327.5 | −39,307.2 | −1,802.9 | 0.0182 | 0.030 |
| VG | 4 | −39,322.9 | −39,296.0 | −1,798.4 | 0.0129 | 0.250 |
| NIG | 4 | −39,375.3 | −39,348.4 | −1,850.8 | 0.0113 | 0.399 |

![Figure 2. Return histogram with all five fitted densities.](week3/figures/week3_density_all_models.png)

*Figure 2. The return histogram with all five fitted PDFs. The four non-Gaussian curves are nearly indistinguishable at this scale; the Gaussian is visibly too flat at the peak and too fat in the shoulders.*

![Figure 3. QQ plots for all five models.](week3/figures/week3_qq_all_models.png)

*Figure 3. QQ plots. The Gaussian S-shape (top left) is the signature of heavy tails; the NIG sits closest to the diagonal throughout.*

The answers to the four hierarchy questions of Section 3.2 read straight off Table 2. Tail shape alone is worth ΔAIC = −1,771: the Laplace, with the Gaussian's parameter count, captures 96% of the best model's total gain. That is the single most clarifying number in the project. Power-law tails add only a little over exponential ones (Student-t against Laplace, 17 AIC units). The full VG is actually slightly worse than the Student-t on AIC, its two extra parameters not quite paying rent. But the NIG beats everything, by 26 log-likelihood units over the VG on identical parameter counts, so the choice of random clock is detectable: the Inverse Gaussian subordinator captures something in this data that the Gamma one cannot. On the KS test only the two Lévy models survive; the NIG (D = 0.0113, p = 0.40) fits best of all.

### 5.3 What it costs to be wrong

**Table 3. Daily VaR and ES under each fitted model. Negative values are losses. 97.5% is the FRTB ES level.**

| Level | Gaussian | Laplace | Student-t | VG | NIG |
|-------|----------|---------|-----------|----|----|
| VaR 95% | −1.99% | −1.80% | −1.69% | −1.92% | −1.88% |
| ES 95% | −2.50% | −2.61% | −3.00% | −2.82% | −3.08% |
| VaR 97.5% | −2.38% | −2.36% | −2.37% | −2.54% | −2.65% |
| ES 97.5% | −2.84% | −3.17% | −4.02% | −3.45% | −3.94% |
| VaR 99% | −2.82% | −3.10% | −3.52% | −3.36% | −3.77% |
| ES 99% | −3.24% | −3.91% | −5.81% | −4.28% | −5.19% |

![Figure 4. VaR and ES at the three levels for all five models.](week3/figures/week3_risk_comparison.png)

*Figure 4. VaR and ES at 95%, 97.5% (the FRTB level) and 99%. The models agree closely on VaR at moderate levels and diverge enormously on ES in the deep tail.*

Two patterns in Table 3 matter. First, the models nearly agree on 95% and 97.5% VaR, the quantities everyone used to report, and then diverge violently on ES: at 99%, the span runs from −3.24% (Gaussian) to −5.81% (Student-t), a 79.5% gap. The regulatory shift from VaR to ES (BCBS, 2013) is exactly the shift from the metric these models agree on to the metric they disagree on, which is why the distributional choice suddenly has capital consequences. Second, the shortfall is already material at the level the regulation actually uses: at 97.5%, the Gaussian ES of −2.84% sits 28% below the NIG's −3.94%. A desk that sets ES-based capital from the Gaussian holds roughly 37% too little against the NIG benchmark at 99%, and the deficit does not wait for the deep tail to appear.

There is one instructive oddity in the VaR rows: at 95%, the Gaussian VaR is the most conservative of the five. The Student-t, having committed its probability mass to the far tail, has less left in the shoulders, so its 5% quantile sits closer to zero. I flag it here because the same geometry resurfaces out of sample in Section 7, where it stops being a curiosity and becomes a failed backtest.

### 5.4 The Bayesian answers

**Table 4. Posterior summaries against the MLE. Intervals are 94% highest-density intervals.**

| Model | Parameter | Posterior mean | 94% HDI | MLE |
|-------|-----------|----------------|---------|-----|
| Gaussian | μ | +0.000215 | (−0.00007, +0.00049) | +0.000223 |
| | σ | 0.012233 | (0.01202, 0.01244) | 0.012234 |
| Laplace | μ | +0.000599 | (+0.00040, +0.00080) | +0.000602 |
| | b | 0.008081 | (0.00790, 0.00827) | 0.008078 |
| Student-t | μ | +0.000647 | (+0.00043, +0.00084) | +0.000656 |
| | σ | 0.007084 | (0.00686, 0.00733) | 0.007077 |
| | ν | 2.658 | (2.46, 2.87) | 2.648 |
| NIG | μ | +0.001080 | (+0.00081, +0.00135) | +0.001111 |
| | α | 52.35 | (47.1, 57.7) | 52.34 |
| | β | −5.86 | (−8.75, −3.25) | −6.10 |
| | δ | 0.007575 | (0.00717, 0.00798) | 0.007574 |

![Figure 5. NUTS diagnostics for the NIG.](week4/figures/week4_trace_nig.png)

*Figure 5. NUTS diagnostics for the NIG, the hardest model of the four: posterior densities per chain against the MLE (left) and the raw chains (right). Four parameters and a Bessel function in the density, and the mixing is as clean as the Gaussian's.*

With 6,287 observations and loose priors, the posteriors reproduce the MLEs almost exactly, which is reassurance about both rather than news. The two results that justify the Bayesian effort are the intervals. The entire 94% interval for ν, (2.46, 2.87), sits above 2: the finite variance is not a point estimate that happens to land north of the singularity, it holds across the whole credible range, under a prior that leaned the other way. And the entire interval for the NIG's β, (−8.75, −3.25), sits below zero: the left skew is a stable feature of the data, not something that could round away.

### 5.5 Posterior predictive checks

A posterior is only useful if the model behind it can regenerate the data. For each model I drew 1,000 parameter values from the posterior, simulated a full 6,287-day replicate history from each, and scored the real data against the replicate cloud with two-sided posterior predictive p-values: a p near 0.5 means the model produces data like ours comfortably, a p near 0 or 1 means the real data sits outside almost every replicate.

**Table 5. Posterior predictive p-values. Entries marked † fall outside (0.05, 0.95). ACF²(1) is the lag-1 autocorrelation of squared returns.**

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

![Figure 6. Posterior predictive tail risk for the four models.](week5/figures/week5_ppc_risk.png)

*Figure 6. Posterior predictive VaR and ES at the three levels. Each marker is a model's replicate median with its 94% band; dashed lines are the observed values. The Gaussian bands sit clear of the truth beyond 95%; the NIG covers it on all six statistics.*

The table is the project's in-sample verdict in one place. The Gaussian fails everything beyond the centre: across a thousand replicate histories its worst simulated day never got past about −5%, against the real −12.8%. The Laplace closes roughly half the tail gap and then stops, at exactly the excess kurtosis of 3 it owns by construction. The Student-t covers the deep tail quantiles but its moment checks are degenerate: with ν near 2.66 the replicate kurtosis ranges into the thousands because the population moment does not exist, so a moment-based pass or fail from that model means nothing, and its skew "pass" is just a symmetric model with wild third moments straddling everything. The NIG passes all eleven marginal checks with tight, meaningful bands, including the skew (replicates centred at −0.54 against the observed −0.39) and the FRTB 97.5% ES.

And every model, the NIG included, fails the twelfth statistic identically. The observed lag-1 autocorrelation of squared returns is 0.32; every model's replicate band sits on zero, because independent draws cannot cluster, whatever their marginal. This is Cont's (2001) volatility clustering formally locating the boundary of the entire model class, and it is the reason the out-of-sample test in Section 7 exists.

## 6. Parameters that will not sit still

Everything in Section 5 assumed one distribution for 25 years. This section tests that assumption from three directions and finds it wrong in interesting, structured ways.

### 6.1 Two kinds of crisis

Fitting each model separately inside the four shock windows splits the crises into two families.

**Table 6. Sub-period estimates, each window fitted independently.**

| Period | n | Gaussian σ (ann.) | Student-t ν | VG ν | VG θ | NIG α | NIG β |
|--------|---|-------------------|-------------|------|------|-------|-------|
| Full sample | 6,287 | 19.4% | 2.65 | 1.17 | −0.0007 | 52.3 | −6.1 |
| Dot-com crash | 671 | 23.5% | 6.53 | 0.40 | +0.0019 | 98.6 | +10.0 |
| GFC | 378 | 38.3% | 2.61 | 1.25 | −0.0030 | 25.9 | −2.7 |
| COVID-19 | 104 | 50.5% | 2.29 | 1.91 | −0.0050 | 17.8 | −4.2 |
| Fed rate hikes | 501 | 19.5% | 6.53 | 0.49 | −0.0000 | 111.9 | −3.9 |

![Figure 7. Sub-period scale and tail parameters.](week3/figures/week3_subperiod_params.png)

*Figure 7. Annualised scale by period and model (left) and the tail parameters (right); the dashed line is the ν = 2 variance singularity.*

The GFC and COVID windows return ν near 2.3 to 2.6 and NIG α of 18 to 26: clusters of extreme single days, tails at their heaviest exactly when volatility is also at its highest. The dot-com unwind and the rate-hike cycle return ν near 6.5 and α near 100: elevated volatility with nearly Gaussian shape, grinding drawdowns rather than convulsions. Volatility alone cannot distinguish the two families; the tail parameters can. Consistent with this, the AIC advantage of the Lévy models over the Gaussian is 73 to 79 units inside the GFC against 15 to 23 in the calm periods: the models earn their keep precisely when the market is worst.

### 6.2 Tracking the parameters through time

Year-by-year VG and NIG fits (with μ pinned to zero, since 250 observations cannot identify a daily drift) turn the parameters into time series. The scale parameters trace the crises faithfully, the VG's annualised σ peaking at 39.8% in 2008 and 31.3% in 2020 against 6.5% in 2017. Tail heaviness is itself time-varying: NIG α bottoms out near 22 in exactly 2008 and 2020, and the most negative skew readings land on the dot-com unwind, the GFC and the 2022 drawdown. In the calmest years the NIG runs into its Gaussian limit, where α and δ grow together essentially unbounded, individually huge standard errors, only their product identified. That ridge is a recurring numerical character in this project: roughly a third of quarterly fits sit on it, and it later broke a library quantile routine in the backtest (Section 7.1).

At quarterly frequency there is enough data to regress the parameters on the VIX. The NIG scale δ tracks it closely (R² = 0.64 in logs, t = 9.7 across the 67 well-identified quarters), and the skew parameters show a significant VIX relationship at quarterly frequency that annual fits lacked the power to find. The tail-decay parameters, by contrast, stay flat against the VIX at every frequency I tried. The VIX tells you how wide the distribution is, not how heavy or lopsided its tails are, and ES-based capital is driven by the latter.

### 6.3 The calendar

Three calendar findings round out the picture of where the risk lives.

![Figure 8. Volatility and tail parameter by weekday.](week6/figures/week6_weekday_params.png)

*Figure 8. Annualised volatility by weekday with 95% intervals (left) and the per-day Student-t ν against the full-sample 2.648 (right).*

The week has a shape, and it is a staircase. Monday, which carries the whole weekend inside it, is the most volatile day (21.4% annualised) and has much the heaviest tail (ν = 2.17, brushing the variance boundary); Friday is the calmest and lightest (17.8%, ν = 3.10). Likelihood-ratio tests settle the right resolution: a Monday / midweek / Friday grouping beats a pooled week decisively (LR = 42.9 on 4 df, p < 0.0001), and five separate days add nothing over the three groups (p = 0.59). Second, splitting close-to-close returns at the open shows the closed market holds only about 20% of the variance but nearly all of the jumps: overnight ν = 2.14 with excess kurtosis 36, against intraday ν = 2.79 and kurtosis 5. The index gaps on information that arrives while trading is shut, which is also what Monday's weekend tail is. Third, weekly returns fit with ν = 3.38 against the daily 2.65: aggregation lightens the tail exactly as Cont's (2001) aggregational Gaussianity predicts, so every tail conclusion in this paper is specific to the daily horizon.

One null result is worth reporting because I expected the opposite: quarterly earnings seasons leave index-level volatility unchanged to the decimal (16.5%, 16.4%, 16.5% annualised before, during and after). Single-name earnings surprises evidently diversify away in a 500-stock index; the tail risk that survives aggregation is macro.

## 7. The out-of-sample backtest

Everything so far has been in-sample. For the final test I made the models work for a living: each one refitted on a rolling window and asked, every day, for tomorrow's Value-at-Risk, with the forecasts scored by the realised returns they had never seen.

### 7.1 Design

I chose a rolling window of 500 trading days, about two years: long enough for a stable four-parameter NIG fit, short enough to adapt across regimes. I refitted every 21 trading days, roughly the monthly cycle a risk desk would run, and held parameters fixed between refits. Each day the previous fit issued one-day-ahead VaR at 95%, 97.5% and 99%, taken as the corresponding quantile of the fitted distribution, and a violation is a realised return strictly below the forecast. That gives 5,787 out-of-sample days per model, January 2002 to December 2024, with 289.4, 144.7 and 57.9 violations expected at the three levels. The models are the Bayesian four, with the Laplace again carrying the VG family.

One numerical fight worth recording: on calm windows the rolling NIG regularly lands on the Gaussian-limit ridge from Section 6.2, and the library quantile routine refused to converge there, so I solved the quantile myself, bracketing the root with the NIG's own mean and standard deviation and running Brent's method on the CDF. On the ridge the quantile smoothly approaches the Gaussian one, which is the behaviour I wanted.

The scoring framework is Christoffersen (1998). His starting point is that testing the average violation rate is not enough: a model can be right on average and still fail in runs, underestimating risk exactly when accuracy matters most. He therefore tests two properties. Unconditional coverage (LR_uc, the Kupiec test) asks whether the violation *rate* equals the nominal 5%, 2.5% or 1%; independence (LR_ind) asks whether a violation today is more likely given one yesterday, against a first-order Markov alternative; and conditional coverage (LR_cc = LR_uc + LR_ind, chi-squared with 2 df) requires both at once. A correct model needs the right number of violations arriving at unpredictable times.

### 7.2 Coverage

**Table 7. Rolling backtest over 5,787 out-of-sample days. Hits are observed violations; p-values from the chi-squared limits.**

| Model | Level | Expected | Hits | LR_uc | p_uc | LR_ind | p_ind | p_cc |
|-------|-------|----------|------|-------|------|--------|-------|------|
| Gaussian | 95% | 289 | 308 | 1.2 | 0.265 | 47.2 | <0.0001 | <0.0001 |
| Gaussian | 97.5% | 145 | 223 | 37.4 | <0.0001 | 39.7 | <0.0001 | <0.0001 |
| Gaussian | 99% | 58 | 152 | 106.9 | <0.0001 | 36.1 | <0.0001 | <0.0001 |
| Laplace | 95% | 289 | 323 | 4.0 | 0.046 | 35.7 | <0.0001 | <0.0001 |
| Laplace | 97.5% | 145 | 193 | 15.0 | 0.0001 | 34.6 | <0.0001 | <0.0001 |
| Laplace | 99% | 58 | 110 | 37.5 | <0.0001 | 28.0 | <0.0001 | <0.0001 |
| Student-t | 95% | 289 | 358 | 16.0 | 0.0001 | 39.0 | <0.0001 | <0.0001 |
| Student-t | 97.5% | 145 | 218 | 33.1 | <0.0001 | 35.9 | <0.0001 | <0.0001 |
| Student-t | 99% | 58 | 104 | 30.0 | <0.0001 | 18.6 | <0.0001 | <0.0001 |
| NIG | 95% | 289 | 313 | 2.0 | 0.159 | 37.5 | <0.0001 | <0.0001 |
| NIG | 97.5% | 145 | 177 | 6.9 | 0.009 | 31.9 | <0.0001 | <0.0001 |
| NIG | 99% | 58 | 91 | 16.3 | 0.0001 | 19.3 | <0.0001 | <0.0001 |

![Figure 9. Rolling 99% VaR forecasts and violations.](week7/figures/week7_var_series.png)

*Figure 9. Rolling one-day 99% VaR forecasts against realised returns, violations marked. The crosses bunch inside the GFC and COVID windows for every model.*

At 95%, the Gaussian passes unconditional coverage comfortably (p = 0.27) and the Student-t is the worst model in the table (358 hits against 289). This is the Section 5.3 geometry coming due: a tail heavy enough to cover crashes thins the shoulders where the everyday 5% quantile lives, so at moderate levels the heavy-tailed model under-covers while the thin-tailed one is fine. Tail heaviness is a budget, not a free improvement.

At 99% the ordering flips and the Gaussian collapses: 152 violations against 58 expected, a factor of 2.6, LR_uc = 106.9. This is the out-of-sample twin of the 79.5% in-sample ES gap. The NIG comes closest (91 hits, a factor of 1.57), then the Student-t and Laplace, and even the NIG is rejected on pure coverage: a static tail fitted to the last two years is still too short when the regime breaks.

### 7.3 Timing

The LR_ind column has no survivors: every model, at every level, is rejected with statistics between 18.6 and 47.2. The mechanism shows in the transition counts. After a Gaussian 95% violation, the chance of another violation the next day was about 16%, three times the unconditional rate. The models are not wrong at random times; they are wrong in runs.

![Figure 10. Cumulative violations against the nominal rate.](week7/figures/week7_cumulative_hits.png)

*Figure 10. Cumulative violations at 95% (left) and 99% (right) against the nominal accrual (dashed). The staircases are flat for years and then vertical in late 2008 and March 2020.*

Figure 10 is the independence failure made visible, and it is also the exact pattern Christoffersen (1998) documented for static interval forecasts in his original exchange-rate application: right on average, wrong in clusters. It is the out-of-sample confirmation of the posterior predictive ACF failure in Section 5.5, now with real forecasts and real money implications. A rolling window softens the problem but cannot fix it; refitting monthly on two years of data always arrives about a month late to a regime change.

### 7.4 The regulatory view

Basel's backtesting regime keys the capital multiplier off the count of 99% violations in the trailing 250 days: green up to 4, yellow from 5 to 9, red at 10 or more, where the internal model is presumed broken.

**Table 8. Share of out-of-sample days in each Basel zone (99% VaR, trailing 250 days), and the FRTB-style ES check at 97.5% (mean realised loss on breach days over the mean ES the model forecast for those days).**

| Model | Green | Yellow | Red | Worst 250-day count | ES shortfall ratio |
|-------|-------|--------|-----|---------------------|--------------------|
| Gaussian | 53.2% | 16.0% | 30.8% | 32 | 1.33 |
| Laplace | 63.8% | 17.5% | 18.7% | 28 | 1.20 |
| Student-t | 65.3% | 16.7% | 17.9% | 22 | 0.97 |
| NIG | 66.6% | 19.3% | 14.1% | 15 | 1.07 |

![Figure 11. Basel traffic light: rolling 250-day violation counts.](week7/figures/week7_basel_traffic.png)

*Figure 11. Trailing 250-day count of 99% violations per model over the Basel green, yellow and red zones.*

A Gaussian desk spends almost a third of two decades in the red zone, and its worst 250-day window contains 32 violations, more than three times the red threshold. The ES column says the same thing in FRTB units: on the days the Gaussian's 97.5% VaR was breached, the realised losses ran 33% past the ES the model had booked, and ES at 97.5% is precisely the number FRTB capital is built on (BCBS, 2013). The Lévy tails roughly halve the red-zone time, the NIG's worst window is half the Gaussian's, and the NIG prices its own breaches to within 7%. The Student-t's promised shortfall is actually adequate (ratio 0.97), the mirror image of its 95% coverage failure: deep-tail honesty bought at the shoulders. But no model stays out of the red through 2008 and 2020, which is the traffic-light restatement of the independence failure.

## 8. Conclusions

### 8.1 The scorecard

**Table 9. The five models across the whole project. The VG's Bayesian and backtest columns are carried by the Laplace, its symmetric special case.**

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

### 8.2 What I found

The Gaussian assumption fails where it is most expensive. The fitted ν of 2.648, with its whole posterior interval above the variance singularity, says the tails are heavy as a stable population property. The price of ignoring that is concentrated in the tail the regulation cares about: 79.5% of missing 99% ES in-sample, 2.6 times the nominal violations out of sample, 31% of two decades in the Basel red zone, and breach-day losses one third beyond the booked ES.

Most of the gain is exponential tails; the last part is the NIG's. The Laplace, at the Gaussian's own parameter count, captures 96% of the best model's AIC improvement, so the bulk of the fix is simply admitting that tails are not Gaussian. The NIG earns the remainder with real structure: the only KS pass worth the name, every posterior predictive check including skew and the FRTB ES, a posterior β negative across its entire interval, and the closest-to-nominal coverage at every backtest level. Between them sits the anomaly I found most instructive: the Student-t, the classic heavy-tail fix, is the worst 95% forecaster in the table because its tail is bought from its shoulders.

No static distribution passes the timing test. Every model failed the independence test at every level, in-sample as the posterior predictive ACF check and out-of-sample as clustered violations in 2008 and 2020, exactly as Cont (2001) documents and exactly as Christoffersen (1998) found for the static forecasts in his original application. What stays wrong after the marginal is fixed is the arrival pattern of the violations, and that is a property of dynamics, which an independent-increments model discards by construction.

### 8.3 Limitations

There are five caveats I want to state plainly. Everything rests on daily log-returns of one equity index, and both the skew findings (equity-specific per Cont, 2001) and the tail estimates (which lighten to ν = 3.4 at weekly frequency) are specific to that choice. The model class is iid by design; I set out to measure what a static marginal can and cannot do, and the independence failures are that design's honest boundary, quantified rather than escaped. Sub-sample NIG fits are sometimes only weakly identified, sitting on the Gaussian-limit ridge where individual parameters mean little even though quantile forecasts stay well behaved. The backtest's 500-day window and 21-day refit cycle are conventional but not innocent, and the Basel traffic-light figures are descriptive; I did not implement the full regulatory apparatus around the multiplier. And the Bayesian and backtest columns represent the VG family by its Laplace special case; given that the full VG sits between the Laplace and the NIG on every in-sample measure this is unlikely to change any ranking, but it is a corner I cut knowingly.

### 8.4 Closing remark

I set out to ask whether heavy-tailed Lévy models materially improve tail risk measurement under the Basel metrics. They do, by amounts that would change a capital number: roughly a third more Expected Shortfall at the FRTB's own confidence level, and half the regulatory red-zone time out of sample. The same battery of tests also showed me what a better marginal cannot buy: the arrival times of the violations, which no independent-increments model can place. A natural continuation would put these Lévy distributions to work as the innovation law inside a conditional volatility model, asking whether the NIG's tail and a dynamic scale together can pass the one test nothing in this paper passed. That division of labour, the size of tail risk against its timing, measured on one dataset with one consistent toolkit, is what this project contributes.

## References

Basel Committee on Banking Supervision (2013). *Fundamental Review of the Trading Book: A Revised Market Risk Framework.* Bank for International Settlements.

Christoffersen, P. F. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.

Cont, R. (2001). Empirical Properties of Asset Returns: Stylised Facts and Statistical Issues. *Quantitative Finance*, 1(2), 223–236.

Madan, D. B., Carr, P. P. and Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *European Finance Review*, 2(1), 79–105.
