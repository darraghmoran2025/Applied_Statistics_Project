# Week 5: Posterior predictive checks and convergence diagnostics

## 1. How the check works

I have a posterior over parameters. For each draw from that posterior I generate a complete replicate dataset the same size as the real one, 6,287 returns, from that draw's likelihood. Do this for a thousand draws and I have a thousand fake histories, each one a fair sample of what the fitted model thinks the market should produce. Then I compare the real data against that cloud.

I did this in numpy and scipy rather than rebuilding the models in PyMC, because Week 4 already saved every posterior draw to CSV. Reading those back and sampling the likelihood at each one gives the same thing without re-running the sampler.

Value at risk and expected shortfall at 95% and 99% are the risk numbers the project actually reports, computed on each replicate exactly as they are on the real series. For each statistic I get its spread across the thousand replicates, and I score the fit with a two-sided posterior predictive p-value: the probability that a replicate is at least as extreme as the observed value. A p near 0.5 means the model produces data with that feature comfortably. A p near 0 or 1 means the real data sits out past almost every replicate, and the model cannot reach it. I flag anything below 0.05 or above 0.95.

---

## 2. Gaussian

Figure 1 shows the Gaussian's replicate densities in blue with the real data in black, and the right panel zooms into the left tail on a log scale. The Gaussian was fitted to a daily volatility near 1.2% and it reproduces that. The tail panel is where it falls apart. The real density still has visible mass out past −5%, and the Gaussian replicates have effectively none.

The statistics say the same thing. Observed excess kurtosis is 10.4; the Gaussian replicates run between −0.11 and +0.12, as a normal distribution must. The real worst day is −12.8%; the worst day the Gaussian manages across a thousand replicate histories sits around −5%. Both p-values are zero. The risk numbers carry the same verdict in the units the project cares about: observed 99% expected shortfall is −5.07%, and the Gaussian replicates put it near −3.24%. The 79.5% shortfall gap from Week 2 is not an artefact of one estimator. It is the Gaussian being structurally unable to generate the losses the market actually delivered.

At 95% value at risk the Gaussian is not too thin but slightly too fat: observed −1.88% against a replicate −1.99%. That is the flip side of forcing a single bell curve onto a peaked, heavy-tailed sample. To cover the fat middle the fitted variance has to stretch, which pushes the ordinary 95% quantile out a little too far. It is only when you go further into the tail, to 99% and to expected shortfall, that the missing mass shows up and the Gaussian collapses.

![Figure 1. Posterior predictive density for the Gaussian.](../figures/week5_ppc_gaussian.png)

*Figure 1. Posterior predictive check for the Gaussian. Left: replicate return densities (blue) against the observed data (black). Right: the left tail on a log scale, where the observed mass past −5% has no counterpart in the replicates.*

---

## 3. Laplace

The Laplace is my Variance-Gamma model. Figure 2 is a clear step up from the Gaussian. Its tails reach much further, and its 95% value at risk now matches the data (p = 0.13).

It still fails the deep tail. Observed excess kurtosis is 10.4 against replicates near 2.9, which is exactly the kurtosis a Laplace has by construction. Its 99% expected shortfall lands near −3.89% against the observed −5.07%, better than the Gaussian's −3.24%.

![Figure 2. Posterior predictive density for the Laplace.](../figures/week5_ppc_laplace.png)

*Figure 2. Posterior predictive check for the Laplace. The exponential tails reach much further than the Gaussian's but still fall short of the observed mass in the deep left tail.*

---

## 4. Student-t

The Student-t is the first model where it gets the deep tail correct. Its 99% value at risk matches the data almost exactly (p = 0.94), and its 99% expected shortfall covers the observed −5.07% comfortably.

Observe the excess kurtosis in Figure 3 and Table 1. The replicate median is about 49, and the 94% band runs from 14 all the way to 1,200. The observed 10.4 sits below almost all of it. The posterior for ν sits around 2.66, and a Student-t only has finite kurtosis when ν exceeds 4.

For a genuinely heavy-tailed model, moment-based checks like kurtosis are close to meaningless, because the moment they measure does not exist in the population. The quantile-based checks, value at risk and expected shortfall, stay well defined no matter how heavy the tail, and those are the ones the Student-t passes.

![Figure 3. Posterior predictive density for the Student-t.](../figures/week5_ppc_student_t.png)

*Figure 3. Posterior predictive check for the Student-t. The tails now cover the observed data, but with ν near 2.66 the replicate tails are so heavy that the densest part of the figure is dominated by occasional extreme replicates.*

---

## 5. NIG

The NIG passes our checks. Every statistic I tested falls inside its replicate band: standard deviation (p = 0.82), excess kurtosis (p = 0.21), skew (p = 0.56), the minimum and maximum, and value at risk and expected shortfall at both levels. Its 99% expected shortfall is −5.17% against the observed −5.07%. Figure 4 shows replicate densities sitting on the data through the body and into both tails, and unlike the Student-t its kurtosis is finite and stable.

The skew result is the one I find most satisfying. The replicate skew is centred at −0.54 against the observed −0.39, and the whole band stays negative. The model is not just heavy-tailed in a symmetric way.

![Figure 4. Posterior predictive density for the NIG.](../figures/week5_ppc_nig.png)

*Figure 4. Posterior predictive check for the NIG. Replicate densities track the observed data through the body and into both tails, with the heavier left tail reproduced rather than assumed.*

Figure 5 puts the four risk numbers side by side. For each of value at risk and expected shortfall at 95% and 99% it draws each model's replicate band against the observed value as a dashed line. The Gaussian band sits clear of the dashed line in the 99% panels, the Laplace closes about half the distance, and the Student-t and NIG bands straddle it. The picture is the Week 2 shortfall gap drawn in posterior-predictive form, with the NIG the only model whose band covers the truth on all four.

![Figure 5. Posterior predictive tail risk for all four models.](../figures/week5_ppc_risk.png)

*Figure 5. Posterior predictive value at risk and expected shortfall at 95% and 99%. Each marker is a model's replicate median with its 94% band; the dashed line is the observed value. The Gaussian misses badly at 99%; the NIG covers all four.*

**Table 1. Posterior predictive p-values by model. Values near 0.5 indicate the model reproduces that feature; values flagged with † sit beyond 0.05 or 0.95, where the observed data lies outside almost all replicates.**

| Statistic | Gaussian | Laplace | Student-t | NIG |
|-----------|----------|---------|-----------|-----|
| Std deviation | 0.99 † | 0.00 † | 0.05 | 0.82 |
| Excess kurtosis | 0.00 † | 0.00 † | 0.00 † | 0.21 |
| Skew | 0.00 † | 0.00 † | 0.81 | 0.56 |
| Minimum | 0.00 † | 0.00 † | 0.22 | 0.36 |
| Maximum | 0.00 † | 0.01 † | 0.05 | 0.28 |
| VaR 95% | 0.01 † | 0.13 | 0.00 † | 0.93 |
| ES 95% | 0.00 † | 0.00 † | 0.92 | 0.55 |
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

![Figure 6. Trace diagnostics for all four models.](../figures/week5_diagnostics_nig.png)

*Figure 6. Per-chain traces for the NIG.*

---

## 7. Where this leaves the project

The NIG reproduces every feature of the data including the tails and the skew. The Student-t gets the tail quantiles right but has moments too heavy to pin down. The Laplace closes part of the Gaussian's gap but stops at exponential tails. The Gaussian fails the moment the test moves past the centre of the distribution. This confirms our Week 2 shortfall but in a different manner.

Next I will initiate my rolling backtest in Weeks 6 and 7, the natural next step from here.
