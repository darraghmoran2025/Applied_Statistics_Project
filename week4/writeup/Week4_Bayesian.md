# Week 4: Bayesian estimation with NUTS

## 1. Bayesian Refitting

Weeks 2 and 3 fitted every model by maximum likelihood. That gives one best value for each parameter and a standard error read off the curvature of the likelihood. The standard error leans on an assumption: that the sampling distribution of the estimate is roughly normal. For the Student-t degrees of freedom it is worth a second look, because the MLE lands at ν ≈ 2.65, not far above ν = 2, the point where the variance of the distribution blows up to infinity. Close to that boundary the likelihood stops being symmetric.

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

![Figure 1. Prior predictive returns for the Gaussian.](../figures/week4_prior_predictive_gaussian.png)

*Figure 1. Prior predictive returns for the Gaussian. The grey histogram is daily returns drawn from the prior alone, before the data is seen; the red lines mark the largest move in the actual sample (±12.8%). This is the tightest of the four.*

![Figure 2. Prior predictive returns for the Laplace.](../figures/week4_prior_predictive_laplace.png)

*Figure 2. Prior predictive returns for the Laplace. The symmetric tails reach further than the Gaussian, out to about ±16.7% at the 99.9% level.*

![Figure 3. Prior predictive returns for the Student-t.](../figures/week4_prior_predictive_student_t.png)

*Figure 3. Prior predictive returns for the Student-t. Tails sit between the Gaussian and the Laplace, reaching about ±12.4%.*

![Figure 4. Prior predictive returns for the NIG.](../figures/week4_prior_predictive_nig.png)

*Figure 4. Prior predictive returns for the NIG. The one asymmetric prior: its left tail stretches further than its right, the skew the β parameter allows.*

The fraction of simulated days beyond ±25%, a move bigger than anything in the sample, was 0.0001% for the Gaussian, 0.02% for the Student-t, 0.03% for the Laplace and 0.08% for the NIG. All four are negligible. Had any come back fat with absurd moves, I would have tightened the scale or tail prior and tried again. None did, so I sampled.

---

## 4. Sampling and convergence

I ran four chains for each model, with 1,000 tuning steps and 1,000 kept draws per chain. NUTS works by following the gradient of the log-density, which is straightforward for the Gaussian, Laplace and Student-t. The NIG was the real task. Its density contains a modified Bessel function, K₁, and the sampler needs the derivative of that. PyTensor supplies it, so I could code the NIG density directly instead of introducing the latent time-change variable the distribution is built from. I checked my version of the density against the one in scipy and they matched, which gave me confidence the gradient it was differentiating was the right one.

Every model converged cleanly. R-hat sat at 1.00 across the board, and the effective sample size never fell below about 2,000. Figures 5 to 8 show the traces. In each one the chains sit on top of each other on the left, and the raw draws on the right form a flat, stationary band with no drift and no chain wandering off, which is what good mixing looks like. The Gaussian and Laplace settle fastest, as their two-parameter posteriors should. The Student-t panel is the one I watch most closely, because it shows the ν chains circling steadily around 2.66 rather than creeping toward the ν = 2 boundary. The NIG, with four parameters and a Bessel function in its density, mixes just as well as the rest.

![Figure 5. NUTS trace for the Gaussian.](../figures/week4_trace_gaussian.png)

*Figure 5. NUTS diagnostics for the Gaussian. Left: the posterior for each parameter, one line per chain, with the Week 3 MLE dashed. Right: the raw chains.*

![Figure 6. NUTS trace for the Laplace.](../figures/week4_trace_laplace.png)

*Figure 6. NUTS diagnostics for the Laplace. Two parameters, fast and clean convergence.*

![Figure 7. NUTS trace for the Student-t.](../figures/week4_trace_student_t.png)

*Figure 7. NUTS diagnostics for the Student-t. The ν chains (bottom) circle around 2.66 and stay clear of the ν = 2 boundary.*

![Figure 8. NUTS trace for the NIG, the hardest of the four.](../figures/week4_trace_nig.png)

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

Everything so far, here and in Weeks 2 and 3, fits one set of parameters to the whole 2000–2024 sample. A single fit cannot say whether the heavy tails are a permanent feature of the market or something that switches on in crises and fades in calm years. To see that, I refitted the Variance-Gamma and NIG densities by maximum likelihood one calendar year at a time, twenty-five fits each, and read the parameters as a time series.

Two choices shape these fits. First, the location is fixed at μ = 0. Within a single year there are only about 250 returns, the daily drift is tiny and barely identified, and leaving it free just adds a noisy nuisance parameter; pinning it to zero lets the scale, tail and asymmetry parameters carry the year's shape, and whatever small mean exists is absorbed by the asymmetry parameter (θ for VG, β for NIG), which is where it belongs. Second, the daily scale parameters σ and δ are reported annualised (× √252), so a year's scale reads on the same footing as the VIX and the Gaussian σ. The densities, the α > |β| reparametrisation and the standard errors are reused from the Week 3 code, so these yearly fits are numerically of a piece with the full-sample ones.

![Figure 9. Year-by-year Variance-Gamma parameters (μ = 0): annualised scale σ, asymmetry θ, and variance rate ν, with shock windows shaded and ±1 SE bars.](../figures/week4_yearly_vg_params.png)

*Figure 9. Year-by-year Variance-Gamma fit (μ = 0). Top to bottom: the annualised scale σ, the asymmetry θ, and the variance rate ν (tail heaviness). Shaded bands are the four shock windows; bars are ±1 standard error.*

The VG scale moves with the crises. σ peaks in 2008 at 39.8% annualised and again in 2020 at 31.3%, the GFC and COVID years, and bottoms out in the calm of 2017 at 6.5%. The variance rate ν, which sets the tail heaviness, tends to run higher through the turbulent 2008 to 2020 stretch than in the quiet mid-2000s or the 2021 to 2023 years, though it is a good deal noisier than the scale.

![Figure 10. Year-by-year NIG parameters (μ = 0): tail α and scale δ on log axes, asymmetry β with ±1 SE bars; shock windows shaded.](../figures/week4_yearly_nig_params.png)

*Figure 10. Year-by-year Normal-Inverse-Gaussian fit (μ = 0). Tail-heaviness α and annualised scale δ are on log axes; asymmetry β is linear with ±1 SE bars. In the very calm years 2004, 2005 and 2023 the NIG runs into its Gaussian limit, where α and δ both grow large and are only weakly identified individually (the SEs there are enormous, so the bars are omitted on the log panels); their product stays finite and the tails are simply close to normal.*

The NIG says the same thing through its tail. Its tail-heaviness α (smaller means heavier tails) falls to its lowest values, around 22, in exactly 2008 and 2020, so the heaviest tails of the whole sample land on the two largest crises. In the calmest years the NIG drifts toward the Gaussian: α runs into the thousands and the scale destabilises, which is the signature of that limit rather than a failed fit. The asymmetry β is most negative, most left-skewed, in 2001 to 2002, in 2008 and in 2022, the dot-com unwind, the GFC and the rate-hike drawdown. That matches the full-sample posterior in Section 5, which put β firmly below zero.

The two models agree. Tail heaviness and scale are not fixed features of the market; they rise sharply in crises and settle back in calm years, and the negative skew bunches into the stressed periods. The single full-sample fit is then best read as an average over a distribution that changes through time, and the heavy-tailed, left-skewed picture it gives comes mostly from the crisis years rather than evenly from the whole sample.

Figures 9 and 10 track the parameters; Figures 11 to 15 show the fitted densities they produce, one panel per year against that year's empirical distribution, in the same five-year-block layout as the Week 2 marginals. Each panel shades its background with the shock window the year belongs to, overlays the fitted VG (green) and NIG (orange dashed) densities on the empirical kernel estimate, and prints the year's annualised scale together with the two tail parameters. The crisis years stand out by eye: 2008 and 2020 have the sharpest peaks and the longest tails, the calm years are visibly closer to normal, and the two Lévy densities sit almost on top of each other throughout, which is why the choice between them turns on tail risk rather than overall shape.

![Figure 11. Per-year fitted VG and NIG densities, 2000–2004.](../figures/week4_marginals_2000_2004.png)

*Figure 11. Per-year empirical distribution (kernel estimate) with fitted VG and NIG densities (μ = 0), 2000–2004. Panel backgrounds mark the shock windows; each panel prints the annualised scale σ, the VG ν, and the NIG α.*

![Figure 12. Per-year fitted VG and NIG densities, 2005–2009.](../figures/week4_marginals_2005_2009.png)

*Figure 12. As Figure 11, for 2005–2009. The GFC years 2008–2009 carry the sharpest peaks and heaviest tails of the block.*

![Figure 13. Per-year fitted VG and NIG densities, 2010–2014.](../figures/week4_marginals_2010_2014.png)

*Figure 13. As Figure 11, for 2010–2014.*

![Figure 14. Per-year fitted VG and NIG densities, 2015–2019.](../figures/week4_marginals_2015_2019.png)

*Figure 14. As Figure 11, for 2015–2019.*

![Figure 15. Per-year fitted VG and NIG densities, 2020–2024.](../figures/week4_marginals_2020_2024.png)

*Figure 15. As Figure 11, for 2020–2024. The COVID year 2020 has the most extreme peak-and-tail shape of the whole sample.*
