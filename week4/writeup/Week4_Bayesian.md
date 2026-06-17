# Week 4: Bayesian estimation with NUTS

## 1. Why refit everything the Bayesian way

Weeks 2 and 3 fitted every model by maximum likelihood. That gives one best value for each parameter and a standard error read off the curvature of the likelihood. The standard error leans on an assumption: that the sampling distribution of the estimate is roughly normal. For most of these parameters that is harmless. For the Student-t degrees of freedom it is worth a second look, because the MLE lands at ν ≈ 2.65, not far above ν = 2, the point where the variance of the distribution blows up to infinity. Close to that boundary the likelihood stops being symmetric, and a normal error bar can quietly mislead.

So this week I refitted four models in a Bayesian setting and drew their full posteriors with the No-U-Turn Sampler (NUTS) in PyMC. I am not trying to overturn the Week 3 numbers. I want the whole shape of the uncertainty rather than a centre and a width, and above all I want to know whether the posterior for ν stays on the finite-variance side of 2 or whether it leaks below.

The four models are the Gaussian, the Laplace, the Student-t and the NIG. The Laplace stands in for the Variance-Gamma family here. It is the symmetric VG with θ = 0 and ν = 1, and unlike the full VG it has a closed-form density, so it carries the heavy-tail idea without dragging in any special functions. The NIG keeps its full four-parameter density.

---

## 2. Choosing the priors

A Bayesian fit needs a prior on each parameter, and the prior is a modelling choice I have to defend. I wanted priors that are weakly informative. Loose enough that 6,287 daily returns do the talking, but tight enough to rule out values that make no financial sense. A model that thinks a fifty percent daily move is ordinary is not a model worth sampling. I started each prior from the Week 3 MLE and then widened it well beyond the plausible range.

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

A prior can read sensibly in a table and still imply nonsense once it runs through the model. The honest way to find out is to simulate. Before any sampling I drew returns straight from each prior and looked at what came out. This prior predictive check is how I actually decided the priors were acceptable, not the table above.

Figures 1 to 4 show the four. The shared feature is the one I was after: almost all the simulated mass sits inside the worst move the market actually produced, marked by the red lines at ±12.8%, with only thin tails past it. The differences between them matter too. The Gaussian prior is the tightest, its 99.9% range inside ±9.7%. The Student-t and Laplace reach further, to roughly ±12.4% and ±16.7%, because their priors permit heavier tails. The NIG is the only lopsided one, its left tail stretching to about −18.6% against +16.0% on the right, which is the asymmetry its β parameter exists to capture. So the priors are not just harmless. They already carry the tail character each model is meant to bring, before the data has said a word.

![Figure 1. Prior predictive returns for the Gaussian.](../figures/week4_prior_predictive_gaussian.png)

*Figure 1. Prior predictive returns for the Gaussian. The grey histogram is daily returns drawn from the prior alone, before the data is seen; the red lines mark the largest move in the actual sample (±12.8%). This is the tightest of the four.*

![Figure 2. Prior predictive returns for the Laplace.](../figures/week4_prior_predictive_laplace.png)

*Figure 2. Prior predictive returns for the Laplace. The symmetric tails reach further than the Gaussian, out to about ±16.7% at the 99.9% level.*

![Figure 3. Prior predictive returns for the Student-t.](../figures/week4_prior_predictive_student_t.png)

*Figure 3. Prior predictive returns for the Student-t. Tails sit between the Gaussian and the Laplace, reaching about ±12.4%.*

![Figure 4. Prior predictive returns for the NIG.](../figures/week4_prior_predictive_nig.png)

*Figure 4. Prior predictive returns for the NIG. The one asymmetric prior: its left tail stretches further than its right, the skew the β parameter allows.*

The numbers back up the eye. The fraction of simulated days beyond ±25%, a move bigger than anything in the sample, was 0.0001% for the Gaussian, 0.02% for the Student-t, 0.03% for the Laplace and 0.08% for the NIG. All four are negligible. Had any come back fat with absurd moves, I would have tightened the scale or tail prior and tried again. None did, so I sampled.

---

## 4. Sampling and convergence

I ran four chains for each model, with 1,000 tuning steps and 1,000 kept draws per chain. NUTS works by following the gradient of the log-density, which is straightforward for the Gaussian, Laplace and Student-t. The NIG was the real task. Its density contains a modified Bessel function, K₁, and the sampler needs the derivative of that. PyTensor supplies it, so I could code the NIG density directly instead of introducing the latent time-change variable the distribution is built from. I checked my version of the density against the one in scipy and they matched to machine precision, which gave me confidence the gradient it was differentiating was the right one.

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

The posteriors come out almost exactly where the MLE did. With this much data and priors this loose, that is what should happen, and seeing it is reassuring rather than dull: it tells me the sampler and the MLE agree. The payoff is not a new centre. It is the shape around it.

**Table 2. Posterior summaries against the Week 3 MLE. HDI is the 94% highest-density interval.**

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

## 6. Where this leaves the project

The Bayesian pass did not rewrite Week 3. It backed it up and tightened the one claim that needed tightening. The Student-t tail is heavy but its variance is finite, and that statement now rests on a full posterior instead of a single number with an asymptotic error bar. The NIG left skew survives the same scrutiny.

The honest caveats are about the priors. They are weakly informative, not absent, and with a shorter sample they would carry more weight than they do here. The Laplace also only captures the symmetric half of the VG idea, so it speaks to how thick the tails are, not to skew. Next week I will run posterior predictive checks, pushing each fitted posterior back through the data to see where the models still miss, which is the natural test of whether these fits are good for anything beyond describing the sample they were trained on.
