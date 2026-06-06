# Week 2 Results: Gaussian and Student-t MLE Benchmarks

## 1. Data

The analysis uses 6,287 daily log-returns on the S&P 500 index, computed as
r_t = log(S_t / S_{t-1}), covering the period 2 January 2000 to 31 December 2024.
This 25-year window spans three major stress episodes — the dot-com collapse
(2000–2002), the Global Financial Crisis (2007–2009), and the COVID-19 crash
(February–March 2020) — providing a demanding sample for tail-risk estimation.


## 2. Parameter Estimates

### 2.1 Gaussian MLE

For the Normal distribution N(μ, σ²), MLE reduces to the sample mean and
biased sample standard deviation. Standard errors follow from the Fisher
information for the Normal family.

| Parameter | Estimate  | Standard Error | t-statistic |
|-----------|-----------|----------------|-------------|
| μ         | +0.000223 | 0.000154       | 1.45        |
| σ         | 0.012234  | 0.000109       | —           |

The estimated daily drift μ = 0.022% is statistically indistinguishable from
zero at the 5% level (|t| = 1.45 < 1.96), consistent with the efficient
market hypothesis at the daily frequency. The estimated scale σ = 1.22% per
day corresponds to an annualised volatility of approximately 19.4%
(σ × √252), in line with long-run S&P 500 volatility estimates in the
literature.

### 2.2 Student-t MLE

The location-scale Student-t distribution t(ν, μ, σ) was fitted by numerical
maximum likelihood using the L-BFGS-B algorithm, with standard errors
recovered from the observed Fisher Information matrix (inverse numerical
Hessian of the negative log-likelihood at the optimum).

| Parameter | Estimate  | Standard Error | t-statistic |
|-----------|-----------|----------------|-------------|
| ν         | 2.6483    | 0.1102         | —           |
| μ         | +0.000656 | 0.000110       | 5.96        |
| σ (scale) | 0.007077  | 0.000128       | —           |

The degrees-of-freedom estimate ν̂ = 2.65 (95% CI: approximately 2.43–2.87)
is the central quantitative finding of Week 2. Several implications follow
directly.

**Implied volatility.** The scale parameter σ in the location-scale
Student-t is not the standard deviation; the actual standard deviation is
σ√(ν/(ν−2)) = 0.007077 × √(2.6483/0.6483) ≈ 1.43% per day, meaningfully
higher than the Gaussian estimate of 1.22%. The Student-t allocates the
"extra" spread into the tails rather than the body of the distribution,
producing both a sharper central peak and heavier extremes — the twin
features of leptokurtosis visible in the density overlay (Figure 1, top
panel).

**Moment structure.** The estimated ν = 2.65 implies a distribution whose
variance is finite (requires ν > 2), but whose skewness and kurtosis are
formally undefined (require ν > 3 and ν > 4 respectively). The fitted model
is therefore consistent with a distribution that has infinite kurtosis — a
natural consequence of the three major crash events in the sample pushing
the MLE toward the lower boundary of the parameter space. This motivates the
use of the Variance-Gamma (VG) and Normal Inverse Gaussian (NIG) models in
Weeks 3–4, which can accommodate heavy tails while preserving finite moments
at all orders.

**Precision of ν.** The standard error SE(ν̂) = 0.110 implies a
coefficient of variation of roughly 4.2%, indicating a well-identified
estimate. The lower confidence bound (~2.43) remains above the ν = 2
singularity at which the distribution's variance diverges.


## 3. Model Diagnostics

### 3.1 Density Overlay (Figure 1, top panel)

The Gaussian curve is too flat at the centre and assigns excess probability
to the moderate-loss region (approximately ±1% to ±4%), spreading density
into regions where the data is sparse. The Student-t, by contrast, achieves
a taller and sharper central peak consistent with the high frequency of near-
zero returns, while simultaneously retaining mass in the extreme tails.

### 3.2 QQ Plots (Figure 1, bottom panels)

**Gaussian QQ.** The characteristic S-shaped departure from the diagonal
confirms fat tails in both directions. At the left extreme, the empirical
data reaches approximately −10 standardised units, against a Gaussian
theoretical prediction of approximately −3.5 — the worst daily crashes are
roughly three times more severe than the Gaussian model predicts. The
analogous upward deviation in the right tail is consistent with the
occurrence of large positive daily moves following extreme stress (e.g.
circuit-breaker rebounds). This S-shape is the primary visual evidence for
the project's central hypothesis: the Gaussian model systematically
underestimates tail risk.

**Student-t QQ.** The alignment with the diagonal is substantially improved
across the central 99% of the distribution. The residual departures are
confined to the three or four most extreme observations at each tail — events
that are plausibly structural breaks (9/11, Lehman Brothers, COVID) rather
than draws from any single stationary distribution. The improved fit confirms
that ν̂ = 2.65 captures the dominant tail behaviour of the data, while the
residual outliers motivate the richer Lévy process models in Weeks 3–4.


## 4. Risk Measures

Table 1 reports Value-at-Risk (VaR) and Expected Shortfall (ES) under each
model at the 95% and 99% confidence levels. All figures are expressed as
daily log-returns; negative values denote losses.

**Table 1: VaR and Expected Shortfall — Gaussian vs Student-t**

| Confidence | VaR (Gaussian) | ES (Gaussian) | VaR (Student-t) | ES (Student-t) |
|------------|----------------|---------------|-----------------|----------------|
| 95%        | −1.990%        | −2.501%       | −1.694%         | −3.000%        |
| 99%        | −2.824%        | −3.238%       | −3.515%         | −5.813%        |

### 4.1 The 95% Crossover

At the 95% level, the Gaussian VaR (−1.99%) is more conservative than the
Student-t VaR (−1.69%). This counterintuitive reversal reflects the smaller
scale parameter of the fitted t-distribution (σ̂ = 0.71% vs σ̂ = 1.22% for
the Gaussian): at moderate quantile depths, the narrower scale of the t-fit
dominates its heavier tails, producing a smaller loss estimate. The crossover
in VaR — from Gaussian-more-conservative to Student-t-more-conservative —
occurs somewhere between the 95th and 99th percentiles.

However, even at 95%, the Expected Shortfall tells the opposite story: the
Student-t ES of −3.00% is 19.9% more severe than the Gaussian ES of −2.50%.
ES measures the average loss beyond VaR, so once the tail is entered the
heavier-tailed model immediately dominates. This divergence between VaR and
ES rankings is itself an important finding: risk managers who rely on VaR
alone may observe that the Gaussian appears more conservative, and
incorrectly conclude the Gaussian is the safer model.

### 4.2 The 99% Level — Core Finding

At the 99% level the Student-t is more conservative on both measures.
The Student-t VaR of −3.52% is 24.5% larger in magnitude than the Gaussian
VaR of −2.82%. The gap is most dramatic for Expected Shortfall: the Student-t
ES of −5.81% is 79.5% larger in magnitude than the Gaussian ES of −3.24%.

In practical terms, on the worst 1% of trading days, the Gaussian model
predicts an average loss of approximately 3.24%, while the Student-t
predicts 5.81% — a gap of 2.57 percentage points that compounds materially
at a portfolio level. A portfolio manager relying on Gaussian ES to set
capital reserves would hold approximately 44% less capital than a model
calibrated to the empirically observed tail structure would require
(3.238 / 5.813 ≈ 0.557, implying a 44.3% shortfall).


## 5. Summary and Implications for Weeks 3–4

Week 2 establishes the two benchmark models and produces the following
hierarchy of findings, ordered by significance to the project thesis:

1. **ν̂ = 2.65 confirms extreme fat tails.** The S&P 500 daily return
   distribution has tail behaviour consistent with an infinite-kurtosis
   process over the 2000–2024 period. This is direct evidence against the
   Gaussian (Black-Scholes) assumption.

2. **The Gaussian dramatically underestimates 99% tail risk.** The 79.5%
   gap in 99% ES between the Student-t and Gaussian benchmarks quantifies
   the cost of the distributional misspecification. This figure sets the
   scale against which VG and NIG improvements will be assessed.

3. **VaR and ES can give conflicting model rankings.** At 95%, the Gaussian
   VaR appears more conservative while the Student-t ES is larger. This
   reinforces the Basel III and FRTB regulatory shift from VaR to ES as the
   primary risk metric.

4. **Residual tail outliers motivate richer models.** The Student-t QQ plot
   shows good fit for the central 99% of observations but retains departures
   at the extreme few data points. Weeks 3–4 will test whether the
   Variance-Gamma and Normal Inverse Gaussian distributions — which have
   independent parameters controlling skewness and kurtosis — reduce these
   residual departures and improve the AIC/BIC model selection criteria.
