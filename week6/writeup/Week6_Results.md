# Week 6 results: calendar structure and parameter regressions

## 1. Overview

This week follows the Week 6 plan: regress each Lévy parameter separately with the NIG δ first, split volatility into market-open and market-closed sessions, contrast the days of the week under two week definitions, measure volatility week on week, and look at quarterly earnings. Everything runs on the same 6,287 daily S&P 500 log-returns as Weeks 2 to 5, plus open prices and the VIX where those are needed.

Four results stand out. Quarterly NIG δ tracks the VIX closely (R² = 0.64 in logs), and at quarterly frequency the skew parameters finally show the VIX relationship the annual regression could not detect. The week has a real shape: Monday against midweek against Friday is exactly the right grouping, and five separate days add nothing beyond it. The market earns four fifths of its variance while open but takes its worst tail risk while closed. And earnings seasons, at the index level, do not move volatility at all.

---

## 2. Day-of-week structure

I fitted the Gaussian and Student-t separately to each weekday. Monday returns span the weekend, Friday close to Monday close, so the weekend gap lives inside Monday.

**Table 1. Per-weekday MLEs, 2000-2024. Mean in basis points per day; σ annualised.**

| Day | n | Mean (bp) | σ (ann.) | Student-t ν | SE(ν) | Excess kurtosis |
|-----|---|-----------|----------|-------------|-------|-----------------|
| Monday | 1,178 | +0.1 | 21.4% | 2.17 | 0.18 | 17.5 |
| Tuesday | 1,289 | +5.5 | 19.4% | 2.88 | 0.27 | 8.7 |
| Wednesday | 1,290 | +1.6 | 18.9% | 2.70 | 0.25 | 6.9 |
| Thursday | 1,268 | +3.4 | 19.6% | 2.61 | 0.24 | 8.3 |
| Friday | 1,262 | +0.3 | 17.8% | 3.10 | 0.35 | 5.3 |

The means are statistically flat (a per-day mean has a standard error near 3.4 bp), which is what an efficient market should give. The second moments are not flat. Monday carries the highest volatility and by far the heaviest tail: ν = 2.17 with a standard error of 0.18, so its interval brushes the ν = 2 variance boundary that the full sample stays clear of. Friday is the calmest and lightest-tailed day. The excess kurtosis column tells the same story, 17.5 on Monday against 5.3 on Friday.

The plan asked me to compare two week definitions, and the likelihood-ratio tests give a clean answer:

**Table 2. Gaussian likelihood-ratio tests between week definitions.**

| Comparison | LR | df | p |
|------------|----|----|---|
| Mon / midweek / Fri vs pooled week | 42.9 | 4 | < 0.0001 |
| Five days vs pooled week | 45.7 | 8 | < 0.0001 |
| Five days vs Mon / midweek / Fri | 2.8 | 4 | 0.594 |

The pooled "Monday (inclusive) to Friday" definition is rejected decisively, but the fully saturated five-day model is no better than the three-group version. Monday, midweek and Friday is the right resolution: Tuesday, Wednesday and Thursday are statistically one day.

![Figure 1. Volatility and tail parameter by weekday.](../figures/week6_weekday_params.png)

*Figure 1. Left: annualised Gaussian volatility by weekday with 95% intervals. Right: Student-t ν by weekday against the full-sample 2.648. Monday is the most volatile and heaviest-tailed day; Friday the calmest.*

---

## 3. Market open vs market closed

The close-to-close return splits exactly into an overnight part (prior close to today's open, the market closed) and an intraday part (open to close, the market open).

One data problem first: Yahoo's ^GSPC open prices are stale in the early sample. The open equals the prior close on 96% of days in 2000-2004, 31% in 2005-2009 and 12% in 2010-2014, then near zero from 2015. The split therefore runs on 2015-2024 (n = 2,515, stale fraction 0.08%); the rest of the project is unaffected because it needs closes only.

**Table 3. Return components, 2015-2024.**

| Component | σ (ann.) | Variance share | Student-t ν | Excess kurtosis | Skew |
|-----------|----------|----------------|-------------|-----------------|------|
| Close-to-close | 17.9% | 100% | 2.68 | 15.7 | −0.81 |
| Overnight (closed) | 8.0% | 20.0% | 2.14 | 36.1 | −1.76 |
| Intraday (open) | 14.0% | 61.0% | 2.79 | 5.2 | −0.41 |

The two shares sum to 81%; the remaining 19% comes from twice the positive covariance (the components correlate at +0.27, so overnight moves tend to carry into the day session).

The split is lopsided in an interesting way. The open market carries three times the closed market's variance, but the closed market carries the tail. Overnight ν is 2.14, right at the variance boundary, with excess kurtosis of 36 and skew of −1.8; intraday returns are comparatively tame at ν = 2.79 and kurtosis 5. Jumps arrive while the market is shut, when news accumulates and no trading can absorb it gradually, and they arrive disproportionately on the downside. This is the Lévy story of the whole project told by the clock: the diffusive part of the return accrues while the market is open, and the jump part accrues while it is closed.

![Figure 2. Overnight vs intraday densities.](../figures/week6_open_close_density.png)

*Figure 2. Overnight and intraday return densities, 2015-2024, linear and log scale. The overnight density is narrower in the body but crosses over in the tails.*

The weekday and session dimensions interact exactly where they should: Monday's overnight volatility is about 10% annualised against 7.3 to 7.8% for the other days, and that gap is the weekend premium identified in Section 2, now located in the session where it actually accrues.

![Figure 3. Volatility by weekday and session.](../figures/week6_open_close_weekday.png)

*Figure 3. Annualised volatility by weekday, split into overnight and intraday. The Monday overnight bar contains the weekend gap.*

---

## 4. Week-on-week volatility

Weekly realised volatility (the root of the summed squared daily returns within each week, annualised) gives the week-on-week measure the plan asked for, on the full 2000-2024 sample of 1,302 weeks.

Volatility persists strongly from week to week: vol_w = 4.30 + 0.72 × vol_(w−1) with R² = 0.52. A calm week is followed by a calm week and a violent one by a violent one, which is the weekly-resolution version of the volatility clustering that the Week 5 posterior predictive check showed no static model can produce.

Weekly returns also confirm aggregational Gaussianity from the Week 1 literature review: the Student-t fitted to weekly returns gives ν = 3.38 (SE 0.35) against the daily 2.648. One step of time aggregation already lightens the tail measurably, though weekly returns remain far from Gaussian.

![Figure 4. Weekly realised volatility, 2000-2024.](../figures/week6_weekly_vol.png)

*Figure 4. Left: weekly realised volatility with the four shock windows shaded. Right: week w against week w−1, slope 0.72, R² 0.52.*

---

## 5. Quarterly parameter regressions

The plan's central item: fit the Lévy models through time and regress each parameter separately. Yearly fits give only 25 points, so I refitted VG(σ, θ, ν) and NIG(α, β, δ) with μ = 0 one calendar quarter at a time, 100 quarters of roughly 63 returns each, reusing the zero-mean machinery from the Week 4 yearly fits.

One identification caveat governs how the results are read. In calm quarters the NIG runs into its Gaussian limit: α and δ grow together along a likelihood ridge where only their ratio (the variance) is determined. Thirty-three of the 100 quarters sit on that ridge (α above 500), and the δ values they report are artifacts, so the δ regressions use the 67 well-identified quarters. That a third of quarters are statistically indistinguishable from Gaussian is itself a finding, and it repeats the sub-period message of Week 3: heavy tails are episodic, not permanent.

**Table 4. One regression per parameter. Newey-West standard errors, 4 lags. The VIX column is from the VIX-only specification; the full specification adds realised volatility, drawdown and the parameter's own lag.**

| Parameter | R² (VIX only) | VIX t-stat | R² (full) | AR(1) slope | AR(1) R² |
|-----------|---------------|------------|-----------|-------------|----------|
| NIG δ (ann., 67 quarters) | 0.50 | +3.0 | 0.60 | +0.40 | 0.16 |
| NIG log α | 0.02 | −1.3 | 0.24 | +0.26 | 0.07 |
| NIG β | 0.17 | −3.2 | 0.28 | −0.01 | 0.00 |
| VG σ (ann.) | 0.82 | +19.9 | 1.00 | +0.52 | 0.27 |
| VG θ | 0.19 | −5.1 | 0.65 | +0.04 | 0.00 |
| VG ν | 0.04 | −2.1 | 0.22 | +0.19 | 0.04 |

The δ result is the one the plan named. In levels, δ against the VIX gives R² = 0.50; in logs, which tames the crisis quarters' leverage, R² = 0.64 with a t-statistic of 9.7. The scale of the NIG is, to a good approximation, a VIX read-out. The VG σ says the same even more strongly (R² = 0.82 at n = 100, matching the 0.90 the annual regression found at n = 25). The full-specification R² of 1.00 for σ is not a triumph, it is a tautology: quarterly realised volatility is nearly the same quantity as the fitted VG scale, and the table keeps it only for completeness.

The new finding relative to Week 3 is the skew. The annual regression found no VIX relationship for either asymmetry parameter (p = 0.60 and 0.16 at n = 25). At quarterly frequency both show up clearly: VG θ has t = −5.1 and NIG β has t = −3.2, higher-VIX quarters are more left-skewed quarters. The annual nulls were a power problem, which the Week 3 limitations section suspected. The tail parameters, in contrast, stay flat against the VIX at any frequency (log α R² = 0.02, VG ν R² = 0.04), so the Week 3 conclusion survives intact and sharper: the VIX prices the scale and, at quarterly resolution, the skew of the return distribution, but not its tail decay.

"Two parameters as a function of themselves": the AR(1) row gives each parameter regressed on its own lag. The scale parameters persist (log δ has an AR slope of 0.58 with R² = 0.34; VG σ 0.52), the skew and tail parameters do not (AR slopes indistinguishable from zero for β and θ). Scale is a slowly moving state; asymmetry and tail weight are episode-specific and reset from quarter to quarter.

![Figure 5. Quarterly NIG parameters through time.](../figures/week6_quarterly_nig.png)

*Figure 5. Quarterly NIG δ (top) and α (bottom), both on log axes with shock windows shaded. Grey crosses mark the 33 weakly identified quarters on the Gaussian-limit ridge, where δ and α are determined only through their ratio. Among the identified quarters, α reaches its sample minimum of about 7 in 2020Q1, the heaviest quarterly tail of the sample.*

![Figure 6. The δ regressions.](../figures/week6_delta_regressions.png)

*Figure 6. Left: quarterly NIG δ against the quarter's average VIX, well-identified quarters only. Right: δ against its own lag, the AR(1) from the plan.*

---

## 6. Earnings seasons

The index has no earnings dates of its own, so this section uses the aggregate reporting calendar as a proxy: each earnings season is the month starting on the 15th of January, April, July and October, when the bulk of S&P 500 constituents report. That covers 34% of trading days. A constituent-level event study would need individual reporting dates and is a different exercise.

**Table 5. In-season vs out-of-season, 2000-2024.**

| Window | n | σ (ann.) | Student-t ν | Excess kurtosis |
|--------|---|----------|-------------|-----------------|
| In season | 2,161 | 19.1% | 2.87 | 7.4 |
| Out of season | 4,126 | 19.6% | 2.55 | 11.8 |

The event-study profile is flat to the decimal: average annualised volatility is 16.5% in the 21 days before a season, 16.4% during it and 16.5% in the 21 days after, with paired t-statistics all below 0.15 in magnitude across 99 seasons.

The null is informative rather than disappointing. Earnings risk is idiosyncratic, and inside a 500-name index the single-name surprises diversify away almost completely. If anything the in-season distribution is slightly lighter-tailed (ν = 2.87 against 2.55), because the macro shocks that actually drive index tails, March 2020, August 2011, the 2008 cascade, mostly landed outside the reporting windows. Index-level tail risk is macro risk, not earnings risk.

![Figure 7. Volatility around earnings seasons.](../figures/week6_earnings_profile.png)

*Figure 7. Mean annualised volatility before, during and after the 99 earnings seasons, with 95% intervals. The profile is flat.*

---

## 7. Limitations

The open/close split rests on ten years of clean opens, so its estimates are less precise than the full-sample ones and describe the post-2015 regime only. The quarterly NIG fits are noisy for the tail and skew parameters at 63 observations per quarter, and a third of quarters sit on the Gaussian-limit ridge; the identified-subset treatment handles δ but means the δ results describe turbulent-to-normal quarters, not calm ones. The regressions are contemporaneous attribution in the sense of Week 3, not forecasts, and the VIX and realised volatility remain strongly correlated, so their coefficients in the full specification are partial associations. The earnings windows are a calendar proxy at the index level; a constituent-level study could still find effects the index averages away.
