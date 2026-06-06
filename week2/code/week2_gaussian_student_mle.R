# =============================================================================
# Week 2 — Gaussian & Student-t MLE  (R implementation)
# Project: Beyond Black-Scholes: Fitting Lévy Processes to Stock Returns
#
# Objectives:
#   1. Download S&P 500 daily log-returns (2000–2024) via quantmod
#   2. Fit Normal(μ, σ²) and Student-t(ν, μ, σ) by Maximum Likelihood
#   3. Compute standard errors from the Fisher Information matrix
#   4. Estimate VaR and ES at 95% and 99% confidence levels
#   5. Produce density overlay and QQ diagnostic plots (ggplot2)
#
# Install required packages (run once):
#   install.packages(c("quantmod", "MASS", "ggplot2", "gridExtra"))
# =============================================================================

library(quantmod)   # financial data download via getSymbols()
library(MASS)       # fitdistr() — general-purpose MLE wrapper
library(ggplot2)    # publication-quality plots
library(gridExtra)  # arrangeGrob() for multi-panel layout


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

fetch_returns <- function(ticker = "^GSPC",
                          from   = "2000-01-01",
                          to     = "2024-12-31") {
  # getSymbols() stores the downloaded object in the global environment.
  # Ad() extracts the split/dividend-adjusted closing price series.
  suppressMessages(
    getSymbols(ticker, from = from, to = to,
               auto.assign = TRUE, warnings = FALSE)
  )
  prices  <- Ad(GSPC)
  returns <- na.omit(diff(log(prices)))   # r_t = log(S_t / S_{t-1})
  as.numeric(returns)
}

r <- fetch_returns()
cat(sprintf("Data: %d daily log-returns  (S&P 500, 2000-2024)\n\n", length(r)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN MLE
# ─────────────────────────────────────────────────────────────────────────────
# For the Normal distribution, MLE = sample mean and biased sample std.
# MASS::fitdistr() wraps optim() and returns estimates + standard errors
# derived from the observed Fisher Information (inverse Hessian of neg-LL).

fit_gaussian <- function(r) {
  fit <- fitdistr(r, "normal")
  list(
    mu       = unname(fit$estimate["mean"]),
    sigma    = unname(fit$estimate["sd"]),
    se_mu    = unname(fit$sd["mean"]),
    se_sigma = unname(fit$sd["sd"])
  )
}

g <- fit_gaussian(r)
cat("Gaussian MLE\n")
cat(sprintf("  mu    = %+.6f   SE = %.6f\n", g$mu,    g$se_mu))
cat(sprintf("  sigma = %.6f    SE = %.6f\n", g$sigma, g$se_sigma))


# ─────────────────────────────────────────────────────────────────────────────
# 3. STUDENT-t MLE
# ─────────────────────────────────────────────────────────────────────────────
# MASS::fitdistr(r, "t") fits a 3-parameter location-scale t(df, m, s).
# Internally uses optim() with hessian = TRUE, so standard errors are
# sqrt(diag(solve(H))) where H is the Hessian of the negative log-likelihood.
#
# The log-likelihood of one observation from t(df, m, s) is:
#   log f(x) = log dt((x - m)/s, df) - log(s)
# (dividing by s is the Jacobian of the location-scale transformation)

fit_student_t <- function(r) {
  fit <- fitdistr(r, "t")
  list(
    nu       = unname(fit$estimate["df"]),
    mu       = unname(fit$estimate["m"]),
    sigma    = unname(fit$estimate["s"]),
    se_nu    = unname(fit$sd["df"]),
    se_mu    = unname(fit$sd["m"]),
    se_sigma = unname(fit$sd["s"])
  )
}

cat("\nFitting Student-t (numerical optimisation)...\n")
t_fit <- fit_student_t(r)
cat("Student-t MLE\n")
cat(sprintf("  nu    = %.4f      SE = %.4f\n", t_fit$nu,    t_fit$se_nu))
cat(sprintf("  mu    = %+.6f   SE = %.6f\n",  t_fit$mu,    t_fit$se_mu))
cat(sprintf("  sigma = %.6f    SE = %.6f\n",  t_fit$sigma, t_fit$se_sigma))


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALUE-AT-RISK AND EXPECTED SHORTFALL
# ─────────────────────────────────────────────────────────────────────────────
#
# Convention: alpha = confidence level (e.g. 0.95); p = 1 - alpha (tail prob).
# All measures are expressed as daily log-returns; negative = loss.
#
# VaR_alpha  = mu + sigma * F^{-1}(p)           [quantile of fitted model]
#
# Gaussian ES  (conditional expectation of truncated Normal):
#   ES_alpha  = mu - sigma * phi(qnorm(p)) / p
#   where phi is the standard Normal PDF.
#
# Student-t ES  (standard closed-form result):
#   ES_alpha  = mu + sigma * [-f_nu(q_p) * (nu + q_p^2) / (nu - 1)] / p
#   where q_p = qt(p, df = nu) and f_nu is the standard t PDF.
#
# Both ES expressions equal E[r | r <= VaR_alpha].
# ─────────────────────────────────────────────────────────────────────────────

risk_measures <- function(g, t, alphas = c(0.95, 0.99)) {
  results <- data.frame()
  for (a in alphas) {
    p <- 1 - a

    # Gaussian
    q_g   <- qnorm(p)
    var_g <- g$mu + g$sigma * q_g
    es_g  <- g$mu - g$sigma * dnorm(q_g) / p

    # Student-t
    nu  <- t$nu;  mu  <- t$mu;  sig <- t$sigma
    q_t   <- qt(p, df = nu)
    var_t <- mu + sig * q_t
    es_t  <- mu + sig * (-dt(q_t, df = nu) * (nu + q_t^2) / (nu - 1)) / p

    results <- rbind(results, data.frame(
      Confidence   = paste0(as.integer(a * 100), "%"),
      VaR_Gaussian = var_g,
      ES_Gaussian  = es_g,
      VaR_StudentT = var_t,
      ES_StudentT  = es_t
    ))
  }
  results
}

risk <- risk_measures(g, t_fit)
cat("\nRisk Measures (daily log-returns; negative = loss):\n")
print(risk, row.names = FALSE, digits = 6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

make_plots <- function(r, g, t,
                       save_path = "week2_plots_R.png") {

  df <- data.frame(returns = r)
  x  <- seq(min(r), max(r), length.out = 800)

  # Build a data frame of fitted PDF values for both models
  t_label  <- sprintf("Student-t MLE (nu=%.2f)", t$nu)
  lines_df <- rbind(
    data.frame(x = x, y = dnorm(x, g$mu, g$sigma),              model = "Gaussian MLE"),
    data.frame(x = x, y = dt((x - t$mu) / t$sigma, df = t$nu) / t$sigma, model = t_label)
  )
  model_colours <- setNames(c("steelblue", "crimson"), c("Gaussian MLE", t_label))

  # ── Panel 1: density overlay ───────────────────────────────────────────────
  p1 <- ggplot(df, aes(x = returns)) +
    geom_histogram(aes(y = after_stat(density)),
                   bins = 150, fill = "#cfe2f3", colour = NA) +
    geom_line(data  = lines_df,
              aes(x = x, y = y, colour = model), linewidth = 0.85) +
    scale_colour_manual(values = model_colours) +
    coord_cartesian(xlim = c(-0.12, 0.12)) +
    labs(
      title  = "S&P 500 Daily Log-Returns - Gaussian vs Student-t MLE (2000-2024)",
      x      = "Daily log-return",
      y      = "Density",
      colour = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "top")

  # ── Panel 2: Normal QQ ────────────────────────────────────────────────────
  # Standardise returns using Gaussian MLE parameters then compare to N(0,1).
  # S-shaped deviations at the tails confirm fat tails in the data.
  std_g  <- (r - g$mu) / g$sigma
  n      <- length(std_g)
  probs  <- ppoints(n)
  qq_g_df <- data.frame(
    theoretical = qnorm(probs),
    sample      = sort(std_g)
  )
  lim_g <- range(c(qq_g_df$theoretical, qq_g_df$sample))
  p2 <- ggplot(qq_g_df, aes(theoretical, sample)) +
    geom_point(size = 0.4, alpha = 0.5, colour = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.5) +
    coord_fixed(xlim = lim_g, ylim = lim_g) +
    labs(title = "QQ Plot - Gaussian",
         x = "Theoretical quantiles", y = "Sample quantiles") +
    theme_minimal(base_size = 11)

  # ── Panel 3: Student-t QQ ─────────────────────────────────────────────────
  # Standardise using Student-t MLE parameters and compare to t(nu).
  # Tighter alignment here confirms the t-distribution captures fat tails.
  std_t  <- (r - t$mu) / t$sigma
  qq_t_df <- data.frame(
    theoretical = qt(probs, df = t$nu),
    sample      = sort(std_t)
  )
  lim_t <- range(c(qq_t_df$theoretical, qq_t_df$sample))
  p3 <- ggplot(qq_t_df, aes(theoretical, sample)) +
    geom_point(size = 0.4, alpha = 0.5, colour = "crimson") +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.5) +
    coord_fixed(xlim = lim_t, ylim = lim_t) +
    labs(title = sprintf("QQ Plot - Student-t (nu=%.2f)", t$nu),
         x = "Theoretical quantiles", y = "Sample quantiles") +
    theme_minimal(base_size = 11)

  # ── Combine and save ───────────────────────────────────────────────────────
  combined <- gridExtra::arrangeGrob(p1, p2, p3,
                                     layout_matrix = rbind(c(1, 1), c(2, 3)))
  ggsave(save_path, combined, width = 14, height = 10, dpi = 150)
  cat(sprintf("\nPlot saved -> %s\n", save_path))
}

# Run plots (save in same directory as script)
script_dir <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                       error = function(e) getwd())
make_plots(r, g, t_fit,
           save_path = file.path(script_dir, "week2_plots_R.png"))

cat("\nDone.\n")
