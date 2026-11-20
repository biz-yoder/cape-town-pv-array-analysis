
# ============================================================
# HonestDiD Sensitivity Analysis: Rooftop PV Adoption
# Cape Town, South Africa
#
# Estimates sensitivity of grid electricity consumption
# effects to violations of parallel trends, by PV capacity.
#
# Requires: HonestDiD (original analysis: version 0.2.8)
# ============================================================

library(HonestDiD)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

input_dir <- "honestdid_inputs_max5"

capacity_labels <- c(
  "<1 kW",
  "1–2 kW",
  "2–3 kW",
  "3–5 kW",
  ">5 kW"
)

Mbarvec <- c(0, 0.25, 0.5, 0.75, 1, 1.5, 2)

# Event-study window:
# Pre-treatment: -12 through -2
# Reference period: -1 (omitted)
# Post-treatment: 0 through +36

num_pre <- 11
num_post <- 37

# Average treatment effect during months +6 through +12.
# Post-treatment coefficient 1 corresponds to month 0.

l_vec <- rep(0, num_post)
l_vec[7:13] <- 1 / 7

# Grid settings consistent with the original saved results.
# NOTE: Confidence intervals reaching these boundaries may
# be truncated by the search grid.

grid_lower <- -2
grid_upper <- 1
grid_points <- 3001

# ------------------------------------------------------------
# 2. Run HonestDiD for each PV capacity group
# ------------------------------------------------------------

run_honestdid <- function(group_id) {
  
  coef_file <- file.path(
    input_dir,
    sprintf("group_%d_betahat.csv", group_id)
  )
  
  cov_file <- file.path(
    input_dir,
    sprintf("group_%d_covariance.csv", group_id)
  )
  
  coef_data <- read.csv(coef_file)
  cov_data <- read.csv(cov_file, check.names = FALSE)
  
  betahat <- coef_data$att
  sigma <- as.matrix(cov_data[, -1, drop = FALSE])
  storage.mode(sigma) <- "double"
  
  # Verify that the coefficients and covariance matrix align.
  stopifnot(
    identical(coef_data$event_time, cov_data$event_time),
    length(betahat) == num_pre + num_post,
    nrow(sigma) == length(betahat),
    ncol(sigma) == length(betahat),
    identical(
      as.integer(coef_data$event_time),
      c(-12:-2, 0:36)
    )
  )
  
  # Estimated average ATT for months +6 through +12.
  post_betahat <- betahat[(num_pre + 1):length(betahat)]
  average_att <- sum(l_vec * post_betahat)
  
  # Rambachan and Roth sensitivity analysis.
  sensitivity <- HonestDiD::createSensitivityResults_relativeMagnitudes(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = num_pre,
    numPostPeriods = num_post,
    method = "C-LF",
    Mbarvec = Mbarvec,
    l_vec = l_vec,
    alpha = 0.05,
    grid.lb = grid_lower,
    grid.ub = grid_upper,
    gridPoints = grid_points,
    seed = 0
  )
  
  sensitivity$capacity_group <- capacity_labels[group_id + 1]
  sensitivity$average_att <- average_att
  
  return(sensitivity)
}

# ------------------------------------------------------------
# 3. Combine results across five capacity groups
# ------------------------------------------------------------

results <- do.call(
  rbind,
  lapply(0:4, run_honestdid)
)

results <- as.data.frame(results)

results <- results[, c(
  "capacity_group",
  "average_att",
  "Mbar",
  "lb",
  "ub",
  "method",
  "Delta"
)]

print(results, row.names = FALSE)

# ------------------------------------------------------------
# 4. Export sensitivity results
# ------------------------------------------------------------

output_file <- "honestdid_results_reproduced.csv"

write.csv(
  results,
  output_file,
  row.names = FALSE
)

cat("\nSaved results to:", output_file, "\n")
cat("HonestDiD version:",
    as.character(packageVersion("HonestDiD")), "\n")
