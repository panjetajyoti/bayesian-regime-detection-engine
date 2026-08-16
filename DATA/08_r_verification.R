# ==============================================================================
# ZE THETA BAYESIAN REGIME ENGINE - DELIVERABLE 3: R CODEBASE VERIFICATION
# ==============================================================================

# Install required packages if not already present
packages <- c("depmixS4", "MSwM", "data.table", "ggplot2")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(depmixS4)
library(MSwM)
library(data.table)
library(ggplot2)

cat("Loading Python-generated features for R Cross-Verification...\n")
data <- fread("nifty_regime_features.csv")

# Clean Return Series
data <- data[!is.na(ret_1d)]

cat("\n--- 1. Fitting 5-State Gaussian HMM via depmixS4 (R) ---\n")
# Fit 5-State Gaussian HMM
mod <- depmix(ret_1d ~ 1, family = gaussian(), nstates = 5, data = data)
fit_mod <- fit(mod)

# Extract Posterior State Probabilities
post_probs <- posterior(fit_mod)
data$R_HMM_State <- post_probs$state

cat("depmixS4 Model Fit Successful!\n")
print(summary(fit_mod))

cat("\n--- 2. Fitting Markov-Switching Regression via MSwM (R) ---\n")
# Fit Linear Model first
lm_fit <- lm(ret_1d ~ 1, data = data)

# Fit Markov-Switching Model (3 States with switching variance)
ms_fit <- msmFit(lm_fit, k = 3, sw = c(TRUE, TRUE))
cat("MSwM Model Fit Successful!\n")
print(summary(ms_fit))

# Save R Verification Results
fwrite(data, "nifty_r_verification_results.csv")
cat("\nResults saved to 'nifty_r_verification_results.csv'!\n")
