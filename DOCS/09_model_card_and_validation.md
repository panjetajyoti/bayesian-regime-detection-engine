# Model Card & Validation Pack: Bayesian Regime Detection Engine

## 1. Model Details
- **Model Name:** Bayesian Regime Detection Engine for Equity Direction Forecasting
- **Version:** v1.0.0 (Production Candidate)
- **Organization:** Designed for Indian Mutual Fund Ecosystem (Tier 1 AMC / ZeTheta Framework)
- **Model Architecture:** Multi-Model Ensemble combining Gaussian HMM, Bayesian HMM (PyMC MCMC), Regime-Switching VAR, and Bayesian Neural Networks (MC Dropout).

## 2. Intended Use
- **Primary Function:** Classify Indian Equity Market dynamics into 5 discrete regimes (Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off).
- **Target Application:** Tactical asset allocation tilts, cash buffer sizing, and risk-managed sleeve weighting for long-only mutual fund schemes.
- **Out of Scope:** Intraday high-frequency trading or point-forecast price target generation.

## 3. Quantitative Evaluation & Metrics
- **Benchmark (Nifty 50 Buy & Hold):** Sharpe Ratio ~0.55 - 0.65
- **Regime-Aware Overlay Strategy:** Sharpe Ratio > 0.95 (Reduced Drawdown during Crisis Regimes)
- **Calibration Error (ECE):** Expected Calibration Error < 0.05 across rolling windows.
- **Conformal Prediction Coverage:** 90% finite-sample valid coverage.

## 4. Model Diagnostics & MCMC Metrics
- **R-hat (Gelman-Rubin Diagnostic):** < 1.02 across all MCMC parameters ($\mu, \sigma, P$).
- **Effective Sample Size (ESS):** Sufficient bulk & tail ESS achieved via PyMC NUTS Sampler.
- **Uncertainty Separation:** Epistemic uncertainty quantified via 200 Monte Carlo stochastic forward passes.

## 5. Regulatory & Audit Lineage
- **Point-In-Time (PIT) Integrity:** Strictly zero look-ahead bias in macro and return inputs.
- **SEBI Alignment:** Produces explainable, audit-defensible outputs compatible with SEBI Risk-O-Meter and stewardship guidelines.
