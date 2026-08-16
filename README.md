# Bayesian Regime Detection Engine for Equity Direction Forecasting

**Author:** Quantitative Data Analyst Trainee 
**Portfolio:** [GitHub](https://github.com/panjetajyoti) | [LinkedIn](https://www.linkedin.com/in/panjetajyoti2003/)  
**Framework:** ZeTheta Algorithms / Indian AMC Ecosystem
**Framework:** ZeTheta Algorithms / Indian AMC Ecosystem  

## 📌 Executive Summary
This repository contains an end-to-end Bayesian Regime Detection Engine designed for long-only Indian equity mutual fund schemes. Over multi-month horizons, point forecasts for market price collapse due to non-stationarity and low signal-to-noise ratios. Instead, this engine classifies the Indian equity market (Nifty 50) into 5 discrete regimes with calibrated uncertainty:
1. **Risk-On**
2. **Late-Cycle**
3. **Transitional**[cite: 1]
4. **Post-Shock**[cite: 1]
5. **Risk-Off**[cite: 1]

## 🛠️ Repository Architecture & File Index
- `01_data_ingestion.py` - Historical market data acquisition (Nifty 50, India VIX, USD/INR)[cite: 1].
- `02_feature_engineering.py` - Construction of 30+ technical, trend, volatility, and macro features[cite: 1].
- `03_fit_hmm_model.py` - Frequentist Gaussian Hidden Markov Model (hmmlearn)[cite: 1].
- `04_bayesian_hmm.py` - Bayesian HMM via PyMC (NUTS MCMC Sampler) with posterior uncertainty[cite: 1].
- `05_regime_switching_var.py` - Markov-Switching Regression & cross-asset dynamics[cite: 1].
- `06_bayesian_deep_learning.py` - Monte Carlo Dropout Neural Network for epistemic uncertainty quantification[cite: 1].
- `07_ensemble_and_backtest.py` - Stacking ensemble model & 2019–2024 tactical allocation backtest engine[cite: 1].
- `08_r_verification.R` - R implementation (depmixS4 / MSwM) for cross-codebase verification[cite: 1].
- `09_model_card_and_validation.md` - MCMC convergence, calibration (ECE), and model card documentation[cite: 1].
- `10_Main_Project_Report.md` - Comprehensive 40+ page technical report and regulatory alignment[cite: 1].
- `11_Presentation_Deck_and_Script.md` - 18-slide presentation outline and 10-minute demo video script[cite: 1].

## 📊 Outputs & Plots
- `hmm_regimes_plot.png` - HMM 5-state regime classification chart[cite: 1].
- `bayesian_mcmc_trace.png` - MCMC posterior trace diagnostics[cite: 1].
- `regime_switching_prob_plot.png` - High-volatility crash regime probabilities[cite: 1].
- `bnn_uncertainty_plot.png` - Epistemic uncertainty over time[cite: 1].
- `final_backtest_performance.png` - Tactical overlay strategy vs Nifty Buy & Hold benchmark[cite: 1].

## ⚙️ Requirements
`pip install numpy pandas scipy scikit-learn matplotlib seaborn yfinance hmmlearn statsmodels pymc arviz tensorflow`[cite: 1]
