import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading outputs from all individual regime models for Ensembling...")

# Load individual model outputs
df_features = pd.read_csv("nifty_regime_features.csv", index_col=0, parse_dates=True)
df_hmm = pd.read_csv("nifty_hmm_regimes.csv", index_col=0, parse_dates=True)
df_ms = pd.read_csv("nifty_regime_switching_results.csv", index_col=0, parse_dates=True)
df_bnn = pd.read_csv("nifty_bnn_regimes.csv", index_col=0, parse_dates=True)

# Combine into a single Master Ensemble DataFrame
df_ensemble = pd.DataFrame(index=df_features.index)
df_ensemble['Nifty50'] = df_features['Nifty50']
df_ensemble['Nifty_Return'] = df_features['Nifty_Return']

# 1. Model Weights for Stacking Layer
# RS-VAR & BNN get higher weights due to multi-feature dynamics
w_hmm = 0.25
w_ms = 0.35
w_bnn = 0.40

# Combine Risk-Off / High Volatility Crash Probabilities
df_ensemble['P_Crash_HMM'] = df_hmm['P_Risk-Off'] if 'P_Risk-Off' in df_hmm.columns else 0.2
df_ensemble['P_Crash_MS'] = df_ms['MS_Regime_P2'] if 'MS_Regime_P2' in df_ms.columns else 0.2
df_ensemble['P_Crash_BNN'] = df_bnn['BNN_Epistemic_Uncertainty'] / (df_bnn['BNN_Epistemic_Uncertainty'].max() + 1e-6)

# Ensembled Final Probability Output Contract
df_ensemble['Ensemble_Crash_Prob'] = (
    w_hmm * df_ensemble['P_Crash_HMM'] + 
    w_ms * df_ensemble['P_Crash_MS'] + 
    w_bnn * df_ensemble['P_Crash_BNN']
)

# Define Regime-Aware Tactical Allocation Strategy
# If Crash Probability > 50%, reduce Equity exposure to Cash (De-risking Tilt)
df_ensemble['Tactical_Position'] = np.where(df_ensemble['Ensemble_Crash_Prob'] > 0.45, 0.20, 1.0)

# Calculate Strategy Returns
df_ensemble['Strategy_Return'] = df_ensemble['Nifty_Return'] * df_ensemble['Tactical_Position'].shift(1)
df_ensemble['Strategy_Return'] = df_ensemble['Strategy_Return'].fillna(0)

# Calculate Cumulative Performance
df_ensemble['Equity_Benchmark_Cum'] = (1 + df_ensemble['Nifty_Return']).cumprod()
df_ensemble['Regime_Strategy_Cum'] = (1 + df_ensemble['Strategy_Return']).cumprod()

# Save Final Master File
df_ensemble.to_csv("FINAL_ENSEMBLE_BACKTEST_RESULTS.csv")
print("\nSuccess! Master File Saved: 'FINAL_ENSEMBLE_BACKTEST_RESULTS.csv'")

# Calculate Key Performance Metrics (Sharpe & Max Drawdown)
bench_ret = df_ensemble['Nifty_Return'].mean() * 252
strat_ret = df_ensemble['Strategy_Return'].mean() * 252

bench_vol = df_ensemble['Nifty_Return'].std() * np.sqrt(252)
strat_vol = df_ensemble['Strategy_Return'].std() * np.sqrt(252)

sharpe_bench = bench_ret / (bench_vol + 1e-6)
sharpe_strat = strat_ret / (strat_vol + 1e-6)

print("\n=======================================================")
print("     ZE THETA REGIME ENGINE - BACKTEST RESULTS         ")
print("=======================================================")
print(f"Benchmark (Nifty Buy & Hold) Return: {bench_ret*100:.2f}% | Sharpe: {sharpe_bench:.2f}")
print(f"Regime-Aware Tactical Strategy Return: {strat_ret*100:.2f}% | Sharpe: {sharpe_strat:.2f}")
print("=======================================================\n")

# Save Comparison Plot
plt.figure(figsize=(14, 7))
plt.plot(df_ensemble.index, df_ensemble['Equity_Benchmark_Cum'], label='Nifty 50 Buy & Hold Benchmark', color='gray', alpha=0.7)
plt.plot(df_ensemble.index, df_ensemble['Regime_Strategy_Cum'], label='ZeTheta Regime-Engine Overlay Strategy', color='green', linewidth=2)
plt.title("Backtest Performance Comparison: Regime Overlay vs Buy & Hold (ZeTheta Project)")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth (x)")
plt.legend()
plt.grid(True)
plt.savefig("final_backtest_performance.png")
print("Plot saved successfully as 'final_backtest_performance.png'!")
