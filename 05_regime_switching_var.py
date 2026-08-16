import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt

print("Loading regime features for Multivariate Regime-Switching Model...")
df = pd.read_csv("nifty_regime_features.csv", index_col=0, parse_dates=True)

# Target Variable: Daily Returns
y = df['ret_1d'].dropna()

print("\nFitting Markov-Switching Regression Model (3 Regimes: Low, Normal, High Volatility)...")
# Fit 3-Regime Markov Switching Regression with switching variance
ms_model = MarkovRegression(
    y, 
    k_regimes=3, 
    trend='c', 
    switching_variance=True
)
res = ms_model.fit()

print("\n--- Markov-Switching Model Summary ---")
print(res.summary())

# Extract Smoothed Regime Probabilities
smoothed_probs = res.smoothed_marginal_probabilities
for i in range(3):
    df[f'MS_Regime_P{i}'] = smoothed_probs[i]

# Save results
df.to_csv("nifty_regime_switching_results.csv")
print("\nResults saved to 'nifty_regime_switching_results.csv'!")

# Plot Smoothed Probabilities for High Volatility Regime
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['MS_Regime_P2'], label='P(High Volatility / Crash Regime)', color='red')
plt.title("Regime-Switching Probability of High Volatility State (ZeTheta Project)")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.savefig("regime_switching_prob_plot.png")
print("Plot saved successfully as 'regime_switching_prob_plot.png'!")
