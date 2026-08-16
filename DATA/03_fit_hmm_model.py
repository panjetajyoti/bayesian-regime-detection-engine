import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

print("Loading regime features...")
df = pd.read_csv("nifty_regime_features.csv", index_col=0, parse_dates=True)

# 1. Input Features for HMM: Daily Returns and 21-Day Volatility
X = df[['ret_1d', 'vol_21d']].values

print("Fitting 5-State Gaussian Hidden Markov Model (HMM)...")
# 2. Fit Gaussian HMM with 5 hidden states
n_states = 5
model = hmm.GaussianHMM(
    n_components=n_states, 
    covariance_type="diag", 
    n_iter=500, 
    random_state=42
)
model.fit(X)

# Predict hidden regime states and probabilities
df['Regime_State'] = model.predict(X)
probs = model.predict_proba(X)

# 3. Label states dynamically based on Mean Return and Volatility
state_stats = []
for i in range(n_states):
    mean_ret = model.means_[i, 0]
    vol = np.sqrt(model.covars_[i, 0, 0])
    state_stats.append((i, mean_ret, vol))

# Sort: High Return & Low Vol -> Risk-On, Low Return & High Vol -> Risk-Off
state_stats.sort(key=lambda x: (x[1], -x[2]), reverse=True)

regime_names = ['Risk-On', 'Late-Cycle', 'Transitional', 'Post-Shock', 'Risk-Off']
mapping = {state_stats[r][0]: regime_names[r] for r in range(n_states)}

df['Regime_Name'] = df['Regime_State'].map(mapping)

# Add Regime Probabilities to DataFrame
for idx, name in enumerate(regime_names):
    # Find original state index mapped to this name
    orig_state = [k for k, v in mapping.items() if v == name][0]
    df[f'P_{name}'] = probs[:, orig_state]

# 4. Save Classified Regimes
df.to_csv("nifty_hmm_regimes.csv")

print("\n--- HMM Regime Detection Complete ---")
print("Transition Matrix (Probability of moving between states):")
print(np.round(model.transmat_, 3))

print("\nRecent 10 Days Regime Classification:")
print(df[['Nifty50', 'Nifty_Return', 'IndiaVIX', 'Regime_Name']].tail(10))

# 5. Plot Price with Regime Overlays
plt.figure(figsize=(14, 7))
for name in regime_names:
    mask = df['Regime_Name'] == name
    plt.scatter(df.index[mask], df['Nifty50'][mask], label=name, s=5)

plt.title("Nifty 50 - 5-State HMM Regime Classification (ZeTheta Project)")
plt.xlabel("Date")
plt.ylabel("Nifty 50 Level")
plt.legend()
plt.grid(True)
plt.savefig("hmm_regimes_plot.png")
print("\nPlot saved as 'hmm_regimes_plot.png'!")
