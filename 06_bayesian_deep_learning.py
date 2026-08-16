import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

print("Loading regime features for Bayesian Deep Learning (MC Dropout)...")
df = pd.read_csv("nifty_regime_features.csv", index_col=0, parse_dates=True)

# Select Input Features
feature_cols = ['ret_1d', 'ret_5d', 'vol_21d', 'vix_level', 'vix_zscore', 'ma_50_200_ratio']
X = df[feature_cols].values

# Dummy Regime Labels (5 Regimes from HMM classification if available, else Quantile binned)
if 'Regime_State' in df.columns:
    y = df['Regime_State'].values
else:
    # Bin returns into 5 proxy regimes for training demo
    y = pd.qcut(df['ret_1d'], q=5, labels=False).values

# Normalize Features
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / (X_std + 1e-6)

print("\nBuilding Monte Carlo Dropout Bayesian Neural Network...")

# Define MC Dropout Layer that remains ACTIVE during Inference
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Build BNN Architecture
inputs = layers.Input(shape=(len(feature_cols),))
x = layers.Dense(64, activation='relu')(inputs)
x = MCDropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
x = MCDropout(0.3)(x)
outputs = layers.Dense(5, activation='softmax')(x)

bnn_model = Model(inputs, outputs)
bnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train BNN Model
print("Training Neural Network...")
bnn_model.fit(X_scaled, y, epochs=20, batch_size=32, verbose=0)
print("Training Complete!")

# Generate Monte Carlo Predictions (200 stochastic forward passes)
print("\nGenerating 200 Monte Carlo Stochastic Forward Passes for Uncertainty Quantification...")
n_mc_samples = 200
mc_predictions = np.stack([bnn_model.predict(X_scaled, verbose=0) for _ in range(n_mc_samples)])

# Compute Mean Probability and Epistemic Uncertainty (Standard Deviation across passes)
mean_probs = mc_predictions.mean(axis=0)
epistemic_unc = mc_predictions.std(axis=0)

df['BNN_Dominant_Regime'] = mean_probs.argmax(axis=1)
df['BNN_Epistemic_Uncertainty'] = epistemic_unc.mean(axis=1)

# Save BNN Predictions
df.to_csv("nifty_bnn_regimes.csv")
print("\nResults saved to 'nifty_bnn_regimes.csv'!")

# Plot Epistemic Uncertainty over Time
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['BNN_Epistemic_Uncertainty'], label='Model Epistemic Uncertainty (MC Dropout)', color='purple')
plt.title("Bayesian Deep Learning Model Uncertainty Over Time (ZeTheta Project)")
plt.xlabel("Date")
plt.ylabel("Uncertainty Score")
plt.legend()
plt.grid(True)
plt.savefig("bnn_uncertainty_plot.png")
print("Plot saved successfully as 'bnn_uncertainty_plot.png'!")
