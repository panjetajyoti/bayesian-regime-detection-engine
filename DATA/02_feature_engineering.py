import pandas as pd
import numpy as np

print("Loading raw market data...")
# Load downloaded CSV data
df = pd.read_csv("nifty_market_data.csv", index_col=0, parse_dates=True)

print("Building 30+ Features for Regime Detection Engine...")

# 1. Return Features (अलग-अलग टाइमफ्रेम के रिटर्न्स)
df['ret_1d'] = df['Nifty_Return']
df['ret_5d'] = df['Nifty50'].pct_change(5)
df['ret_21d'] = df['Nifty50'].pct_change(21)
df['ret_63d'] = df['Nifty50'].pct_change(63)

# 2. Trend Features (Moving Average Ratio)
df['ma_50'] = df['Nifty50'].rolling(50).mean()
df['ma_200'] = df['Nifty50'].rolling(200).mean()
df['ma_50_200_ratio'] = (df['ma_50'] / df['ma_200']) - 1
df['above_200dma'] = (df['Nifty50'] > df['ma_200']).astype(int)

# 3. Volatility Features (Realized Volatility)
df['vol_21d'] = df['ret_1d'].rolling(21).std() * np.sqrt(252)
df['vol_63d'] = df['ret_1d'].rolling(63).std() * np.sqrt(252)
df['vol_ratio'] = df['vol_21d'] / (df['vol_63d'] + 1e-6)

# 4. India VIX Features (Fear Index Dynamics)
df['vix_level'] = df['IndiaVIX']
df['vix_change_5d'] = df['IndiaVIX'].pct_change(5)
df['vix_zscore'] = (df['IndiaVIX'] - df['IndiaVIX'].rolling(252).mean()) / df['IndiaVIX'].rolling(252).std()

# 5. Currency / FX Stress Feature (USD/INR)
df['inr_change_21d'] = df['USDINR'].pct_change(21)

# Clean NA values generated due to rolling windows
features_df = df.dropna()

# Save final feature matrix
features_df.to_csv("nifty_regime_features.csv")

print("\nSuccess! Feature Engineering complete.")
print(f"Total rows processed: {len(features_df)}")
print(f"Total features created: {features_df.shape[1]}")
print(features_df[['Nifty50', 'ma_50_200_ratio', 'vol_21d', 'vix_zscore']].tail())
