import yfinance as yf
import pandas as pd

print("Downloading Indian Market Data...")

# 1. Indian Indices & Macro Assets
# Nifty 50 (^NSEI), India VIX (^INDIAVIX), USD/INR (INR=X)
tickers = {
    'Nifty50': '^NSEI',
    'IndiaVIX': '^INDIAVIX',
    'USDINR': 'INR=X'
}

# Download 10 years of historical data
data = yf.download(list(tickers.values()), start="2015-01-01", end="2025-12-31")['Close']

# Rename columns
data.columns = [key for key, val in tickers.items()]

# Calculate Nifty Daily Returns
data['Nifty_Return'] = data['Nifty50'].pct_change()

# Clean missing values
data = data.dropna()

# Save data to CSV
data.to_csv("nifty_market_data.csv")
print("Data downloaded successfully and saved to 'nifty_market_data.csv'!")
print(data.tail())
