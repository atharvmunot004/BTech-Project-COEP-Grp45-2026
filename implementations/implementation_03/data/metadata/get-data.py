import yfinance as yf
import pandas as pd

# TCS NSE ticker
ticker = "NTPC.NS"

# Download 10 years of daily data
data = yf.download(
    ticker,
    period="10y",
    interval="1d",
    auto_adjust=True   # adjusts for splits/dividends
)

# Save to CSV
output_file = "NTPC_10yr_daily.csv"
data.to_csv(output_file)

print(f"Downloaded {len(data)} rows for {ticker}")
print(f"Saved to {output_file}")
print(data.head())
