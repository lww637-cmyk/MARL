"""
Cleans and preprocesses raw price data for AAPL, MSFT, NVDA, AMZN, TSLA.
Computes technical indicators (RSI, MACD, Bollinger Bands) using the ta library.
Saves processed data to data/processed/ as CSV files.
"""

import pandas as pd
import numpy as np
import os
import yaml
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from src.data.fetch_price import load_all, split_by_ticker, filter_date_range

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill missing values then drop any remaining NaNs."""
    df = df.copy()
    df = df.ffill()
    df = df.dropna()
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, and Bollinger Bands from close price."""
    df = df.copy()

    # RSI (14-period)
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    # MACD (12, 26, 9)
    macd = MACD(close=df["close"],
                window_slow=26,
                window_fast=12,
                window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands (20-period)
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Daily return
    df["daily_return"] = df["close"].pct_change()

    # Drop NaN rows introduced by indicator windows
    df = df.dropna()

    return df

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise features for RL consumption.
    Price columns scaled relative to rolling mean.
    Bounded indicators (RSI) scaled to 0-1.
    """
    df = df.copy()

    # RSI: already 0-100, scale to 0-1
    df["rsi"] = df["rsi"] / 100.0

    # MACD values: normalise by close price
    df["macd"] = df["macd"] / df["close"]
    df["macd_signal"] = df["macd_signal"] / df["close"]
    df["macd_diff"] = df["macd_diff"] / df["close"]

    # Bollinger Bands: express as ratio to close
    df["bb_upper"] = df["bb_upper"] / df["close"]
    df["bb_lower"] = df["bb_lower"] / df["close"]
    df["bb_mid"] = df["bb_mid"] / df["close"]

    # Volume: log scale to reduce magnitude
    df["volume"] = np.log1p(df["volume"])

    return df

def process_all(config: dict) -> dict:
    """Run full preprocessing pipeline for all tickers."""
    os.makedirs("data/processed", exist_ok=True)

    raw = load_all()
    raw = filter_date_range(raw,
                            config["data"]["start_date"],
                            config["data"]["end_date"])
    ticker_data = split_by_ticker(raw)

    processed = {}
    for ticker, df in ticker_data.items():
        print(f"Processing {ticker}...")
        df = clean(df)
        df = compute_indicators(df)
        df = normalise(df)

        save_path = f"data/processed/{ticker}_processed.csv"
        df.to_csv(save_path)
        processed[ticker] = df
        print(f"  {ticker}: {df.shape[0]} rows, {df.shape[1]} columns → saved to {save_path}")

    return processed

if __name__ == "__main__":
    config = load_config()
    processed = process_all(config)

    print("\nPreprocessing complete.")
    print("Columns per ticker:", list(processed["AAPL"].columns.tolist()))

    print("\nSample (AAPL first 3 rows):")
    print(processed["AAPL"].head(3).to_string())