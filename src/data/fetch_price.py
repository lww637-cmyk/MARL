"""
Loads historical stock price data for AAPL, AMZN, GOOGL, MSFT
from the local CSV file at data/raw/prices_all.csv.
Provides utilities to split by ticker and validate date ranges.
"""

import pandas as pd
import os
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_all(filepath: str = "data/raw/prices_all.csv") -> pd.DataFrame:
    """Load full combined price data from CSV."""
    print(f"Loading price data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows across {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Tickers: {sorted(df['ticker'].unique().tolist())}")
    return df

def split_by_ticker(df: pd.DataFrame) -> dict:
    """Split combined dataframe into dict of {ticker: dataframe}."""
    return {ticker: group.set_index("date").drop(columns="ticker")
            for ticker, group in df.groupby("ticker")}

def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter dataframe to specified date range."""
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

def validate(df: pd.DataFrame, config: dict):
    """Check all expected tickers are present and date range is sufficient."""
    expected = set(config["data"]["tickers"])
    found = set(df["ticker"].unique())
    missing = expected - found
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    else:
        print("All expected tickers present.")

    for ticker, group in df.groupby("ticker"):
        print(f"{ticker}: {len(group)} trading days")

if __name__ == "__main__":
    config = load_config()
    df = load_all()
    df = filter_date_range(df, config["data"]["start_date"], config["data"]["end_date"])
    validate(df, config)
    split = split_by_ticker(df)
    print("\nSplit complete. Keys:", list(split.keys()))
    for ticker, data in split.items():
        print(f"{ticker}: {data.shape}")