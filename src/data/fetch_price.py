"""
Fetches historical stock price data for AAPL, AMZN, MSFT, NVDA, TSLA
from Yahoo Finance using yfinance.
Saves raw data to data/raw/ as individual CSV files and one combined file.
"""

import yfinance as yf
import pandas as pd
import os
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for one ticker from Yahoo Finance."""
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)

    # Flatten MultiIndex columns (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    df["adj_close"] = df["close"]
    df.insert(0, "ticker", ticker)
    return df

def fetch_all(config: dict) -> pd.DataFrame:
    """Fetch all tickers and return as one combined dataframe."""
    tickers = config["data"]["tickers"]
    start   = config["data"]["start_date"]
    end     = config["data"]["end_date"]

    os.makedirs("data/raw", exist_ok=True)

    all_dfs = []
    for ticker in tickers:
        df = fetch_ticker(ticker, start, end)

        # Save individual CSV
        individual_path = f"data/raw/{ticker}_raw.csv"
        df.to_csv(individual_path)
        print(f"  Saved {ticker} → {individual_path} ({len(df)} rows)")

        all_dfs.append(df)

    # Combine all tickers into one file
    combined = pd.concat(all_dfs)
    combined_path = "data/raw/prices_all.csv"
    combined.to_csv(combined_path)
    print(f"\nCombined file saved → {combined_path}")

    return combined

def load_all(filepath: str = "data/raw/prices_all.csv") -> pd.DataFrame:
    """Load combined price data from CSV."""
    print(f"Loading price data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["date"])

    # Handle MultiIndex column names if present
    df.columns = [col.lower() for col in df.columns]

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
    """Check all expected tickers are present and row counts are consistent."""
    expected = set(config["data"]["tickers"])
    found    = set(df["ticker"].unique())
    missing  = expected - found

    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    else:
        print("All expected tickers present.")

    counts = df.groupby("ticker").size()
    print("\nRow counts per ticker:")
    print(counts.to_string())

    if counts.std() > 5:
        print("\nWARNING: Row counts differ significantly across tickers.")
        print("Check for missing data or date range mismatches.")
    else:
        print("\nRow counts consistent across tickers.")

if __name__ == "__main__":
    config = load_config()

    # Fetch fresh data from yfinance
    combined = fetch_all(config)

    # Reload and validate
    df = load_all()
    validate(df, config)

    # Quick price check
    print("\nPrice sanity check (latest close prices):")
    latest = combined.groupby("ticker").last()["close"]
    print(latest.to_string())