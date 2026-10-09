"""
Loads financial news headlines from local CSV,
filters for target tickers, scores with FinBERT,
and aggregates to daily company-specific sentiment scores.
Saves results to data/sentiment/ as CSV files.

NOTE: FinBERT scoring is slow on CPU.
Run score_and_save() on Google Colab for full dataset.
"""

import pandas as pd
import numpy as np
import os
import yaml
from transformers import pipeline

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_news(filepath: str = "data/raw/news_filtered.csv",
              tickers: list = None) -> pd.DataFrame:
    """
    Load and filter news data efficiently.
    Only loads columns needed for sentiment scoring.
    """
    print(f"Loading news data from {filepath}...")

    # Load only needed columns to save memory
    usecols = ["date", "article_title", "stock_symbol"]

    df = pd.read_csv(
        filepath,
        usecols=usecols,
        parse_dates=["date"],
        on_bad_lines="skip"   # skip malformed rows from embedded newlines
    )

    # Normalise timezone-aware dates to date only
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Rename to match our convention
    df = df.rename(columns={"stock_symbol": "ticker"})

    # Filter for target tickers
    if tickers:
        df = df[df["ticker"].isin(tickers)].reset_index(drop=True)

    # Drop rows with missing headline
    df = df.dropna(subset=["article_title"]).reset_index(drop=True)

    print(f"Loaded {len(df)} headlines across {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    for ticker in sorted(df["ticker"].unique()):
        print(f"  {ticker}: {len(df[df['ticker'] == ticker])} headlines")

    return df

def score_headlines(df: pd.DataFrame,
                    batch_size: int = 32) -> pd.DataFrame:
    """
    Score each headline using FinBERT.
    Returns df with added sentiment_score column.
    Positive = +confidence, Negative = -confidence, Neutral = 0.
    """
    print("Loading FinBERT model...")
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )

    headlines = df["article_title"].tolist()
    scores = []

    print(f"Scoring {len(headlines)} headlines in batches of {batch_size}...")
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i: i + batch_size]
        results = finbert(batch)
        for r in results:
            label = r["label"].lower()
            confidence = r["score"]
            if label == "positive":
                scores.append(confidence)
            elif label == "negative":
                scores.append(-confidence)
            else:
                scores.append(0.0)

        if (i // batch_size) % 10 == 0:
            print(f"  Scored {min(i + batch_size, len(headlines))}/{len(headlines)}")

    df = df.copy()
    df["sentiment_score"] = scores
    return df

def aggregate_daily(df: pd.DataFrame) -> dict:
    """
    Aggregate multiple headlines per day into a single daily sentiment score.
    Uses mean sentiment score per ticker per day.
    Also computes headline count and score std as additional features.
    """
    aggregated = {}

    for ticker, group in df.groupby("ticker"):
        daily = group.groupby("date").agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            headline_count=("sentiment_score", "count")
        ).reset_index()

        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
        daily = daily.set_index("date")
        aggregated[ticker] = daily

    return aggregated

def save_sentiment(aggregated: dict):
    """Save daily sentiment scores to data/sentiment/."""
    os.makedirs("data/sentiment", exist_ok=True)
    for ticker, df in aggregated.items():
        path = f"data/sentiment/{ticker}_sentiment.csv"
        df.to_csv(path)
        print(f"Saved {ticker} sentiment → {path} ({len(df)} trading days)")

def score_and_save(config: dict):
    """Full pipeline: load → score → aggregate → save."""
    tickers = config["data"]["tickers"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    # Load and filter
    df = load_news(tickers=tickers)

    # Filter date range
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    print(f"\nAfter date filter: {len(df)} headlines")

    # Score with FinBERT
    df = score_headlines(df)

    # Aggregate to daily
    aggregated = aggregate_daily(df)

    # Save
    save_sentiment(aggregated)
    print("\nSentiment pipeline complete.")

def load_saved_sentiment(ticker: str) -> pd.DataFrame:
    """Load precomputed sentiment scores for a given ticker."""
    path = f"data/sentiment/{ticker}_sentiment.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sentiment file found for {ticker}. "
            f"Run score_and_save() first on Colab."
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df

if __name__ == "__main__":
    config = load_config()

    # Step 1: Just validate the news file loads correctly (safe to run locally)
    tickers = config["data"]["tickers"]
    df = load_news(tickers=tickers)
    print("\nNews file loaded successfully.")
    print("Sample headlines:")
    print(df[["date", "ticker", "article_title"]].head(5).to_string())

    # Step 2: FinBERT scoring — uncomment to run on Colab
    # score_and_save(config)