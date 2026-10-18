"""Tiny-example check: weekend/holiday roll-forward and train-only log1p normalisation."""
import numpy as np
import pandas as pd
from src.data.fetch_sentiment import aggregate_daily
from src.environment.stock_env import build_merged_dataset, TICKERS, TRAIN_END

T = TICKERS[0]

# Trading days: Thu 2020-12-24, (Fri 12-25 holiday), Mon 12-28, Tue 12-29, plus one test-period day
trading_days = pd.DatetimeIndex(["2020-12-24", "2020-12-28", "2020-12-29", "2022-07-01"], name="date")
price = pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0]}, index=trading_days)

headlines = pd.DataFrame({
    "date": pd.to_datetime(["2020-12-24",
                            "2020-12-25", "2020-12-26", "2020-12-26",   # holiday + Saturday
                            "2020-12-27",                               # Sunday
                            "2020-12-28",
                            *["2022-07-01"] * 20]),                     # test day, many headlines
    "ticker": T,
    "sentiment_score": [0.5, 0.9, 0.2, -0.4, 0.8, 0.1, *[0.0] * 20],
})

daily  = aggregate_daily(headlines)
merged = build_merged_dataset({t: price for t in TICKERS},
                              {t: daily[T] if t == T else None for t in TICKERS})[T]
print(merged[["sentiment_score", "sentiment_std", "headline_count",
              "headline_count_norm", "has_sentiment"]].to_string())

# Monday 12-28 must equal stats over the 5 individual headlines from Fri-Mon
mon = pd.Series([0.9, 0.2, -0.4, 0.8, 0.1])
row = merged.loc["2020-12-28"]
assert row["headline_count"] == 5
assert np.isclose(row["sentiment_score"], mon.mean())
assert np.isclose(row["sentiment_std"], mon.std())
assert merged.loc["2020-12-24", "headline_count"] == 1

# Tuesday has no headlines: forward-filled values, flagged as not real
assert merged.loc["2020-12-29", "has_sentiment"] == 0.0

# Normaliser uses the train-period max only: log1p(5) -> 1.0, test day exceeds 1
assert np.isclose(row["headline_count_norm"], 1.0)
assert np.isclose(merged.loc["2022-07-01", "headline_count_norm"], np.log1p(20) / np.log1p(5))
assert pd.Timestamp("2022-07-01") > TRAIN_END

print("\nSentiment merge test passed.")
