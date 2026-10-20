"""
Fit the price-feature scaler on the TRAIN split only and save it to
data/price_scaler.json (committed to git, so Colab gets it with the clone).
Re-run only when the tickers, price features or processed price data change.

  python fit_price_scaler.py
"""

import json
import numpy as np
from src.environment.stock_env import (
    TICKERS, PRICE_FEATURES, PRICE_SCALER_PATH, TRAIN_END,
    load_price_data, load_sentiment_data, build_merged_dataset,
    common_trading_dates, compute_price_scaler, scale_price_features,
)


def main():
    dataset = build_merged_dataset(load_price_data(), load_sentiment_data())
    scaler  = compute_price_scaler(dataset)
    assert scaler["train_end"] <= str(TRAIN_END.date())

    with open(PRICE_SCALER_PATH, "w") as f:
        json.dump(scaler, f, indent=1)
    print(f"Saved {PRICE_SCALER_PATH} — {scaler['n_days']} train days "
          f"({scaler['train_start']} .. {scaler['train_end']})")

    # Range after scaling on the train split, per feature (all tickers)
    train_dates = common_trading_dates(dataset, "train")
    scaled = np.stack([
        scale_price_features(t, dataset[t].loc[train_dates, PRICE_FEATURES].values, scaler)
        for t in TICKERS
    ])                                  # (ticker, day, feature)
    print(f"\n{'feature':<14}{'min':>8}{'max':>8}{'mean':>8}{'std':>8}  clipped")
    for j, feat in enumerate(PRICE_FEATURES):
        v = scaled[:, :, j]
        clipped = (np.abs(v) >= scaler["clip"]).mean() * 100
        print(f"{feat:<14}{v.min():>8.2f}{v.max():>8.2f}{v.mean():>8.2f}{v.std():>8.2f}  {clipped:.2f}%")


if __name__ == "__main__":
    main()
