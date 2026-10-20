"""Price scaler: train-only stats, no test leakage, std = 0 guard, clipping, identical scaling in eval."""
import os
import json
import tempfile
import numpy as np
import pandas as pd
import yaml
from src.environment.stock_env import (
    StockTradingEnv, TICKERS, PRICE_FEATURES, TRAIN_END, PRICE_SCALER_PATH,
    load_price_data, load_sentiment_data, build_merged_dataset,
    common_trading_dates, compute_price_scaler, load_price_scaler,
    scale_price_features,
)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

dataset = build_merged_dataset(load_price_data(), load_sentiment_data())
scaler  = compute_price_scaler(dataset)
train_dates = common_trading_dates(dataset, "train")

# 1. Stats come from train days only and match a direct computation
assert max(train_dates) <= TRAIN_END
assert scaler["n_days"] == len(train_dates)
for t in TICKERS:
    v = dataset[t].loc[train_dates, PRICE_FEATURES]
    assert np.allclose(scaler["mean"][t], v.mean().values)
    assert np.allclose(scaler["std"][t], v.std(ddof=0).values)
print(f"Train-only stats OK ({scaler['n_days']} days, up to {scaler['train_end']})")

# 2. No leakage: corrupting every row after TRAIN_END does not change the scaler
corrupt = {}
for t in TICKERS:
    df = dataset[t].copy()
    df.loc[df.index > TRAIN_END, PRICE_FEATURES] = 1e6
    corrupt[t] = df
assert compute_price_scaler(corrupt) == scaler
print("No test leakage OK")

# 3. Committed file is up to date with the data, and rejects post-TRAIN_END fits
on_disk = load_price_scaler()
assert json.loads(json.dumps(scaler)) == on_disk, \
    f"{PRICE_SCALER_PATH} is stale — re-run fit_price_scaler.py"
with tempfile.TemporaryDirectory() as d:
    bad = dict(on_disk, train_end="2023-01-03")
    path = os.path.join(d, "s.json")
    with open(path, "w") as f:
        json.dump(bad, f)
    try:
        load_price_scaler(path)
        raise AssertionError("scaler fitted past TRAIN_END was accepted")
    except ValueError:
        pass
print("Saved scaler OK")

# 4. std = 0 guard and clipping on a synthetic ticker frame
const = {t: dataset[t].copy() for t in TICKERS}
const[TICKERS[0]]["rsi"] = 0.5
s0 = compute_price_scaler(const)
j  = PRICE_FEATURES.index("rsi")
assert s0["std"][TICKERS[0]][j] == 1.0
row = const[TICKERS[0]].loc[train_dates[0], PRICE_FEATURES].values
assert scale_price_features(TICKERS[0], row, s0)[j] == 0.0
extreme = np.array(scaler["mean"][TICKERS[0]]) + 100 * np.array(scaler["std"][TICKERS[0]])
assert np.all(scale_price_features(TICKERS[0], extreme, scaler) == scaler["clip"])
assert np.all(scale_price_features(TICKERS[0], -extreme, scaler) == -scaler["clip"])
print("std = 0 guard and clipping OK")

# 5. Training and evaluation envs observe identically scaled features
env_train = StockTradingEnv(config, mode="train", split="train")
env_eval  = StockTradingEnv(config, mode="eval", split="train")
env_train.reset(seed=0)
env_eval.reset(seed=0)
n = len(PRICE_FEATURES)
for i in [0, 100, len(train_dates) - 2]:
    for e in (env_train, env_eval):
        e.start_idx, e.timestep = i, 0
    a, b = env_train.observe("price_agent"), env_eval.observe("price_agent")
    assert np.array_equal(a, b)
    for k, t in enumerate(TICKERS):
        raw = dataset[t].loc[train_dates[i], PRICE_FEATURES].values
        assert np.allclose(a[k * n:(k + 1) * n],
                           scale_price_features(t, raw, on_disk), atol=1e-5)
    assert np.all(np.abs(a[:n * len(TICKERS)]) <= scaler["clip"])
print("Identical scaling in train and eval envs OK")

print("\nPrice scaler test passed.")
