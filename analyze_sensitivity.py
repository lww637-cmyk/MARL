"""Transaction-cost and hold-interval sensitivity of the non-LLM strategies.

Reads the --no-llm evaluate_agents.py runs
  results/logs/cost_{R}/  for R = 0, 0.0005, 0.001, 0.002  (--cost-rate R)
  results/logs/hold_{N}/  for N = 5, 10, 20, 40            (--hold-interval N)
and reports, per strategy and setting, the mean and std over the training
seeds of total return, Sharpe, max drawdown and total cost. The two
benchmarks (buy_and_hold, equal_weight_daily_rebalanced) are deterministic
and run once per setting, so their n is 1 and their std is empty. Every
setting is reported; none is selected.

In the hold runs only the *-hold20 rows use the hold interval N (their
labels are kept as "hold20"); the other rows are the same in every hold
setting.

Writes:
  report/tables/sensitivity_cost.csv
  report/tables/sensitivity_hold.csv
"""
import json
import os

import pandas as pd

LOG_DIR = "results/logs"
OUT_DIR = "report/tables"
COST_RATES = ["0", "0.0005", "0.001", "0.002"]
HOLD_INTERVALS = ["5", "10", "20", "40"]
METRICS = {"total_return": "return", "sharpe_ratio": "sharpe",
           "max_drawdown": "drawdown", "total_cost": "total_cost"}
BENCHMARKS = ["buy_and_hold", "equal_weight_daily_rebalanced"]


def summarise(folder: str, setting_col: str, setting) -> pd.DataFrame:
    """Mean/std per strategy over the seeds of one run folder."""
    df = pd.read_csv(os.path.join(folder, "test_period_results.csv"))
    missing = [b for b in BENCHMARKS if b not in set(df["strategy"])]
    if missing:
        raise RuntimeError(f"{folder}: missing benchmark rows {missing}")
    rows = []
    for strategy, sub in df.groupby("strategy", sort=False):
        row = {setting_col: setting, "strategy": strategy, "n": len(sub)}
        for metric, name in METRICS.items():
            row[f"mean_{name}"] = sub[metric].mean()
            row[f"std_{name}"] = sub[metric].std() if len(sub) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def table(prefix: str, settings: list, setting_col: str, meta_key: str) -> pd.DataFrame:
    parts = []
    for s in settings:
        folder = os.path.join(LOG_DIR, f"{prefix}_{s}")
        with open(os.path.join(folder, "run_metadata.json")) as f:
            meta = json.load(f)
        if float(meta[meta_key]) != float(s):
            raise RuntimeError(f"{folder}: {meta_key}={meta[meta_key]}, expected {s}")
        parts.append(summarise(folder, setting_col, meta[meta_key]))
    return pd.concat(parts, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)
    for name, prefix, settings, col, key in [
        ("sensitivity_cost", "cost", COST_RATES, "cost_rate", "cost_rate"),
        ("sensitivity_hold", "hold", HOLD_INTERVALS, "hold_interval", "hold_interval"),
    ]:
        out = table(prefix, settings, col, key)
        path = os.path.join(OUT_DIR, f"{name}.csv")
        out.to_csv(path, index=False)
        print("\n" + "=" * 65)
        print(f"{name.upper()} (mean/std over training seeds; benchmarks n=1)")
        print("=" * 65)
        print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
