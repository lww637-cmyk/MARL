"""Train-period vs test-period results of every non-LLM strategy.

Reads two --no-llm evaluate_agents.py runs over training seeds 0-5:
  results/logs/train_period/train_period_results.csv           (--period train)
  results/logs/controls_check6_verify/test_period_results.csv  (--period test)
The test run reproduces results/logs/controls_check6 exactly on every
strategy that folder holds and adds the MARL-average-hold20-cash* controls.

Per strategy it reports the mean and std over the training seeds of total
return, Sharpe and max drawdown on each period, plus test minus train. The
two benchmarks are deterministic and run once per period, so their n is 1
and their std is empty. The train split is the agents' training data, so the
train columns are in-sample.

max_drawdown is StockTradingEnv's (max value - min value) / max value over the
episode, not a peak-to-trough drawdown: on a path that rises strongly it is
driven by growth (minimum early, maximum late), as in most train rows.

Writes:
  report/tables/train_vs_test.csv

--truedd: the same table from results/logs/dd_train and results/logs/dd_test
(same runs plus max_drawdown_true), with the drawdown columns from
max_drawdown_true ("drawdown_true") and the old max_drawdown kept next to them
labelled "range ratio (not a drawdown)". Writes
report/tables/train_vs_test_truedd.csv; the old table is untouched.
"""
import argparse
import json
import os

import pandas as pd

from analyze_paired import RANGE_RATIO

LOG_DIR = "results/logs"
OUT_DIR = "report/tables"
RUNS = {"train": ("train_period", "train_period_results.csv"),
        "test": ("controls_check6_verify", "test_period_results.csv")}
METRICS = {"total_return": "return", "sharpe_ratio": "sharpe",
           "max_drawdown": "drawdown"}
RUNS_TRUEDD = {"train": ("dd_train", "train_period_results.csv"),
               "test": ("dd_test", "test_period_results.csv")}
METRICS_TRUEDD = {"total_return": "return", "sharpe_ratio": "sharpe",
                  "max_drawdown_true": "drawdown_true", "max_drawdown": RANGE_RATIO}
SEEDS = [0, 1, 2, 3, 4, 5]


def load(period: str, runs: dict = RUNS) -> pd.DataFrame:
    folder, name = runs[period]
    folder = os.path.join(LOG_DIR, folder)
    with open(os.path.join(folder, "run_metadata.json")) as f:
        meta = json.load(f)
    if meta.get("period", "test") != period or meta["use_llm"] or meta["train_seeds"] != SEEDS:
        raise RuntimeError(f"{folder}: unexpected run_metadata {meta}")
    return pd.read_csv(os.path.join(folder, name))


def summarise(df: pd.DataFrame, period: str, metrics: dict = METRICS) -> pd.DataFrame:
    """Mean/std per strategy over the training seeds of one period."""
    rows = []
    for strategy, sub in df.groupby("strategy", sort=False):
        row = {"strategy": strategy, f"n_{period}": len(sub)}
        for metric, name in metrics.items():
            row[f"{period}_mean_{name}"] = sub[metric].mean()
            row[f"{period}_std_{name}"] = sub[metric].std() if len(sub) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(truedd=False):
    runs, metrics, suffix = ((RUNS_TRUEDD, METRICS_TRUEDD, "_truedd") if truedd
                             else (RUNS, METRICS, ""))
    train = summarise(load("train", runs), "train", metrics)
    test = summarise(load("test", runs), "test", metrics)
    if set(train["strategy"]) != set(test["strategy"]):
        raise RuntimeError("train and test runs hold different strategies")
    out = train.merge(test, on="strategy", how="left")
    cols = ["strategy", "n_train", "n_test"]
    for name in metrics.values():
        out[f"diff_mean_{name}"] = out[f"test_mean_{name}"] - out[f"train_mean_{name}"]
        cols += [f"train_mean_{name}", f"train_std_{name}",
                 f"test_mean_{name}", f"test_std_{name}", f"diff_mean_{name}"]
    out = out[cols]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"train_vs_test{suffix}.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truedd", action="store_true",
                        help="use max_drawdown_true; write train_vs_test_truedd.csv")
    main(parser.parse_args().truedd)
