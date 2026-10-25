"""Does the LLM's extra cash track recent market conditions?

For each final3 coordinator call (results/logs/final3/coordinator_calls.jsonl)
takes the LLM cash weight (final_weights, as traded) and the agent-average
cash weight (rebuilt from the prompt, as in analyze_coordinator.py), and
merges them with two features of the equal-weight daily-rebalanced
portfolio at the call date:
  trailing_return_20d   compounded return over the last WINDOW trading days
  trailing_max_dd_20d   max drawdown (positive magnitude) of its value over
                        the same window
Both use closes up to and including the call date only. Weights chosen on
that date earn the close-to-close return to the next day (StockTradingEnv
._execute_step), so the call-date close is already known: no future data.
Prices come from data/processed/{ticker}_processed.csv, the env's source;
the window may reach back before the test split, which is still past data.

Prints the Pearson and Spearman correlation of extra cash (LLM cash minus
average cash) with each feature, per seed and overall. Overall pools all
114 calls; the 19 call dates are shared by every seed, so pooled calls are
not independent.

Writes report/tables/cash_timing.csv (one row per call).
"""
import os

import numpy as np
import pandas as pd

from analyze_coordinator import TICKERS, load_calls

OUT = "report/tables/cash_timing.csv"
WINDOW = 20
FEATURES = ["trailing_return_20d", "trailing_max_dd_20d"]


def equal_weight_value():
    """Equal-weight daily-rebalanced portfolio value index on the common dates."""
    closes = pd.concat(
        [pd.read_csv(f"data/processed/{t}_processed.csv", parse_dates=["date"],
                     index_col="date")["close"].rename(t) for t in TICKERS],
        axis=1).dropna()
    daily = closes.pct_change().mean(axis=1).fillna(0.0)
    return (1 + daily).cumprod()


def trailing_features(value, date):
    """Return and max drawdown over the WINDOW days ending at date (inclusive)."""
    i = value.index.get_loc(date)
    assert i >= WINDOW, f"not enough history before {date.date()}"
    v = value.iloc[i - WINDOW:i + 1].to_numpy()
    drawdown = 1 - v / np.maximum.accumulate(v)
    return v[-1] / v[0] - 1, drawdown.max()


def correlations(df):
    out = {}
    for f in FEATURES:
        out[f"pearson_{f}"] = df["extra_cash"].corr(df[f])
        out[f"spearman_{f}"] = df["extra_cash"].corr(df[f], method="spearman")
    return out


def main():
    value = equal_weight_value()
    rows = []
    for c in load_calls():
        date = pd.Timestamp(c["date"])
        ret, dd = trailing_features(value, date)
        rows.append({"seed": c["seed"], "step": c["step"], "date": c["date"],
                     "llm_cash": c["llm"][-1], "avg_cash": c["avg"][-1],
                     "extra_cash": c["llm"][-1] - c["avg"][-1],
                     "trailing_return_20d": ret, "trailing_max_dd_20d": dd})
    df = pd.DataFrame(rows)
    assert len(df) == 114, len(df)

    print(f"Equal-weight portfolio features over the {WINDOW} trading days "
          f"ending at each call date (inclusive)")
    print("Overall pools all calls; call dates repeat across seeds, so pooled "
          "calls are not independent.\n")
    summary = []
    for scope, part in [*((f"seed {s}", g) for s, g in df.groupby("seed")),
                        ("overall", df)]:
        summary.append({"scope": scope, "n_calls": len(part),
                        "mean_extra_cash": part["extra_cash"].mean(),
                        **correlations(part)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
