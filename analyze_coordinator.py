"""Coordinator behaviour: LLM weights vs the agent-average weights.

Reads results/logs/final3/coordinator_calls.jsonl. The raw agent actions are
not logged separately, so each call's price and sentiment suggestions are
parsed from its prompt (printed to 0.1%) and averaged exactly as
_fallback_weights() does: (price + sentiment) / 2, clipped and normalised.
This is the average-weight baseline the LLM saw at that call. Rounding in the
prompt bounds the error of each average weight by 0.0005.

The LLM weights are final_weights, the allocation that was actually traded.

Reports per seed and overall (all calls pooled; every seed has the same
number of calls, so this equals the mean of the seed means):
  1. mean absolute difference LLM - average, per ticker and cash
  2. mean cash weight, LLM vs average
  3. share of calls where the LLM weight on the stock with the highest
     average weight is lower than the average (by more than TOL); when
     several stocks tie for highest, their mean LLM weight is compared.
     The share over untied calls only is also reported.
  4. mean weight on each ticker, LLM vs average
  5. number of calls with a fallback or parse failure (must be 0)

Writes report/tables/coordinator_behaviour.csv (long format).
"""
import json
import os
import re

import numpy as np
import pandas as pd

CALLS = "results/logs/final3/coordinator_calls.jsonl"
OUT = "report/tables/coordinator_behaviour.csv"
TICKERS = ["INTC", "AMD", "GS", "CVX", "SBUX"]
ASSETS = TICKERS + ["cash"]
TOL = 0.0005  # prompt rounding bound on the average weights


def parse_suggestion(prompt, header):
    """Weights [*TICKERS, cash] listed under the given prompt header."""
    block = prompt.split(header, 1)[1]
    weights = []
    for name in TICKERS + ["Cash"]:
        m = re.search(rf"^\s*{name}: ([0-9.]+)%", block, re.MULTILINE)
        assert m, f"{name} missing under {header!r}"
        weights.append(float(m.group(1)) / 100)
    return np.array(weights)


def agent_average(price, sentiment):
    w = np.clip((price + sentiment) / 2.0, 0.0, 1.0)
    return w / w.sum()


def load_calls():
    rows = []
    with open(CALLS) as f:
        for line in f:
            r = json.loads(line)
            price = parse_suggestion(r["prompt"], "PRICE AGENT SUGGESTION")
            sent = parse_suggestion(r["prompt"], "SENTIMENT AGENT SUGGESTION")
            failed = (r["outcome"] != "ok" or r["error"] is not None
                      or r["parsed_weights"] is None)
            rows.append({"seed": r["seed"], "step": r["step"], "date": r["date"],
                         "llm": np.array(r["final_weights"], dtype=float),
                         "avg": agent_average(price, sent), "failed": failed})
    return rows


def summarise(scope, calls):
    llm = np.stack([c["llm"] for c in calls])
    avg = np.stack([c["avg"] for c in calls])
    n = len(calls)
    out = []

    def add(metric, asset, l, a, value):
        out.append({"scope": scope, "metric": metric, "asset": asset,
                    "llm": l, "average": a, "value": value, "n_calls": n})

    absdiff = np.abs(llm - avg).mean(axis=0)
    for i, asset in enumerate(ASSETS):
        add("mean_abs_diff", asset, None, None, absdiff[i])
    add("mean_abs_diff", "all_assets_mean", None, None, absdiff.mean())

    for i, asset in enumerate(ASSETS):
        add("mean_weight", asset, llm[:, i].mean(), avg[:, i].mean(),
            llm[:, i].mean() - avg[:, i].mean())

    # top stock(s) by average weight; on a tie, compare the LLM's mean
    # weight over the tied stocks with their (common) average weight
    stocks = avg[:, :len(TICKERS)]
    top_val = stocks.max(axis=1)
    is_top = np.isclose(stocks, top_val[:, None], atol=TOL)
    tied = is_top.sum(axis=1) > 1
    llm_top = (llm[:, :len(TICKERS)] * is_top).sum(axis=1) / is_top.sum(axis=1)
    lower = llm_top < top_val - TOL
    add("share_top_stock_cut", "top_avg_stock", None, None, lower.mean())
    add("share_top_stock_cut_untied", "top_avg_stock", None, None,
        lower[~tied].mean())
    add("top_stock_ties", "top_avg_stock", None, None, int(tied.sum()))

    add("n_fallback_or_parse_failure", "", None, None,
        int(sum(c["failed"] for c in calls)))
    return out


def print_scope(scope, rows):
    r = {(x["metric"], x["asset"]): x for x in rows}
    n = rows[0]["n_calls"]
    print(f"\n=== {scope} ({n} calls) ===")
    print("  asset   |abs diff|   LLM mean   avg mean")
    for a in ASSETS:
        w = r[("mean_weight", a)]
        print(f"  {a:<6} {r[('mean_abs_diff', a)]['value']:10.4f} "
              f"{w['llm']:10.4f} {w['average']:10.4f}")
    print(f"  mean |abs diff| over assets: "
          f"{r[('mean_abs_diff', 'all_assets_mean')]['value']:.4f}")
    cash = r[("mean_weight", "cash")]
    print(f"  cash weight: LLM {cash['llm']:.4f} vs average {cash['average']:.4f}")
    print(f"  share of calls LLM cuts the top average stock: "
          f"{r[('share_top_stock_cut', 'top_avg_stock')]['value']:.3f} "
          f"(untied calls only: "
          f"{r[('share_top_stock_cut_untied', 'top_avg_stock')]['value']:.3f}; "
          f"tied calls: {r[('top_stock_ties', 'top_avg_stock')]['value']})")
    print(f"  fallback / parse failures: "
          f"{r[('n_fallback_or_parse_failure', '')]['value']}")


def main():
    calls = load_calls()
    seeds = sorted({c["seed"] for c in calls})
    rows = []
    for seed in seeds:
        seed_rows = summarise(f"seed_{seed}", [c for c in calls if c["seed"] == seed])
        print_scope(f"seed {seed}", seed_rows)
        rows += seed_rows
    overall = summarise("overall", calls)
    print_scope("overall", overall)
    rows += overall

    failures = overall[-1]["value"]
    assert failures == 0, f"{failures} fallback/parse failures in final3"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
