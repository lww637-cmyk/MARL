"""Paired per-seed comparisons of MARL+LLM against same-interval controls, for N = 10, 20, 40.

For each coordinator interval N, MARL+LLM is paired on train seed with the
controls run at that same N:
  MARL-average-hold20, MARL-average-hold20-cash3 / -cash5.5 / -cash8
and, for reference, the single seed-independent equal_weight_daily_rebalanced
value. The "-hold20" labels are fixed names; the hold interval actually used
is that folder's N (see each folder's run_metadata.json).

Sources:
  N=10  results/logs/llm_cf10
  N=20  results/logs/final3 (MARL+LLM, average control, equal weight)
        results/logs/cash_control (cash controls; final3 has none). The two
        folders' MARL-average-hold20 rows are checked to be identical.
  N=40  results/logs/llm_cf40

The statistics are the ones in analyze_paired.py: mean, std, number of
positive seeds, a paired 95% bootstrap CI of the mean (10000 resamples, seed
12345) and an exact two-sided sign test. All comparisons are reported; none
are selected.

Writes:
  report/tables/paired_intervals.csv       summary per N/comparison/metric (long format)
  report/tables/paired_intervals_by_seed.csv per-seed differences (long format)
  report/tables/llm_across_intervals.csv   MARL+LLM and average-control mean/std per N

Note: max_drawdown is stored as a positive magnitude, so a positive
difference in max_drawdown means MARL+LLM had the LARGER (worse) drawdown.
"""
import json
import os

import numpy as np
import pandas as pd

from analyze_paired import BOOT_SEED, METRICS, N_BOOT, OUT_DIR, bootstrap_ci, per_seed, sign_test

LLM = "MARL+LLM"
AVERAGE = "MARL-average-hold20"
CASH = ["MARL-average-hold20-cash3", "MARL-average-hold20-cash5.5",
        "MARL-average-hold20-cash8"]
EQUAL_WEIGHT = "equal_weight_daily_rebalanced"
CONTROLS = [AVERAGE] + CASH + [EQUAL_WEIGHT]

# N -> (main folder, folder holding the cash controls)
SOURCES = {
    10: ("results/logs/llm_cf10", "results/logs/llm_cf10"),
    20: ("results/logs/final3", "results/logs/cash_control"),
    40: ("results/logs/llm_cf40", "results/logs/llm_cf40"),
}
ACROSS_METRICS = METRICS + ["total_cost"]


def load(folder):
    with open(os.path.join(folder, "run_metadata.json")) as f:
        meta = json.load(f)
    return pd.read_csv(os.path.join(folder, "test_period_results.csv")), meta


def load_interval(n):
    main_dir, cash_dir = SOURCES[n]
    main_df, main_meta = load(main_dir)
    assert main_meta["call_frequency"] == n, f"{main_dir}: call_frequency {main_meta['call_frequency']} != {n}"
    assert main_meta.get("hold_interval", n) == n, f"{main_dir}: hold_interval != {n}"
    if cash_dir == main_dir:
        return main_df, [main_dir]

    cash_df, cash_meta = load(cash_dir)
    assert cash_meta["call_frequency"] == n, f"{cash_dir}: call_frequency {cash_meta['call_frequency']} != {n}"
    # The average control must be the same run in both folders
    cols = ["train_seed"] + ACROSS_METRICS
    a = main_df[main_df["strategy"] == AVERAGE].sort_values("train_seed")[cols].reset_index(drop=True)
    b = cash_df[cash_df["strategy"] == AVERAGE].sort_values("train_seed")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    df = pd.concat([main_df[~main_df["strategy"].isin(CASH)],
                    cash_df[cash_df["strategy"].isin(CASH)]], ignore_index=True)
    return df, [main_dir, cash_dir]


def main():
    long_rows, summary_rows, across_rows = [], [], []
    sources = {}
    for n in SOURCES:
        df, sources[n] = load_interval(n)
        seeds = sorted(int(s) for s in df.loc[df["strategy"] == LLM, "train_seed"].unique())
        # Fresh generator per N so each N's CIs do not depend on the others
        rng = np.random.default_rng(BOOT_SEED)
        va = per_seed(df, LLM, seeds)
        for b in CONTROLS:
            vb = per_seed(df, b, seeds)
            label = f"{LLM} minus {b}"
            for metric in METRICS:
                diffs = (va[metric].astype(float) - vb[metric].astype(float)).to_numpy()
                for seed, d in zip(seeds, diffs):
                    long_rows.append({"N": n, "comparison": label, "metric": metric,
                                      "train_seed": seed, "a": va.loc[seed, metric],
                                      "b": vb.loc[seed, metric], "difference": d})
                lo, hi = bootstrap_ci(diffs, rng)
                summary_rows.append({
                    "N": n, "comparison": label, "metric": metric,
                    "mean": diffs.mean(), "std": diffs.std(ddof=1),
                    "n_positive": int((diffs > 0).sum()), "n_seeds": len(diffs),
                    "boot_ci_low": lo, "boot_ci_high": hi,
                    "sign_test_p": sign_test(diffs),
                })

        for strategy in [LLM, AVERAGE]:
            vals = df[df["strategy"] == strategy][ACROSS_METRICS].astype(float)
            row = {"N": n, "strategy": strategy, "n_seeds": len(vals)}
            for metric in ACROSS_METRICS:
                row[f"{metric}_mean"] = vals[metric].mean()
                row[f"{metric}_std"] = vals[metric].std(ddof=1)
            across_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)
    across_df = pd.DataFrame(across_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "paired_intervals.csv"), index=False)
    long_df.to_csv(os.path.join(OUT_DIR, "paired_intervals_by_seed.csv"), index=False)
    across_df.to_csv(os.path.join(OUT_DIR, "llm_across_intervals.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED} (reset per N), percentile CI of mean difference")
    print("max_drawdown is a positive magnitude: positive difference = MARL+LLM has larger drawdown")
    print(f"{EQUAL_WEIGHT} is a single seed-independent value, paired with every seed for reference")
    for n in SOURCES:
        print(f"\n=== N = {n}  sources: {', '.join(sources[n])}")
        part = summary_df[summary_df["N"] == n].drop(columns="N")
        print(part.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMARL+LLM vs average control across intervals (mean/std over seeds)")
    print(across_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
