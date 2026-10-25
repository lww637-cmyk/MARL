"""Paired per-seed comparisons of MARL+LLM against the cash-tilt controls.

MARL+LLM comes from results/logs/final3 and the three controls
(MARL-average-hold20-cash3 / -cash5.5 / -cash8) from results/logs/cash_control,
paired on the same train seed. The statistics are the ones in
analyze_paired.py: mean, std, number of positive seeds, a paired 95%
bootstrap CI of the mean and an exact two-sided sign test. All three
controls are reported; none are selected.

Writes:
  report/tables/paired_differences_cash.csv         per-seed differences (long format)
  report/tables/paired_differences_cash_by_seed.csv one row per comparison/metric, one column per seed
  report/tables/paired_differences_cash_summary.csv summary statistics per comparison

Note: max_drawdown is stored as a positive magnitude, so a positive
difference in max_drawdown means MARL+LLM had the LARGER (worse) drawdown.

--truedd: the same comparisons with max_drawdown_true and the old column
labelled "range ratio (not a drawdown)", MARL+LLM from
results/logs/replay_final3 (replay_llm.py) and the controls from
results/logs/dd_test. Writes the same three files with the suffix _truedd.
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_paired import (BOOT_SEED, METRICS, METRICS_TRUEDD, N_BOOT, OUT_DIR,
                            TRUEDD_SOURCES, load_truedd, paired_ci, per_seed, sign_test)

LLM_RESULTS = "results/logs/final3/test_period_results.csv"
CASH_RESULTS = "results/logs/cash_control/test_period_results.csv"
LLM = "MARL+LLM"
CONTROLS = ["MARL-average-hold20-cash3", "MARL-average-hold20-cash5.5",
            "MARL-average-hold20-cash8"]


def main(truedd=False):
    if truedd:
        df = load_truedd(*TRUEDD_SOURCES)
        df = df[df["strategy"].isin([LLM] + CONTROLS)]
        metrics, suffix = METRICS_TRUEDD, "_truedd"
        sources = f"{TRUEDD_SOURCES[1]} ({LLM}), {TRUEDD_SOURCES[0]} (controls)"
    else:
        llm_df = pd.read_csv(LLM_RESULTS)
        cash_df = pd.read_csv(CASH_RESULTS)
        df = pd.concat([llm_df[llm_df["strategy"] == LLM],
                        cash_df[cash_df["strategy"].isin(CONTROLS)]], ignore_index=True)
        metrics, suffix = METRICS, ""
        sources = f"{LLM_RESULTS} ({LLM}), {CASH_RESULTS} (controls)"
    seeds = sorted(int(s) for s in df.loc[df["strategy"] == LLM, "train_seed"].unique())
    rng, saved = np.random.default_rng(BOOT_SEED), {}

    long_rows, summary_rows = [], []
    va = per_seed(df, LLM, seeds, metrics)
    for b in CONTROLS:
        vb = per_seed(df, b, seeds, metrics)
        label = f"{LLM} minus {b}"
        for metric in metrics:
            diffs = (va[metric].astype(float) - vb[metric].astype(float)).to_numpy()
            for seed, d in zip(seeds, diffs):
                long_rows.append({"comparison": label, "metric": metric,
                                  "train_seed": seed, "a": va.loc[seed, metric],
                                  "b": vb.loc[seed, metric], "difference": d})
            lo, hi = paired_ci(metric, diffs, rng, saved)
            summary_rows.append({
                "comparison": label, "metric": metric,
                "mean": diffs.mean(), "std": diffs.std(ddof=1),
                "n_positive": int((diffs > 0).sum()), "n_seeds": len(diffs),
                "boot_ci_low": lo, "boot_ci_high": hi,
                "sign_test_p": sign_test(diffs),
            })

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    long_df.to_csv(os.path.join(OUT_DIR, f"paired_differences_cash{suffix}.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, f"paired_differences_cash_summary{suffix}.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Sources: {sources}  seeds: {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}, percentile CI of mean difference")
    print("max_drawdown is a positive magnitude: positive difference = MARL+LLM has larger drawdown\n")

    print("Per-seed differences (MARL+LLM minus control)")
    wide = long_df.pivot_table(index=["comparison", "metric"], columns="train_seed",
                               values="difference", sort=False)
    wide.columns = [f"seed{c}" for c in wide.columns]
    wide.to_csv(os.path.join(OUT_DIR, f"paired_differences_cash_by_seed{suffix}.csv"), float_format="%.4f")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nSummary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truedd", action="store_true",
                        help="use max_drawdown_true; write *_truedd.csv")
    main(parser.parse_args().truedd)
