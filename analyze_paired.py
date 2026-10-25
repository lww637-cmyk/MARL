"""Paired per-seed comparisons of test-period results.

Reads results/logs/final3/test_period_results.csv and, for each comparison
(A minus B on the same train seed), reports mean, std, number of positive
seeds, a paired 95% bootstrap CI of the mean, and an exact two-sided sign test.
All comparisons are reported; none are selected.

Writes:
  report/tables/paired_differences.csv        per-seed differences (long format)
  report/tables/paired_differences_by_seed.csv one row per comparison/metric, one column per seed
  report/tables/paired_differences_summary.csv summary statistics per comparison

Note: max_drawdown is stored as a positive magnitude, so a positive
difference in max_drawdown means A had the LARGER (worse) drawdown.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULTS = "results/logs/final3/test_period_results.csv"
OUT_DIR = "report/tables"
METRICS = ["total_return", "sharpe_ratio", "max_drawdown"]
N_BOOT = 10000
BOOT_SEED = 12345

COMPARISONS = [
    ("MARL+LLM", "MARL-average-hold20"),
    ("MARL+LLM", "equal_weight_daily_rebalanced"),
    ("MARL+LLM", "sentiment-only-hold20"),
    ("sentiment-only-hold20", "price-only-hold20"),
    ("MARL-average-hold20", "MARL-no-coordinator"),
]


def per_seed(df, strategy, seeds):
    rows = df[df["strategy"] == strategy]
    if rows["train_seed"].isna().all():
        # Seed-independent baseline: same value paired with every train seed
        assert len(rows) == 1, f"expected one row for {strategy}, got {len(rows)}"
        return pd.DataFrame([rows.iloc[0][METRICS]] * len(seeds), index=seeds)
    rows = rows.set_index(rows["train_seed"].astype(int))[METRICS]
    assert rows.index.is_unique, f"duplicate seeds for {strategy}"
    return rows.loc[seeds]


def bootstrap_ci(diffs, rng):
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def sign_test(diffs):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    return binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue


def main():
    df = pd.read_csv(RESULTS)
    seeds = sorted(int(s) for s in df["train_seed"].dropna().unique())
    rng = np.random.default_rng(BOOT_SEED)

    long_rows, summary_rows = [], []
    for a, b in COMPARISONS:
        va, vb = per_seed(df, a, seeds), per_seed(df, b, seeds)
        label = f"{a} minus {b}"
        for metric in METRICS:
            diffs = (va[metric].astype(float) - vb[metric].astype(float)).to_numpy()
            for seed, d in zip(seeds, diffs):
                long_rows.append({"comparison": label, "metric": metric,
                                  "train_seed": seed, "a": va.loc[seed, metric],
                                  "b": vb.loc[seed, metric], "difference": d})
            lo, hi = bootstrap_ci(diffs, rng)
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
    long_df.to_csv(os.path.join(OUT_DIR, "paired_differences.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "paired_differences_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Source: {RESULTS}  seeds: {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}, percentile CI of mean difference")
    print("max_drawdown is a positive magnitude: positive difference = A has larger drawdown\n")

    print("Per-seed differences (A minus B)")
    wide = long_df.pivot_table(index=["comparison", "metric"], columns="train_seed",
                               values="difference", sort=False)
    wide.columns = [f"seed{c}" for c in wide.columns]
    wide.to_csv(os.path.join(OUT_DIR, "paired_differences_by_seed.csv"), float_format="%.4f")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nSummary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
