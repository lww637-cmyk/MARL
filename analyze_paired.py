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

--truedd: the same comparisons with max_drawdown_true (peak-to-trough), from
results/logs/dd_test (non-LLM rows) and results/logs/replay_final3 (MARL+LLM
replayed by replay_llm.py). The env's old max_drawdown is reported next to it
as "range ratio (not a drawdown)": (max - min) / max, order ignored. Both
use the bootstrap resamples of the old max_drawdown slot (paired_ci), so all
non-drawdown-true rows equal the old tables. Writes the same three files
with the suffix _truedd; the old tables are untouched.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULTS = "results/logs/final3/test_period_results.csv"
OUT_DIR = "report/tables"
METRICS = ["total_return", "sharpe_ratio", "max_drawdown"]
N_BOOT = 10000
BOOT_SEED = 12345

LLM = "MARL+LLM"
RANGE_RATIO = "range ratio (not a drawdown)"
METRICS_TRUEDD = ["total_return", "sharpe_ratio", "max_drawdown_true", RANGE_RATIO]
TRUEDD_SOURCES = ("results/logs/dd_test", "results/logs/replay_final3")

COMPARISONS = [
    ("MARL+LLM", "MARL-average-hold20"),
    ("MARL+LLM", "equal_weight_daily_rebalanced"),
    ("MARL+LLM", "sentiment-only-hold20"),
    ("sentiment-only-hold20", "price-only-hold20"),
    ("MARL-average-hold20", "MARL-no-coordinator"),
]


def load_truedd(base_dir, replay_dir):
    """
    Non-LLM rows of an evaluate_agents.py run that has max_drawdown_true plus
    the MARL+LLM rows replayed by replay_llm.py, with the old max_drawdown
    column renamed RANGE_RATIO.
    """
    base = pd.read_csv(os.path.join(base_dir, "test_period_results.csv"))
    replay = pd.read_csv(os.path.join(replay_dir, "test_period_results.csv"))
    assert LLM not in set(base["strategy"]), f"{base_dir} has {LLM} rows"
    assert set(replay["strategy"]) == {LLM}, f"{replay_dir} has non-{LLM} rows"
    df = pd.concat([base, replay], ignore_index=True)
    return df.rename(columns={"max_drawdown": RANGE_RATIO})


def per_seed(df, strategy, seeds, metrics=METRICS):
    rows = df[df["strategy"] == strategy]
    if rows["train_seed"].isna().all():
        # Seed-independent baseline: same value paired with every train seed
        assert len(rows) == 1, f"expected one row for {strategy}, got {len(rows)}"
        return pd.DataFrame([rows.iloc[0][metrics]] * len(seeds), index=seeds)
    rows = rows.set_index(rows["train_seed"].astype(int))[metrics]
    assert rows.index.is_unique, f"duplicate seeds for {strategy}"
    return rows.loc[seeds]


def bootstrap_ci(diffs, rng):
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_ci(metric, diffs, rng, saved):
    """
    bootstrap_ci, except that RANGE_RATIO reuses the resamples drawn for the
    metric before it (max_drawdown_true): the drawdown pair takes the one
    slot of the old max_drawdown in the rng stream, so with --truedd every
    other CI equals the old tables'. saved: dict kept across calls.
    """
    if metric == RANGE_RATIO:
        after = rng.bit_generator.state
        rng.bit_generator.state = saved["state"]
        ci = bootstrap_ci(diffs, rng)
        rng.bit_generator.state = after
        return ci
    saved["state"] = rng.bit_generator.state
    return bootstrap_ci(diffs, rng)


def sign_test(diffs):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    return binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue


def main(truedd=False):
    if truedd:
        df, metrics, suffix = load_truedd(*TRUEDD_SOURCES), METRICS_TRUEDD, "_truedd"
        source = " + ".join(TRUEDD_SOURCES)
    else:
        df, metrics, suffix, source = pd.read_csv(RESULTS), METRICS, "", RESULTS
    seeds = sorted(int(s) for s in df["train_seed"].dropna().unique())
    rng, saved = np.random.default_rng(BOOT_SEED), {}

    long_rows, summary_rows = [], []
    for a, b in COMPARISONS:
        va, vb = per_seed(df, a, seeds, metrics), per_seed(df, b, seeds, metrics)
        label = f"{a} minus {b}"
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
    long_df.to_csv(os.path.join(OUT_DIR, f"paired_differences{suffix}.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, f"paired_differences_summary{suffix}.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Source: {source}  seeds: {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}, percentile CI of mean difference")
    print("max_drawdown is a positive magnitude: positive difference = A has larger drawdown\n")

    print("Per-seed differences (A minus B)")
    wide = long_df.pivot_table(index=["comparison", "metric"], columns="train_seed",
                               values="difference", sort=False)
    wide.columns = [f"seed{c}" for c in wide.columns]
    wide.to_csv(os.path.join(OUT_DIR, f"paired_differences_by_seed{suffix}.csv"), float_format="%.4f")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nSummary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truedd", action="store_true",
                        help="use max_drawdown_true; write *_truedd.csv")
    main(parser.parse_args().truedd)
