"""Paired per-seed comparisons of the hold controls against static allocations.

Static control (no Gemini): for each train seed, the agent's mean deterministic
allocation over the TRAINING split (six weights incl. cash, as in
policy_check.py) is held fixed on the test period with the same rebalance
schedule and costs as the *-hold20 controls. Produced by
  python evaluate_agents.py --no-llm --train-seeds 0 1 2 3 4 5 --tag static_control
(weights in results/logs/static_control/static_weights.csv).

Comparisons (A minus B on the same train seed):
  sentiment-only-hold20   minus sentiment-static-hold20
  price-only-hold20       minus price-static-hold20
  sentiment-static-hold20 minus price-static-hold20

The statistics are the ones in analyze_paired.py: mean, std, number of
positive seeds, a paired 95% bootstrap CI of the mean (10000 resamples, seed
12345, one generator shared across all rows in order) and an exact two-sided
sign test. All comparisons are reported; none are selected.

Metrics: total_return, sharpe_ratio, max_drawdown_true (peak-to-trough,
positive magnitude: a positive difference means A had the LARGER drawdown).

Writes:
  report/tables/paired_static_control.csv          summary per comparison/metric
  report/tables/paired_static_control_by_seed.csv  per-seed differences (long)
"""
import os

import numpy as np
import pandas as pd

from analyze_paired import BOOT_SEED, N_BOOT, OUT_DIR, bootstrap_ci, per_seed, sign_test

SOURCE = "results/logs/static_control"
METRICS = ["total_return", "sharpe_ratio", "max_drawdown_true"]
COMPARISONS = [
    ("sentiment-only-hold20", "sentiment-static-hold20"),
    ("price-only-hold20", "price-static-hold20"),
    ("sentiment-static-hold20", "price-static-hold20"),
]


def main():
    df = pd.read_csv(os.path.join(SOURCE, "test_period_results.csv"))
    seeds = sorted(int(s) for s in df["train_seed"].dropna().unique())
    rng = np.random.default_rng(BOOT_SEED)

    long_rows, summary_rows = [], []
    for a, b in COMPARISONS:
        va, vb = per_seed(df, a, seeds, METRICS), per_seed(df, b, seeds, METRICS)
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
                "mean_a": va[metric].mean(), "mean_b": vb[metric].mean(),
                "mean": diffs.mean(), "std": diffs.std(ddof=1),
                "n_positive": int((diffs > 0).sum()), "n_seeds": len(diffs),
                "boot_ci_low": lo, "boot_ci_high": hi,
                "sign_test_p": sign_test(diffs),
            })

    long_df, summary_df = pd.DataFrame(long_rows), pd.DataFrame(summary_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    long_df.to_csv(os.path.join(OUT_DIR, "paired_static_control_by_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "paired_static_control.csv"), index=False)

    pd.set_option("display.width", 220)
    print(f"Source: {SOURCE}  seeds: {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}; exact two-sided sign test")
    print("max_drawdown_true is a positive magnitude: positive difference = A has larger drawdown\n")
    wide = long_df.pivot_table(index=["comparison", "metric"], columns="train_seed",
                               values="difference", sort=False)
    wide.columns = [f"seed{c}" for c in wide.columns]
    print("Per-seed differences (A minus B)")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nSummary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
