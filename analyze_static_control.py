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

MARL+LLM vs static (no new Gemini calls): MARL+LLM rows are the replayed
test-period runs in results/logs/replay_final3 (replay_llm.py, the same rows
analyze_paired.py --truedd uses), paired by train seed with
  MARL+LLM minus price-static-hold20
  MARL+LLM minus sentiment-static-hold20
Same statistics; this table draws from its own generator (seed 12345), so the
static-control table above is unchanged by it.

Writes:
  report/tables/paired_static_control.csv          summary per comparison/metric
  report/tables/paired_static_control_by_seed.csv  per-seed differences (long)
  report/tables/paired_llm_vs_static.csv           summary, MARL+LLM vs static
  report/tables/paired_llm_vs_static_by_seed.csv   per-seed differences (long)
"""
import os

import numpy as np
import pandas as pd

from analyze_paired import BOOT_SEED, N_BOOT, OUT_DIR, bootstrap_ci, per_seed, sign_test

SOURCE = "results/logs/static_control"
LLM_SOURCE = "results/logs/replay_final3"
LLM = "MARL+LLM"
METRICS = ["total_return", "sharpe_ratio", "max_drawdown_true"]
COMPARISONS = [
    ("sentiment-only-hold20", "sentiment-static-hold20"),
    ("price-only-hold20", "price-static-hold20"),
    ("sentiment-static-hold20", "price-static-hold20"),
]
LLM_COMPARISONS = [
    (LLM, "price-static-hold20"),
    (LLM, "sentiment-static-hold20"),
]


def compare(df, comparisons, seeds):
    """Per-seed differences and summary rows; one fresh generator per table."""
    rng = np.random.default_rng(BOOT_SEED)
    long_rows, summary_rows = [], []
    for a, b in comparisons:
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
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def load_llm_static(static):
    """replay_final3 MARL+LLM rows plus the static-control rows, checked to share
    the test period and seeds."""
    replay = pd.read_csv(os.path.join(LLM_SOURCE, "test_period_results.csv"))
    assert set(replay["strategy"]) == {LLM}, f"{LLM_SOURCE} has non-{LLM} rows"
    assert LLM not in set(static["strategy"]), f"{SOURCE} has {LLM} rows"
    for col in ("start_date", "end_date", "n_steps"):
        assert set(replay[col]) == set(static[col]), f"{col} differs between sources"
    return pd.concat([static, replay], ignore_index=True)


def report(title, source, long_df, summary_df, seeds):
    print(f"=== {title} ===")
    print(f"Source: {source}  seeds: {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}; exact two-sided sign test")
    print("max_drawdown_true is a positive magnitude: positive difference = A has larger drawdown\n")
    wide = long_df.pivot_table(index=["comparison", "metric"], columns="train_seed",
                               values="difference", sort=False)
    wide.columns = [f"seed{c}" for c in wide.columns]
    print("Per-seed differences (A minus B)")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nSummary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


def main():
    df = pd.read_csv(os.path.join(SOURCE, "test_period_results.csv"))
    seeds = sorted(int(s) for s in df["train_seed"].dropna().unique())
    pd.set_option("display.width", 220)
    os.makedirs(OUT_DIR, exist_ok=True)

    long_df, summary_df = compare(df, COMPARISONS, seeds)
    long_df.to_csv(os.path.join(OUT_DIR, "paired_static_control_by_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "paired_static_control.csv"), index=False)
    report("Hold controls vs static", SOURCE, long_df, summary_df, seeds)

    llm_df = load_llm_static(df)
    llm_seeds = sorted(int(s) for s in llm_df.loc[llm_df["strategy"] == LLM, "train_seed"])
    assert llm_seeds == seeds, f"MARL+LLM seeds {llm_seeds} != static seeds {seeds}"
    long_df, summary_df = compare(llm_df, LLM_COMPARISONS, seeds)
    long_df.to_csv(os.path.join(OUT_DIR, "paired_llm_vs_static_by_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "paired_llm_vs_static.csv"), index=False)
    report("MARL+LLM vs static", f"{LLM_SOURCE} + {SOURCE}", long_df, summary_df, seeds)


if __name__ == "__main__":
    main()
