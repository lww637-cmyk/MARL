"""Stress-episode analysis: the benchmark's deepest drawdown, test period.

Reads logged daily equity curves only (no Gemini, no environment runs):
  results/logs/dd_test/equity_curves.csv               11 strategies, N=20 controls
  results/logs/replay_final3/equity_curves.csv         MARL+LLM, N=20 (headline)
  results/logs/replay_final3_repeat/equity_curves.csv  MARL+LLM, N=20 repeat
  results/logs/replay_llm_cf10/equity_curves.csv       MARL+LLM, N=10
  results/logs/replay_llm_cf40/equity_curves.csv       MARL+LLM, N=40

Episode (pre-specified, fixed from equal_weight_daily_rebalanced ONLY):
  trough   = date of its maximum true drawdown over the test period,
  peak     = last date on or before the trough at which it was at its running
             peak (the peak that drawdown is measured from),
  recovery = first date after the trough with value >= the peak value, or the
             last test date if that never happens.
  fall phase = peak -> trough, recovery phase = trough -> recovery (counted in
  trading days, i.e. steps). No strategy curve enters the choice of dates.

Per strategy and seed, over the SAME dates:
  fall_return       V_trough / V_peak - 1
  recovery_return   V_recovery / V_trough - 1
  episode_return    V_recovery / V_peak - 1
  episode_drawdown_true  max (running peak - V) / running peak on the slice
                    peak..recovery, running peak starting at V_peak
                    (positive magnitude)
  worst_day         min daily return V_t / V_{t-1} - 1 over t in (peak, recovery]
  downside_capture  fall_return / benchmark fall_return

Paired per-seed differences (A minus B, same train seed), A = MARL+LLM
(replay_final3), B = MARL-average-hold20, each cash control and
equal_weight_daily_rebalanced (seed-independent, paired with every seed), for
fall_return, recovery_return and episode_drawdown_true. Same bootstrap as
analyze_paired.py (one rng, seed 12345, 10000 resamples, percentile 95% CI of
the mean, drawn in comparison x metric order) and exact two-sided sign test.
episode_drawdown_true is a positive magnitude: positive difference = A fell
deeper. All comparisons are reported; none are selected.

Writes (existing results are not touched):
  report/tables/stress_episode.csv         one row per strategy x seed
  report/tables/stress_episode_paired.csv  per-seed differences + summary
  report/tables/stress_episode_meta.json   the three dates and day counts
"""
import json
import os

import numpy as np
import pandas as pd

from analyze_paired import N_BOOT, BOOT_SEED, bootstrap_ci, sign_test

LOGS = "results/logs"
OUT_DIR = "report/tables"
EQUAL_WEIGHT = "equal_weight_daily_rebalanced"
LLM = "MARL+LLM"

# folder -> (label suffix for MARL+LLM rows, coordinator interval N)
SOURCES = {
    "dd_test": (None, 20),
    "replay_final3": ("", 20),
    "replay_final3_repeat": ("-repeat", 20),
    "replay_llm_cf10": ("-N10", 10),
    "replay_llm_cf40": ("-N40", 40),
}

CASH = ["MARL-average-hold20-cash3", "MARL-average-hold20-cash5.5",
        "MARL-average-hold20-cash8"]
COMPARE_TO = ["MARL-average-hold20", *CASH, EQUAL_WEIGHT]
PAIRED_METRICS = ["fall_return", "recovery_return", "episode_drawdown_true"]


def load_curve(folder):
    df = pd.read_csv(os.path.join(LOGS, folder, "equity_curves.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def episode_dates(v, dates):
    """Peak, trough, recovery indices of the deepest drawdown of v."""
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / peak
    trough = int(np.argmax(dd))
    assert dd[trough] > 0, "benchmark has no drawdown"
    p = int(np.where(v[:trough + 1] == peak[trough])[0][-1])
    after = np.where(v[trough + 1:] >= v[p])[0]
    recovered = len(after) > 0
    rec = trough + 1 + int(after[0]) if recovered else len(v) - 1
    return p, trough, rec, recovered, float(dd[trough])


def episode_stats(v, p, t, r):
    seg = v[p:r + 1]
    run_peak = np.maximum.accumulate(seg)
    daily = seg[1:] / seg[:-1] - 1
    return {
        "fall_return": v[t] / v[p] - 1,
        "recovery_return": v[r] / v[t] - 1,
        "episode_return": v[r] / v[p] - 1,
        "episode_drawdown_true": float(((run_peak - seg) / run_peak).max()),
        "worst_day": float(daily.min()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = {f: load_curve(f) for f in SOURCES}

    # ── episode from the benchmark only ────────────────────────────────────
    ew = curves["dd_test"]
    ew = ew[ew["strategy"] == EQUAL_WEIGHT].sort_values("step")
    ew_v = ew["portfolio_value"].to_numpy(dtype=np.float64)
    calendar = ew["date"].reset_index(drop=True)
    p, t, r, recovered, ew_dd = episode_dates(ew_v, calendar)
    d = lambda i: str(calendar.iloc[i].date())
    meta = {
        "benchmark": EQUAL_WEIGHT,
        "source": "results/logs/dd_test/equity_curves.csv",
        "peak_date": d(p), "trough_date": d(t), "recovery_date": d(r),
        "recovered": recovered,
        "fall_days": t - p, "recovery_days": r - t, "episode_days": r - p,
        "benchmark_max_drawdown_true": ew_dd,
        "benchmark_peak_value": float(ew_v[p]),
        "benchmark_trough_value": float(ew_v[t]),
        "benchmark_recovery_value": float(ew_v[r]),
        "note": ("Dates fixed from the equal-weight daily rebalanced benchmark "
                 "only; day counts are trading days (steps). If not recovered, "
                 "recovery_date is the last test date."),
    }
    with open(os.path.join(OUT_DIR, "stress_episode_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Episode from {EQUAL_WEIGHT} (max true drawdown {ew_dd:.4f})")
    print(f"  peak     {meta['peak_date']}")
    print(f"  trough   {meta['trough_date']}")
    print(f"  recovery {meta['recovery_date']}"
          + ("" if recovered else "  (never regained the peak: last test date)"))
    print(f"  fall phase {t - p} trading days, recovery phase {r - t} trading days\n")

    # ── per strategy / seed ────────────────────────────────────────────────
    rows = []
    for folder, (suffix, n) in SOURCES.items():
        df = curves[folder]
        if suffix is not None:
            assert set(df["strategy"]) == {LLM}, folder
            df = df.assign(strategy=LLM + suffix)
        else:
            assert LLM not in set(df["strategy"]), folder
        for (strategy, seed), g in df.groupby(["strategy", "train_seed"],
                                              dropna=False, sort=False):
            g = g.sort_values("step")
            assert g["date"].reset_index(drop=True).equals(calendar), (folder, strategy, seed)
            v = g["portfolio_value"].to_numpy(dtype=np.float64)
            rows.append({"strategy": strategy,
                         "train_seed": None if pd.isna(seed) else int(seed),
                         "interval_N": n, "source": folder,
                         **episode_stats(v, p, t, r)})
    res = pd.DataFrame(rows)
    res["train_seed"] = res["train_seed"].astype("Int64")
    bench_fall = res.loc[res["strategy"] == EQUAL_WEIGHT, "fall_return"]
    assert len(bench_fall) == 1
    res["downside_capture"] = res["fall_return"] / bench_fall.iloc[0]
    res.to_csv(os.path.join(OUT_DIR, "stress_episode.csv"), index=False)

    pd.set_option("display.width", 220)
    cols = ["fall_return", "recovery_return", "episode_return",
            "episode_drawdown_true", "worst_day", "downside_capture"]
    print("Per strategy: mean (std over seeds)")
    agg = res.groupby("strategy", sort=False)[cols].agg(["mean", "std"])
    out = pd.DataFrame({c: [f"{m:+.4f} ({s:.4f})" if pd.notna(s) else f"{m:+.4f}"
                            for m, s in zip(agg[(c, "mean")], agg[(c, "std")])]
                        for c in cols}, index=agg.index)
    print(out.to_string(), "\n")

    # ── paired differences ─────────────────────────────────────────────────
    seeds = sorted(int(s) for s in res["train_seed"].dropna().unique())

    def per_seed(strategy):
        s = res[res["strategy"] == strategy]
        if s["train_seed"].isna().all():
            assert len(s) == 1, strategy
            return pd.DataFrame([s.iloc[0][PAIRED_METRICS]] * len(seeds), index=seeds)
        s = s.set_index(s["train_seed"].astype(int))[PAIRED_METRICS]
        assert s.index.is_unique, strategy
        return s.loc[seeds]

    rng = np.random.default_rng(BOOT_SEED)
    va = per_seed(LLM)
    paired = []
    for b in COMPARE_TO:
        vb = per_seed(b)
        label = f"{LLM} minus {b}"
        for metric in PAIRED_METRICS:
            diffs = (va[metric].astype(float) - vb[metric].astype(float)).to_numpy()
            lo, hi = bootstrap_ci(diffs, rng)
            row = {"comparison": label, "metric": metric,
                   "mean": diffs.mean(), "std": diffs.std(ddof=1),
                   "n_positive": int((diffs > 0).sum()), "n_seeds": len(diffs),
                   "boot_ci_low": lo, "boot_ci_high": hi,
                   "sign_test_p": sign_test(diffs)}
            row.update({f"seed{s}": x for s, x in zip(seeds, diffs)})
            paired.append(row)
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(OUT_DIR, "stress_episode_paired.csv"), index=False)

    print(f"Paired differences ({LLM} replay_final3 minus B), seeds {seeds}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}; exact two-sided sign test")
    print("episode_drawdown_true is a positive magnitude: positive = A fell deeper")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nWrote {OUT_DIR}/stress_episode.csv, stress_episode_paired.csv, "
          "stress_episode_meta.json")


if __name__ == "__main__":
    main()
