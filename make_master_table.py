"""One master table of test-period results, built only from existing logs.

No Gemini calls, no new episodes, no existing file is modified.

Rows (in this order) and sources:
  results/logs/dd_test        buy_and_hold, equal_weight_daily_rebalanced,
                              MARL-no-coordinator, price-agent-only,
                              sentiment-agent-only, price-only-hold20,
                              sentiment-only-hold20, MARL-average-hold20,
                              MARL-average-hold20-cash3/-cash5.5/-cash8
  results/logs/static_control price-static-hold20, sentiment-static-hold20
  results/logs/replay_final3  MARL+LLM (coordinator interval N=20, replayed
                              test-period runs, replay_llm.py)

Columns: number of seeds, then mean and std over seeds (pandas default,
ddof=1, as evaluate_agents.summarise) of total_return, sharpe_ratio,
max_drawdown_true (peak-to-trough), turnover and total_cost. The std of a
single deterministic run (n=1) is left empty. The old max_drawdown column
(a range ratio, not a drawdown) is deliberately not used.

Checks printed (the script stops if any fails):
  1. every strategy present in both dd_test and static_control has
     identical per-seed rows in both (all five metrics, exact equality);
  2. every non-LLM master row equals the corresponding row of its source
     folder's test_period_summary.csv exactly (n, means and stds);
  3. MARL+LLM: replay_checks.csv in replay_final3 is all True, and the
     master row equals the N=20 MARL+LLM row of
     report/tables/llm_across_intervals_truedd.csv (exact where possible;
     max absolute difference reported).

Writes:
  report/tables/master_table.csv
  report/tables/master_table.md   (Markdown table + LaTeX tabular)
"""
import json
import os

import numpy as np
import pandas as pd

DD_TEST = "results/logs/dd_test"
STATIC = "results/logs/static_control"
REPLAY = "results/logs/replay_final3"
ACROSS = "report/tables/llm_across_intervals_truedd.csv"
OUT_DIR = "report/tables"
LLM = "MARL+LLM"

ROWS = [
    ("buy_and_hold", DD_TEST),
    ("equal_weight_daily_rebalanced", DD_TEST),
    ("MARL-no-coordinator", DD_TEST),
    ("price-agent-only", DD_TEST),
    ("sentiment-agent-only", DD_TEST),
    ("price-only-hold20", DD_TEST),
    ("sentiment-only-hold20", DD_TEST),
    ("MARL-average-hold20", DD_TEST),
    ("MARL-average-hold20-cash3", DD_TEST),
    ("MARL-average-hold20-cash5.5", DD_TEST),
    ("MARL-average-hold20-cash8", DD_TEST),
    ("price-static-hold20", STATIC),
    ("sentiment-static-hold20", STATIC),
    (LLM, REPLAY),
]
METRICS = ["total_return", "sharpe_ratio", "max_drawdown_true", "turnover", "total_cost"]
# master column stem -> test_period_summary.csv column stem
SUMMARY_STEM = {"total_return": "return", "sharpe_ratio": "sharpe",
                "max_drawdown_true": "drawdown_true", "turnover": "turnover",
                "total_cost": "total_cost"}


def read(path):
    # round_trip: pandas' default fast float parser can be off by one ulp,
    # which would break the exact comparisons below
    return pd.read_csv(path, float_precision="round_trip")


def results(folder):
    return read(os.path.join(folder, "test_period_results.csv"))


def master_row(df, strategy, folder):
    sub = df[df["strategy"] == strategy]
    assert len(sub) > 0, f"{strategy} not in {folder}"
    row = {"strategy": strategy, "source": folder, "n_seeds": len(sub)}
    for m in METRICS:
        row[f"{m}_mean"] = sub[m].mean()
        row[f"{m}_std"] = sub[m].std() if len(sub) > 1 else np.nan
    return row


def check_shared(dd, st):
    print("Check 1: shared strategies, dd_test vs static_control, per-seed rows")
    shared = [s for s in dd["strategy"].unique() if s in set(st["strategy"])]
    ok_all = True
    for s in shared:
        a = dd[dd["strategy"] == s].set_index("train_seed", drop=False).sort_index()
        b = st[st["strategy"] == s].set_index("train_seed", drop=False).sort_index()
        same_seeds = a["train_seed"].fillna(-1).tolist() == b["train_seed"].fillna(-1).tolist()
        ok = same_seeds and all((a[m].to_numpy() == b[m].to_numpy()).all() for m in METRICS)
        ok_all &= ok
        print(f"  {s:32s} n={len(a)}  {'identical' if ok else 'MISMATCH'}")
    return ok_all


def check_summary(table):
    print("Check 2: master rows vs source test_period_summary.csv (exact)")
    ok_all = True
    summaries = {f: read(os.path.join(f, "test_period_summary.csv")).set_index("strategy")
                 for f in (DD_TEST, STATIC)}
    for _, r in table[table["strategy"] != LLM].iterrows():
        s = summaries[r["source"]].loc[r["strategy"]]
        ok = int(s["n"]) == r["n_seeds"]
        for m, stem in SUMMARY_STEM.items():
            ok &= float(s[f"mean_{stem}"]) == r[f"{m}_mean"]
            if r["n_seeds"] > 1:
                ok &= float(s[f"std_{stem}"]) == r[f"{m}_std"]
            else:
                ok &= s[f"std_{stem}"] == "-"
        ok_all &= ok
        print(f"  {r['strategy']:32s} ({os.path.basename(r['source'])})  {'exact' if ok else 'MISMATCH'}")
    return ok_all


def check_llm(table):
    print("Check 3: MARL+LLM (replay_final3)")
    checks = read(os.path.join(REPLAY, "replay_checks.csv"))
    bool_cols = [c for c in checks.columns if c.endswith("_exact") or c == "call_steps_ok"]
    replay_ok = bool(checks[bool_cols].astype(bool).all().all())
    print(f"  replay_checks.csv: {len(checks)} seeds, all {len(bool_cols)} exactness flags True: {replay_ok}")
    with open(os.path.join(DD_TEST, "run_metadata.json")) as f:
        n = json.load(f)["call_frequency"]
    print(f"  coordinator interval N={n} (dd_test run_metadata.json, the paired N=20 base)")
    across = read(ACROSS)
    ref = across[(across["N"] == 20) & (across["strategy"] == LLM)].iloc[0]
    r = table[table["strategy"] == LLM].iloc[0]
    ok = int(ref["n_seeds"]) == r["n_seeds"]
    exact, max_diff = True, 0.0
    for m in ("total_return", "sharpe_ratio", "max_drawdown_true", "total_cost"):
        for stat in ("mean", "std"):
            a, b = float(ref[f"{m}_{stat}"]), r[f"{m}_{stat}"]
            exact &= a == b
            max_diff = max(max_diff, abs(a - b) / max(abs(a), 1e-300))
    ok &= max_diff < 1e-12
    print(f"  vs {ACROSS} (N=20): n_seeds equal, "
          f"{'exact' if exact else f'equal to float rounding (max rel. diff = {max_diff:.1e})'}"
          f"{'' if ok else '  MISMATCH'}")
    return replay_ok and ok


def fmt(mean, std, kind):
    if kind == "pct":
        m, s, d = 100 * mean, 100 * std, 1
    elif kind == "money":
        m, s, d = mean, std, 0
    else:
        m, s, d = mean, std, 2
    if np.isnan(s):
        return f"{m:,.{d}f}"
    return f"{m:,.{d}f} ± {s:,.{d}f}"


DISPLAY = [("total_return", "Total return (%)", "pct"),
           ("sharpe_ratio", "Sharpe", "num"),
           ("max_drawdown_true", "Max drawdown (%)", "pct"),
           ("turnover", "Turnover", "num"),
           ("total_cost", "Total cost ($)", "money")]


def write_markdown(table, path):
    head = ["Strategy", "n"] + [d[1] for d in DISPLAY]
    rows = [[r["strategy"], str(r["n_seeds"])]
            + [fmt(r[f"{m}_mean"], r[f"{m}_std"], k) for m, _, k in DISPLAY]
            for _, r in table.iterrows()]
    md = ["# Master table: test period 2022-07-01 to 2023-12-15", "",
          "Mean ± std over training seeds (std with ddof=1; none for single "
          "deterministic runs). Max drawdown is the true peak-to-trough "
          "drawdown (max_drawdown_true). MARL+LLM: replayed runs in "
          "results/logs/replay_final3, coordinator interval N=20. Built by "
          "make_master_table.py.", "",
          "| " + " | ".join(head) + " |",
          "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]

    def tex(s):
        return s.replace("±", r"$\pm$").replace("%", r"\%").replace("$)", r"\$)").replace("_", r"\_")

    md += ["", "## LaTeX", "", "```latex",
           r"\begin{tabular}{l r " + "r " * len(DISPLAY) + "}", r"\toprule",
           " & ".join(tex(h) for h in head) + r" \\", r"\midrule"]
    for r in rows:
        md.append(" & ".join(tex(c) for c in r) + r" \\")
    md += [r"\bottomrule", r"\end{tabular}", "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    dd, st, rp = results(DD_TEST), results(STATIC), results(REPLAY)
    frames = {DD_TEST: dd, STATIC: st, REPLAY: rp}
    table = pd.DataFrame([master_row(frames[f], s, f) for s, f in ROWS])
    assert "max_drawdown_mean" not in table.columns

    ok = check_shared(dd, st)
    ok &= check_summary(table)
    ok &= check_llm(table)
    print(f"\nAll checks passed: {ok}")
    if not ok:
        raise SystemExit("check failed; nothing written")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "master_table.csv")
    md_path = os.path.join(OUT_DIR, "master_table.md")
    table.to_csv(csv_path, index=False)
    write_markdown(table, md_path)

    show = table[["strategy", "n_seeds"]].copy()
    for m, label, k in DISPLAY:
        show[label] = [fmt(a, b, k) for a, b in zip(table[f"{m}_mean"], table[f"{m}_std"])]
    print("\n" + show.to_string(index=False).replace("±", "+/-"))
    print(f"\nSaved -> {csv_path}\nSaved -> {md_path}")


if __name__ == "__main__":
    main()
