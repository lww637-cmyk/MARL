"""Stability of SHAP group importance across three independent draws.

Draw 0 = the published run (SHAP seed 12345, report/tables/shap_by_*.csv);
draws 1 and 2 = compute_shap.py --seeds 0..5 --shap-seed {1,2}
(results/logs/shap_multiseed_draw{1,2}/shap_by_*.csv). Each draw redraws the
explained test steps, the training-split background steps and the
KernelExplainer sampling; the checkpoints are the same.

Per train seed x agent x group, two measures:
  group_abs_sum      mean |sum of the group's SHAP values| (shap_by_group.csv)
  share              sum of per-feature mean |SHAP| over the group, as a
                     share of the agent's total (analyze_shap_groups.py)
and over the three draws: each draw's value, mean, std (ddof=1), min, max,
range and range / mean. Also the spread of the over-seed mean across draws,
and per seed x agent the Spearman rank correlation of per-feature importance
between draws.

Writes report/tables/shap_draw_stability.csv and
report/tables/shap_draw_feature_rankcorr.csv.
"""
import itertools
import os

import pandas as pd

from analyze_shap_groups import group_shares

DRAWS = {0: "report/tables",
         1: "results/logs/shap_multiseed_draw1",
         2: "results/logs/shap_multiseed_draw2"}
OUT_DIR = "report/tables"
KEY = ["seed", "agent", "group"]


def load(draw, folder):
    g = pd.read_csv(os.path.join(folder, "shap_by_group.csv")).rename(
        columns={"feature": "group", "value": "group_abs_sum"})
    s = group_shares(os.path.join(folder, "shap_by_feature.csv"))[KEY + ["share"]]
    f = pd.read_csv(os.path.join(folder, "shap_by_feature.csv"))
    return g.merge(s, on=KEY, how="outer").assign(draw=draw), f.assign(draw=draw)


def main():
    parts = [load(d, p) for d, p in DRAWS.items()]
    groups = pd.concat([p[0] for p in parts], ignore_index=True)
    feats = pd.concat([p[1] for p in parts], ignore_index=True)

    rows = []
    for measure in ["group_abs_sum", "share"]:
        w = groups.pivot_table(index=KEY, columns="draw", values=measure)
        w = w.dropna()  # group_abs_sum has no row for groups an agent lacks
        st = pd.DataFrame({f"draw{d}": w[d] for d in DRAWS})
        st["mean"] = w.mean(axis=1)
        st["std"] = w.std(axis=1, ddof=1)
        st["min"] = w.min(axis=1)
        st["max"] = w.max(axis=1)
        st["range"] = st["max"] - st["min"]
        st["range_over_mean"] = st["range"] / st["mean"]
        rows.append(st.reset_index().assign(measure=measure))
    out = pd.concat(rows, ignore_index=True)
    out = out[["measure", *KEY, *[f"draw{d}" for d in DRAWS], "mean", "std",
               "min", "max", "range", "range_over_mean"]]
    out.to_csv(os.path.join(OUT_DIR, "shap_draw_stability.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print("Per seed: value in each draw and spread across the three draws")
    print(out.to_string(index=False, float_format="{:.4f}".format))

    print("\nMean over the six seeds, per draw, and spread of that mean across draws")
    for measure in ["group_abs_sum", "share"]:
        m = groups.dropna(subset=[measure]).groupby(["agent", "group", "draw"])[measure].mean().unstack("draw")
        m["mean"] = m.mean(axis=1)
        m["std"] = m[list(DRAWS)].std(axis=1, ddof=1)
        m["range"] = m[list(DRAWS)].max(axis=1) - m[list(DRAWS)].min(axis=1)
        print(f"\n{measure}\n{m.to_string(float_format='{:.4f}'.format)}")

    print("\nMean over seeds of the across-draw std, vs the across-seed std (draw 0)")
    for measure in ["group_abs_sum", "share"]:
        across_draw = out[out.measure == measure].groupby(["agent", "group"])["std"].mean()
        across_seed = (groups[groups.draw == 0].dropna(subset=[measure])
                       .groupby(["agent", "group"])[measure].std(ddof=1))
        print(f"\n{measure}")
        print(pd.DataFrame({"mean across-draw std": across_draw,
                            "across-seed std (draw 0)": across_seed}).to_string(float_format="{:.4f}".format))

    rc = []
    for (seed, agent), g in feats.groupby(["seed", "agent"]):
        w = g.pivot(index="feature", columns="draw", values="value")
        for a, b in itertools.combinations(DRAWS, 2):
            rc.append({"seed": seed, "agent": agent, "draw_a": a, "draw_b": b,
                       "spearman": w[a].corr(w[b], method="spearman")})
    rc = pd.DataFrame(rc)
    rc.to_csv(os.path.join(OUT_DIR, "shap_draw_feature_rankcorr.csv"), index=False)
    print("\nSpearman rank correlation of per-feature importance between draws")
    print(rc.pivot_table(index=["agent", "seed"], columns=["draw_a", "draw_b"],
                         values="spearman").to_string(float_format="{:.3f}".format))
    print(f"\nWrote {OUT_DIR}/shap_draw_stability.csv, shap_draw_feature_rankcorr.csv")


if __name__ == "__main__":
    main()
