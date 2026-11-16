"""SHAP group shares from the per-feature table (no SHAP rerun).

Reads report/tables/shap_by_feature.csv (mean |SHAP| per feature, averaged
over explained observations and the six output weights; seed x agent).
For each seed and agent:
  group_importance = sum over the group's features of per-feature mean |SHAP|
  share            = group_importance / sum over all of the agent's features
  n_features       = number of features in the group
Groups as in compute_shap.feature_group (price_indicators, sentiment_features,
portfolio_cash_state).

Note: this differs from report/tables/shap_by_group.csv, which sums the
signed SHAP values over a group's features before taking |.| (so offsetting
features cancel there and not here).

Writes report/tables/shap_group_shares.csv. Usable on any folder holding a
shap_by_feature.csv via group_shares(path).
"""
import os

import pandas as pd

from compute_shap import feature_group

TABLES_DIR = "report/tables"


def group_shares(path: str) -> pd.DataFrame:
    feat = pd.read_csv(path)
    feat["group"] = feat["feature"].map(feature_group)
    g = (feat.groupby(["seed", "agent", "group"])["value"]
             .agg(group_importance="sum", n_features="count").reset_index())
    g["agent_total"] = g.groupby(["seed", "agent"])["group_importance"].transform("sum")
    g["share"] = g["group_importance"] / g["agent_total"]
    return g[["seed", "agent", "group", "n_features", "group_importance",
              "agent_total", "share"]]


def main():
    out = group_shares(os.path.join(TABLES_DIR, "shap_by_feature.csv"))
    out.to_csv(os.path.join(TABLES_DIR, "shap_group_shares.csv"), index=False)

    pd.set_option("display.width", 200)
    for col, fmt in [("group_importance", "{:.5f}"), ("share", "{:.4f}")]:
        wide = out.pivot_table(index=["agent", "group"], columns="seed", values=col)
        wide["mean"] = wide.mean(axis=1)
        wide["std"] = wide.drop(columns="mean").std(axis=1, ddof=1)
        n = out.groupby(["agent", "group"])["n_features"].first()
        wide.insert(0, "n_features", n)
        print(f"\n{col} (sum of per-feature mean |SHAP|), per seed, mean and std over seeds")
        print(wide.to_string(float_format=fmt.format))
    print(f"\nWrote {TABLES_DIR}/shap_group_shares.csv")


if __name__ == "__main__":
    main()
