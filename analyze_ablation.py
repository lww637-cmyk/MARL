"""Input-ablation test of the single-agent strategies, test period (no Gemini).

Strategies (same environment, transaction cost and hold schedule as
evaluate_agents.py, run through evaluate_agents.run_episode on the fixed
full test-period episode):
  price-only-hold20, sentiment-only-hold20   target recomputed every
                                             hold_interval steps (20, from
                                             results/logs/dd_test/run_metadata.json)
  price-agent-only, sentiment-agent-only     the daily versions

Interventions, applied to the observation just before agent.predict:
  none               unmodified (must reproduce the logged results exactly)
  shuffle_sentiment  sentiment agent: the 20 sentiment features (4 per ticker)
                     replaced by those of another day of the test episode,
                     whole days permuted jointly (one permutation of the
                     episode's days per draw, numpy default_rng(draw),
                     draws 1..20)
  shuffle_price      price agent: the 50 scaled price indicator features,
                     same 20 draws
  norm_const         both agents: portfolio_norm (last observation entry,
                     portfolio value / initial capital) held at 1.0
                     (deterministic, one run)
The portfolio weights in the observation are left as the env computes them.
shuffle_sentiment is run on the sentiment strategies, shuffle_price on the
price strategies (on the other agent's strategies it is a no-op), norm_const
on all four. The per-day feature blocks are recorded from the unmodified run
(they depend on the date only; checked identical across seeds).

Reported per seed x strategy x intervention: total return, Sharpe and true
max drawdown, mean and std over draws, and the change versus the unmodified
run (mean and std over draws). Nothing is selected.

Writes report/tables/ablation_inputs.csv (summary) and
report/tables/ablation_inputs_draws.csv (every run).
"""
import json
import os

import numpy as np
import pandas as pd

from evaluate_agents import (agent_strategies, load_agents, load_config,
                             make_test_env, run_episode, seed_dirs)
from src.environment.stock_env import N_PRICE_FEATURES, N_SENTIMENT_FEATURES, N_TICKERS

LOGS = "results/logs/dd_test"
OUT_DIR = "report/tables"
STRATEGIES = ["price-only-hold20", "sentiment-only-hold20",
              "price-agent-only", "sentiment-agent-only"]
DRAWS = list(range(1, 21))
METRICS = {"total_return": "return", "sharpe_ratio": "sharpe",
           "max_drawdown_true": "drawdown_true"}
N_FEAT = {"price": N_PRICE_FEATURES * N_TICKERS,
          "sentiment": N_SENTIMENT_FEATURES * N_TICKERS}


class InterventionAgent:
    """
    Wraps an agent. Counts predict() calls (one per step in run_episode) and
    edits the observation: features[:n] <- blocks[perm[step]] if perm is
    given, last entry <- 1.0 if norm_const. Records the original features.
    """
    def __init__(self, agent, n_feat, blocks=None, perm=None, norm_const=False):
        self.agent, self.n_feat = agent, n_feat
        self.blocks, self.perm, self.norm_const = blocks, perm, norm_const
        self.step, self.seen = 0, []

    def predict(self, obs):
        obs = np.array(obs, dtype=np.float32)
        self.seen.append(obs[:self.n_feat].copy())
        if self.perm is not None:
            obs[:self.n_feat] = self.blocks[self.perm[self.step]]
        if self.norm_const:
            obs[-1] = 1.0
        self.step += 1
        return self.agent.predict(obs)


def run(env, agents, mode, hold, blocks=None, shuffle=None, perm=None, norm_const=False):
    wrapped = {}
    for name, agent in agents.items():
        wrapped[name] = InterventionAgent(
            agent, N_FEAT[name], blocks[name] if blocks else None,
            perm if shuffle == name else None, norm_const)
    s = run_episode(env, wrapped["price"], wrapped["sentiment"],
                    agent_mode=mode, hold_every=hold)
    for w in wrapped.values():
        assert w.step == s["n_steps"], (w.step, s["n_steps"])
    return s, {k: np.array(w.seen) for k, w in wrapped.items()}


def main():
    config = load_config()
    with open(os.path.join(LOGS, "run_metadata.json")) as f:
        meta = json.load(f)
    assert meta["period"] == "test" and not meta["use_llm"]
    hold_interval, seeds = meta["hold_interval"], meta["train_seeds"]
    strategies = {k: v for k, v in agent_strategies(False, hold_interval).items()
                  if k in STRATEGIES}
    assert list(strategies) == [s for s in agent_strategies(False, hold_interval) if s in STRATEGIES]

    logged = pd.read_csv(os.path.join(LOGS, "test_period_results.csv"),
                         float_precision="round_trip")
    curves = pd.read_csv(os.path.join(LOGS, "equity_curves.csv"),
                         float_precision="round_trip")
    env = make_test_env(config, full_period=True)
    print(f"Test episode, hold interval {hold_interval}, seeds {seeds}, "
          f"cost rate {env.transaction_cost}")
    assert env.transaction_cost == meta["cost_rate"]

    rows, ref_blocks = [], None
    for seed, ckpt in seed_dirs(config, seeds).items():
        price_agent, sentiment_agent = load_agents(env, config, ckpt)
        agents = {"price": price_agent, "sentiment": sentiment_agent}
        blocks = None
        for label in STRATEGIES:
            mode, _, hold, tilt = strategies[label]
            assert tilt is None

            # unmodified run, checked against the logged results
            base, seen = run(env, agents, mode, hold)
            lg = logged[(logged.strategy == label) & (logged.train_seed == seed)]
            assert len(lg) == 1
            for m in ["total_return", "sharpe_ratio", "max_drawdown", "max_drawdown_true",
                      "final_value", "turnover", "total_cost"]:
                if base[m] != lg[m].iloc[0]:
                    raise RuntimeError(f"seed {seed} {label}: {m} {base[m]!r} != logged {lg[m].iloc[0]!r}")
            cv = curves[(curves.strategy == label) & (curves.train_seed == seed)].sort_values("step")
            if not np.array_equal(base["equity_curve"]["portfolio_value"].to_numpy(),
                                  cv["portfolio_value"].to_numpy()):
                raise RuntimeError(f"seed {seed} {label}: equity curve differs from log")
            if blocks is None:
                blocks = seen
                if ref_blocks is None:
                    ref_blocks = blocks
                for k in blocks:
                    assert np.array_equal(blocks[k], ref_blocks[k]), "feature blocks differ"
            n_days = len(blocks["price"])
            print(f"seed {seed} {label:22s} unmodified reproduces log exactly "
                  f"(return {base['total_return']:+.6f})")
            rows.append({"train_seed": seed, "strategy": label, "intervention": "none",
                         "draw": None, **{m: base[m] for m in METRICS}})

            agent = mode   # "price" or "sentiment"
            plan = [("norm_const", None)] + [(f"shuffle_{agent}", d) for d in DRAWS]
            for intervention, draw in plan:
                if draw is None:
                    s, _ = run(env, agents, mode, hold, norm_const=True)
                else:
                    perm = np.random.default_rng(draw).permutation(n_days)
                    s, _ = run(env, agents, mode, hold, blocks, shuffle=agent, perm=perm)
                rows.append({"train_seed": seed, "strategy": label,
                             "intervention": intervention, "draw": draw,
                             **{m: s[m] for m in METRICS}})

    draws = pd.DataFrame(rows)
    draws["draw"] = draws["draw"].astype("Int64")
    base = draws[draws.intervention == "none"].set_index(["train_seed", "strategy"])
    for m in METRICS:
        draws[f"delta_{m}"] = draws[m] - base.loc[
            list(zip(draws.train_seed, draws.strategy)), m].to_numpy()
    os.makedirs(OUT_DIR, exist_ok=True)
    draws.to_csv(os.path.join(OUT_DIR, "ablation_inputs_draws.csv"), index=False)

    out = []
    for (seed, label, iv), g in draws.groupby(["train_seed", "strategy", "intervention"],
                                              sort=False):
        r = {"train_seed": seed, "strategy": label, "intervention": iv, "n_runs": len(g)}
        for m, short in METRICS.items():
            r[f"{short}_mean"] = g[m].mean()
            r[f"{short}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
            r[f"{short}_unmodified"] = base.loc[(seed, label), m]
            r[f"delta_{short}_mean"] = g[f"delta_{m}"].mean()
            r[f"delta_{short}_std"] = g[f"delta_{m}"].std(ddof=1) if len(g) > 1 else np.nan
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(OUT_DIR, "ablation_inputs.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = out[out.intervention != "none"]
    for short in METRICS.values():
        print(f"\n{short}: unmodified, intervened mean (std over draws), change mean (std)")
        t = show.assign(**{
            "unmodified": show[f"{short}_unmodified"].map("{:+.4f}".format),
            "intervened": [f"{m:+.4f} ({s:.4f})" if pd.notna(s) else f"{m:+.4f}"
                           for m, s in zip(show[f"{short}_mean"], show[f"{short}_std"])],
            "change": [f"{m:+.4f} ({s:.4f})" if pd.notna(s) else f"{m:+.4f}"
                       for m, s in zip(show[f"delta_{short}_mean"], show[f"delta_{short}_std"])],
        })
        print(t.pivot_table(index=["strategy", "intervention"], columns="train_seed",
                            values="change", aggfunc="first").to_string())
        print(t[["train_seed", "strategy", "intervention", "unmodified",
                 "intervened", "change"]].to_string(index=False))

    print("\nChange vs unmodified, mean over seeds of per-seed mean change (std over seeds)")
    for short in METRICS.values():
        a = show.groupby(["strategy", "intervention"], sort=False)[f"delta_{short}_mean"].agg(["mean", "std"])
        print(f"\n{short}\n{a.to_string(float_format='{:+.4f}'.format)}")
    print(f"\nWrote {OUT_DIR}/ablation_inputs.csv, ablation_inputs_draws.csv")


if __name__ == "__main__":
    main()
