"""Rebuild MARL+LLM equity curves from a run's coordinator_calls.jsonl, no API calls.

For every train seed of a run folder, the logged final_weights of each
MARL+LLM coordinator call are applied from that call's step and held until
the next call, through evaluate_agents.run_episode on the same full test
period environment and cost rate as the run. That is exactly what the run
did: between calls MARL+LLM trades back to the cached LLM weights.

Checks, per seed (any failure stops the script before anything is written):
  * the logged call steps are 0, N, 2N, ... for the run's call_frequency N
  * the summary numbers match the logged row EXACTLY: total_return,
    sharpe_ratio, max_drawdown, final_value, turnover, total_cost, n_steps
  * at every call step, the prompt rebuilt from the replayed state and the
    seed's real agents (build_prompt, as in the run) equals the logged
    prompt. The call log has no numeric portfolio value field; the prompt
    holds the value to the cent ("Current value: $..."), the total return,
    the recent 5-day return and volatility and both agents' suggestions,
    so this compares the portfolio value at every rebalance step.

Folders: final3, final3_repeat, llm_cf10, llm_cf40 (or --runs ...).

Writes, per folder, to results/logs/replay_{folder}/:
  test_period_results.csv  replayed MARL+LLM rows (same columns as
                           evaluate_agents.py, incl. max_drawdown_true)
  equity_curves.csv        daily portfolio values per seed
  replay_checks.csv        per seed: call count and the checks above
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

# The coordinator module builds a Gemini client on import: give it a dummy
# key (load_dotenv does not override it) and replace the client below.
os.environ["GEMINI_API_KEY"] = "replay-no-api-calls"
import src.coordinator.llm_coordinator as llm
import evaluate_agents as ev
from src.checkpoints import checkpoint_dir


class NoClient:
    """Any attribute access fails, so no Gemini request can be made."""
    def __getattr__(self, name):
        raise RuntimeError("replay_llm.py must not call Gemini")


llm.client = NoClient()

LOG_DIR = "results/logs"
RUNS = ["final3", "final3_repeat", "llm_cf10", "llm_cf40"]
LLM = "MARL+LLM"
EXACT = ["total_return", "sharpe_ratio", "max_drawdown", "final_value",
         "turnover", "total_cost", "n_steps"]


class ReplayAgent:
    """
    Stand-in for the price agent (run_episode agent_mode="price"): returns
    the logged LLM weights of the latest call. At each call step it rebuilds
    the prompt from the env state and the real agents and stores it.
    """
    def __init__(self, env, calls: dict, price_agent, sentiment_agent):
        self.env, self.calls = env, calls
        self.price_agent, self.sentiment_agent = price_agent, sentiment_agent
        self.weights = None
        self.prompts = {}

    def predict(self, obs):
        env, step = self.env, self.env.timestep
        if step in self.calls:
            price_action = self.price_agent.predict(env.observe("price_agent"))
            sent_action = self.sentiment_agent.predict(env.observe("sentiment_agent"))
            context = {
                "portfolio_value": env.portfolio_value,
                "initial_capital": env.initial_capital,
                "timestep"       : env.timestep,
                "recent_returns" : env.returns_history[-10:],
            }
            self.prompts[step] = llm.build_prompt(price_action, sent_action, context)
            self.weights = np.array(self.calls[step]["final_weights"], dtype=np.float32)
        return self.weights


def load_run(folder: str):
    with open(os.path.join(folder, "run_metadata.json")) as f:
        meta = json.load(f)
    # round_trip: parse the logged floats back to the exact values written
    results = pd.read_csv(os.path.join(folder, "test_period_results.csv"),
                          float_precision="round_trip")
    with open(os.path.join(folder, "coordinator_calls.jsonl"), encoding="utf-8") as f:
        calls = pd.DataFrame([json.loads(line) for line in f])
    calls = calls[(calls["phase"] == "test_period") & (calls["strategy"] == LLM)]
    return meta, results, calls


def replay_run(name: str, base_config: dict):
    folder = os.path.join(LOG_DIR, name)
    meta, results, calls = load_run(folder)
    if not meta["use_llm"]:
        raise RuntimeError(f"{folder}: not an LLM run")
    config = copy.deepcopy(base_config)
    config.setdefault("coordinator", {})["call_frequency"] = meta["call_frequency"]
    config["environment"]["transaction_cost"] = meta.get(
        "cost_rate", base_config["environment"]["transaction_cost"])
    n = meta["call_frequency"]

    env = ev.make_test_env(config, full_period=True)
    rows, curves, checks, failures = [], [], [], []
    for seed in meta["train_seeds"]:
        seed_calls = calls[calls["seed"] == seed]
        by_step = {int(r["step"]): r for _, r in seed_calls.iterrows()}
        logged = results[(results["strategy"] == LLM) & (results["train_seed"] == seed)]
        if len(logged) != 1:
            raise RuntimeError(f"{folder}: {len(logged)} MARL+LLM rows for seed {seed}")
        logged = logged.iloc[0]

        price_agent, sentiment_agent = ev.load_agents(env, config, checkpoint_dir(config, seed))
        agent = ReplayAgent(env, by_step, price_agent, sentiment_agent)
        summary = ev.run_episode(env, agent, agent, config=config, agent_mode="price")
        curve = summary.pop("equity_curve")
        summary.update(strategy=LLM, train_seed=seed)

        expected_steps = list(range(0, int(logged["n_steps"]), n))
        steps_ok = sorted(by_step) == expected_steps
        numbers_ok = {m: summary[m] == logged[m] for m in EXACT}
        prompt_bad = [s for s in sorted(by_step)
                      if agent.prompts.get(s) != by_step[s]["prompt"]]
        check = {"run": name, "train_seed": seed, "n_calls": len(by_step),
                 "call_steps_ok": steps_ok,
                 **{f"{m}_exact": ok for m, ok in numbers_ok.items()},
                 "prompts_matching": len(by_step) - len(prompt_bad),
                 "prompt_mismatch_steps": " ".join(map(str, prompt_bad))}
        checks.append(check)
        if not steps_ok or not all(numbers_ok.values()) or prompt_bad:
            failures.append(check)
            for m in EXACT:
                if not numbers_ok[m]:
                    print(f"  MISMATCH {name} seed {seed} {m}: "
                          f"replayed {summary[m]!r} logged {logged[m]!r}")
            for s in prompt_bad[:3]:
                print(f"  PROMPT MISMATCH {name} seed {seed} step {s}")
        rows.append(summary)
        curves.append(curve.assign(strategy=LLM, train_seed=seed))
        print(f"  {name} seed {seed}: calls={len(by_step)} steps_ok={steps_ok} "
              f"exact={all(numbers_ok.values())} prompts={check['prompts_matching']}/{len(by_step)} "
              f"return={summary['total_return']:.6f} max_drawdown_true={summary['max_drawdown_true']:.4f}")
    return (pd.DataFrame(rows)[list(results.columns[:7]) + ["max_drawdown_true"]
                               + list(results.columns[7:])],
            pd.concat(curves, ignore_index=True), pd.DataFrame(checks), failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=RUNS)
    args = parser.parse_args()
    base_config = ev.load_config()

    outputs, all_failures = {}, []
    for name in args.runs:
        print(f"\n=== {name}")
        outputs[name] = replay_run(name, base_config)
        all_failures += outputs[name][3]

    if all_failures:
        print(f"\nSTOP: {len(all_failures)} seed(s) do not match; nothing written.")
        print(pd.DataFrame(all_failures).to_string(index=False))
        sys.exit(1)

    for name, (rows, curves, checks, _) in outputs.items():
        out = os.path.join(LOG_DIR, f"replay_{name}")
        os.makedirs(out, exist_ok=True)
        rows.to_csv(os.path.join(out, "test_period_results.csv"), index=False)
        curves.to_csv(os.path.join(out, "equity_curves.csv"), index=False)
        checks.to_csv(os.path.join(out, "replay_checks.csv"), index=False)
        print(f"Saved → {out}/")
    print("\nAll seeds of all runs replay exactly.")


if __name__ == "__main__":
    main()
