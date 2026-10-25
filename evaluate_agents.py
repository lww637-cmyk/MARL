import time

"""
Evaluation script for trained MARL agents. Test split only (>= 2022-07-01).

Primary evaluation:
    ONE fixed episode covering the full test period, identical for
    every strategy:
      buy_and_hold                  equal weights bought once, then held
                                    (weights drift, no further trades)
      equal_weight_daily_rebalanced equal target weights every day
      price-agent-only              price agent's action alone
      sentiment-agent-only          sentiment agent's action alone
      MARL-no-coordinator           average of both agents
      MARL+LLM                      LLM coordinator every N steps
      MARL-average-hold20           average of both agents, recomputed every
                                    N steps (coordinator schedule), held
                                    in between
      price-only-hold20             price agent, same hold schedule
      sentiment-only-hold20         sentiment agent, same hold schedule
    The agent strategies are run once per
    training seed ({checkpoint root}/seed{N}/) to report variance
    across training seeds. Each seed folder must hold a
    training_setup.json matching the current setup, or it is refused.

N is coordinator.call_frequency in config.yaml (DEFAULT_CALL_FREQUENCY if
missing). It drives both MARL+LLM and the *-hold20 controls; the "hold20"
labels are kept as-is so existing result files stay comparable. The value
used is written to {log dir}/run_metadata.json.

Secondary robustness check (--robustness):
    Random-start windows of config episode_length inside the test split.
    All strategies use exactly the same windows (same seeds). The test split
    is short, so these windows OVERLAP heavily and are NOT independent
    samples — do not treat their spread as a confidence interval.

--no-llm skips MARL+LLM entirely (no Gemini client, no API calls).

Every coordinator call is appended to {log dir}/coordinator_calls.jsonl
(seed, step, prompt, raw response, parsed and final weights, outcome,
token counts), and outcome counts per seed are printed at the end.

Every row reports turnover (sum of |weight change| over the episode,
including the initial buy from cash) and total_cost (transaction costs
paid, in currency), both as charged by StockTradingEnv. Std columns are
"-" when a strategy has a single run (n=1).

Run this after downloading checkpoints from Colab.
"""

import argparse
import numpy as np
import pandas as pd
import yaml
import os
import json
from collections import Counter
from src.environment.stock_env import StockTradingEnv, TICKERS, N_TICKERS
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.checkpoints import (
    checkpoint_dir, load_checked_agents, training_setup, check_setup,
)

# Global seed for full reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
import random
random.seed(GLOBAL_SEED)

try:
    import torch
    torch.manual_seed(GLOBAL_SEED)
except ImportError:
    pass

BAH_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0], dtype=np.float32)
ROBUSTNESS_NOTE = "overlapping windows, not independent"

# benchmark label -> rebalance daily to BAH_WEIGHTS?
BUY_AND_HOLD = "buy_and_hold"
EQUAL_WEIGHT = "equal_weight_daily_rebalanced"
BENCHMARKS   = {BUY_AND_HOLD: False, EQUAL_WEIGHT: True}

# Fallback only, used when config.yaml has no coordinator.call_frequency
# (20 trading days, about one calendar month)
DEFAULT_CALL_FREQUENCY = 20

# agent strategy label -> (agent_mode, use_llm, held on the coordinator schedule)
AGENT_STRATEGIES = {
    "MARL-no-coordinator"  : ("both", False, False),
    "price-agent-only"     : ("price", False, False),
    "sentiment-agent-only" : ("sentiment", False, False),
    "MARL-average-hold20"  : ("both", False, True),
    "price-only-hold20"    : ("price", False, True),
    "sentiment-only-hold20": ("sentiment", False, True),
    "MARL+LLM"             : ("both", True, False),
}


def call_frequency(config: dict) -> int:
    """Coordinator cadence in trading days: coordinator.call_frequency,
    else DEFAULT_CALL_FREQUENCY."""
    freq = (config or {}).get("coordinator", {}).get("call_frequency")
    return DEFAULT_CALL_FREQUENCY if freq is None else int(freq)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DEFAULT_TRAIN_SEEDS = [0, 1, 2]


def make_test_env(config: dict, full_period: bool) -> StockTradingEnv:
    """Evaluation env restricted to the test split."""
    return StockTradingEnv(config, mode="eval", split="test",
                           full_period=full_period)


def load_agents(env: StockTradingEnv, config: dict, ckpt_dir: str) -> tuple:
    """Final agents from ckpt_dir, refused unless trained with the current setup."""
    price_agent     = PriceAgent(env, config)
    sentiment_agent = SentimentAgent(env, config)
    load_checked_agents(ckpt_dir, config, [price_agent, sentiment_agent])
    return price_agent, sentiment_agent


def seed_dirs(config: dict, train_seeds: list, root: str = None) -> dict:
    """{train_seed: checkpoint folder}."""
    return {s: checkpoint_dir(config, s, root) for s in train_seeds}


def reset_window(env: StockTradingEnv, window_seed: int = None):
    """
    Reset env to an evaluation window. Shared by every strategy so
    windows are identical across strategies.
    window_seed=None -> start of the test split (fixed full-period episode).
    window_seed=int  -> deterministic random start inside the test split.
    """
    if window_seed is None:
        env.mode = "eval"
        env.reset()
        return

    env.mode = "train"
    np.random.seed(GLOBAL_SEED + window_seed)  # fully controlled per seed
    env.reset(seed=GLOBAL_SEED + window_seed)
    env.mode = "eval"


class CallLog:
    """
    Coordinator calls, one JSON line each, written as they happen so a
    run that stops part-way keeps the calls already paid for.
    """
    def __init__(self, path: str):
        self.path    = path
        self.records = []
        open(path, "w").close()

    def write(self, record: dict):
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def print_outcomes(self):
        """Outcome counts per (phase, seed)."""
        print("\n" + "="*65)
        print(f"COORDINATOR CALL OUTCOMES ({self.path})")
        print("="*65)
        counts = {}
        for r in self.records:
            counts.setdefault((r["phase"], r["seed"]), Counter())[r["outcome"]] += 1
        if not counts:
            print("  no coordinator calls")
        for (phase, seed), c in counts.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
            print(f"  {phase:11s} seed={seed} | calls={sum(c.values()):4d} | {parts}")


def agent_strategies(use_llm: bool, frequency: int) -> dict:
    """
    label -> (agent_mode, use_llm, hold_every) from AGENT_STRATEGIES, with
    the held strategies on the coordinator schedule (every frequency steps).
    MARL+LLM is left out when use_llm is False.
    """
    return {k: (mode, llm, frequency if hold else None)
            for k, (mode, llm, hold) in AGENT_STRATEGIES.items()
            if use_llm or not llm}


def _ticker_returns(env: StockTradingEnv, idx: int) -> np.ndarray:
    """Close-to-close return of each ticker from day idx to idx+1."""
    d0, d1 = env.common_dates[idx], env.common_dates[idx + 1]
    return np.array([
        (env.dataset[t].loc[d1, "close"] - env.dataset[t].loc[d0, "close"])
        / (env.dataset[t].loc[d0, "close"] + 1e-8)
        for t in TICKERS
    ])


def run_episode(env: StockTradingEnv,
                price_agent: PriceAgent = None,
                sentiment_agent: SentimentAgent = None,
                window_seed: int = None,
                use_llm: bool = False,
                config: dict = None,
                fixed_weights: np.ndarray = None,
                rebalance: bool = True,
                agent_mode: str = "both",
                hold_every: int = None,
                on_call=None) -> dict:
    """
    Run one evaluation episode on the window selected by reset_window().
    fixed_weights : if given, ignore agents and trade to these weights.
    rebalance     : with fixed_weights, True rebalances to them every step
                    (equal-weight daily-rebalanced); False buys them once
                    and lets them drift (true buy-and-hold).
    agent_mode    : "both" (average / LLM), "price" or "sentiment" only.
    LLM coordinator called every call_frequency(config) steps, only when
    agent_mode="both".
    hold_every    : without the LLM, recompute the agent target weights
                    only every hold_every steps (same schedule as the
                    coordinator) and trade back to that held target in
                    between. None recomputes every step.
    on_call       : called with each coordinator call's record (step,
                    date, prompt, response, weights, outcome, tokens).
    Adds turnover and total_cost (as charged by the env) to the summary.
    """
    if agent_mode not in ("both", "price", "sentiment"):
        raise ValueError(f"agent_mode must be both/price/sentiment, got {agent_mode!r}")
    reset_window(env, window_seed)

    held       = None   # drifted buy-and-hold weights after the first buy
    turnover   = 0.0
    total_cost = 0.0

    cached_llm_weights  = None
    llm_call_counter    = 0
    cached_hold_weights = None

    # Fix Bug 2: bypass AEC interface — directly control execution
    for step in range(env.episode_length):
        if not env.agents:
            break

        # Get current date safely
        date_idx = env.start_idx + env.timestep
        if date_idx >= env.n_steps - 1:
            break

        if fixed_weights is not None:
            final_weights = fixed_weights if held is None else held
        else:
            # Get observations for both agents
            price_obs = env.observe("price_agent")
            sent_obs  = env.observe("sentiment_agent")

            # Get both agent suggestions
            price_action = price_agent.predict(price_obs)
            sent_action  = sentiment_agent.predict(sent_obs)

            if agent_mode == "price":
                final_weights = price_action
            elif agent_mode == "sentiment":
                final_weights = sent_action
            elif use_llm and config is not None:
                # imported here so --no-llm never creates the Gemini client
                from src.coordinator.llm_coordinator import coordinate_or_average
                # Call LLM every call_frequency(config) steps
                if llm_call_counter % call_frequency(config) == 0:
                    context = {
                        "portfolio_value": env.portfolio_value,
                        "initial_capital": env.initial_capital,
                        "timestep"       : env.timestep,
                        "recent_returns" : env.returns_history[-10:],
                    }
                    record = {"step": env.timestep,
                              "date": str(env.common_dates[date_idx].date())}
                    cached_llm_weights = coordinate_or_average(
                        price_action, sent_action,
                        context, config, use_llm=True, record=record
                    )
                    if on_call is not None:
                        on_call(record)
                    #time.sleep(1)
                llm_call_counter += 1

                final_weights = cached_llm_weights \
                    if cached_llm_weights is not None \
                    else (price_action + sent_action) / 2.0
            else:
                final_weights = (price_action + sent_action) / 2.0

            if hold_every is not None and not use_llm:
                if step % hold_every == 0:
                    cached_hold_weights = final_weights
                final_weights = cached_hold_weights

        # Normalise weights
        final_weights = np.clip(final_weights, 0.0, 1.0)
        final_weights = final_weights / (final_weights.sum() + 1e-8)

        if held is not None:
            # buy-and-hold: the drifted holdings are the current weights,
            # so the env sees no trade and charges no cost
            env.portfolio_weights = final_weights.copy()

        step_turnover = float(np.abs(final_weights - env.portfolio_weights).sum())
        turnover     += step_turnover
        total_cost   += env.portfolio_value * step_turnover * env.transaction_cost

        # Fix Bug 2: directly inject final weights into environment
        # bypassing AEC turn structure so LLM decision is actually used
        env.suggestions = {
            "price_agent"    : final_weights,
            "sentiment_agent": final_weights,
        }

        # Execute the step directly
        env._execute_step()

        if fixed_weights is not None and not rebalance:
            grown = final_weights.astype(np.float64).copy()
            grown[:N_TICKERS] *= 1 + _ticker_returns(env, date_idx)
            held = grown / grown.sum()

        env.timestep += 1
        env.suggestions = {}

        # Check termination
        if env.terminations.get("price_agent", False):
            break

    summary = env.get_episode_summary()
    summary["turnover"]   = turnover
    summary["total_cost"] = total_cost
    summary["start_date"] = str(env.common_dates[env.start_idx].date())
    summary["end_date"]   = str(env.common_dates[env.start_idx + env.timestep].date())
    return summary


def _print_row(prefix: str, summary: dict):
    print(f"  {prefix} | "
          f"{summary['start_date']} → {summary['end_date']} | "
          f"Return: {summary['total_return']*100:6.2f}% | "
          f"Sharpe: {summary['sharpe_ratio']:7.4f} | "
          f"Drawdown: {summary['max_drawdown']*100:5.2f}% | "
          f"Turnover: {summary['turnover']:7.2f} | "
          f"Cost: {summary['total_cost']:9.2f}")


def _tagged(call_log: CallLog, **tags):
    """on_call hook writing each record to call_log with tags first."""
    if call_log is None:
        return None
    return lambda record: call_log.write({**tags, **record})


# ── primary: one fixed full-test-period episode ──────────────────────────────

def evaluate_test_period(config: dict, dirs: dict,
                         use_llm: bool = True,
                         call_log: CallLog = None) -> pd.DataFrame:
    """
    One full-test-period episode per strategy, same window for all.
    Agent strategies are repeated per training seed (dirs maps
    train_seed -> checkpoint folder); the two benchmarks are deterministic
    so they run once. use_llm=False skips MARL+LLM.
    Coordinator calls go to call_log, tagged with the training seed.
    """
    env = make_test_env(config, full_period=True)
    print(f"\nFixed test-period episode: "
          f"{env.common_dates[0].date()} → {env.common_dates[-1].date()} "
          f"({env.n_steps} trading days)")

    rows = []
    for label, rebalance in BENCHMARKS.items():
        summary = run_episode(env, fixed_weights=BAH_WEIGHTS,
                              rebalance=rebalance)
        summary.update(strategy=label, train_seed=None)
        rows.append(summary)
        _print_row(f"{label:29s}", summary)

    for train_seed, ckpt_dir in dirs.items():
        price_agent, sentiment_agent = load_agents(env, config, ckpt_dir)
        for label, (mode, llm, hold) in agent_strategies(use_llm, call_frequency(config)).items():
            summary = run_episode(env, price_agent, sentiment_agent,
                                  use_llm=llm, config=config,
                                  agent_mode=mode, hold_every=hold,
                                  on_call=_tagged(call_log, phase="test_period",
                                                  seed=train_seed,
                                                  strategy=label))
            summary.update(strategy=label, train_seed=train_seed)
            rows.append(summary)
            _print_row(f"{label:21s} train_seed={train_seed}", summary)

    return pd.DataFrame(rows)


# ── secondary: random-start windows (overlapping, not independent) ──────────

def evaluate_multi_seed(ckpt_dir: str,
                        config: dict,
                        seeds: list = None,
                        use_llm: bool = False,
                        agent_mode: str = "both",
                        label: str = None,
                        hold_every: int = None,
                        call_log: CallLog = None,
                        train_seed: int = None) -> pd.DataFrame:
    """
    Evaluate trained agents on random-start test windows.
    Windows overlap heavily — robustness check only.
    Coordinator calls go to call_log, tagged with train_seed and window.
    """
    if seeds is None:
        seeds = list(range(10))

    env = make_test_env(config, full_period=False)
    price_agent, sentiment_agent = load_agents(env, config, ckpt_dir)

    if label is None:
        label = "MARL+LLM" if use_llm else "MARL-no-coordinator"
    print(f"\nEvaluating {label} across {len(seeds)} windows "
          f"({ROBUSTNESS_NOTE})...")

    rows = []
    for seed in seeds:
        summary = run_episode(
            env, price_agent, sentiment_agent,
            window_seed=seed, use_llm=use_llm, config=config,
            agent_mode=agent_mode, hold_every=hold_every,
            on_call=_tagged(call_log, phase="robustness", seed=train_seed,
                            window_seed=seed, strategy=label)
        )
        summary["seed"]     = seed
        summary["strategy"] = label
        rows.append(summary)
        _print_row(f"Seed {seed:2d}", summary)

    return pd.DataFrame(rows)


def buy_and_hold_multi_seed(config: dict,
                             seeds: list = None,
                             rebalance: bool = False) -> pd.DataFrame:
    """
    Benchmark on exactly the same windows as evaluate_multi_seed
    (same seeds, episode length, test split). rebalance=False is true
    buy-and-hold, rebalance=True the equal-weight daily-rebalanced one.
    """
    if seeds is None:
        seeds = list(range(10))

    label = EQUAL_WEIGHT if rebalance else BUY_AND_HOLD
    print(f"\nEvaluating {label} across {len(seeds)} windows "
          f"({ROBUSTNESS_NOTE})...")
    env = make_test_env(config, full_period=False)

    rows = []
    for seed in seeds:
        summary = run_episode(env, window_seed=seed,
                              fixed_weights=BAH_WEIGHTS, rebalance=rebalance)
        summary["seed"]     = seed
        summary["strategy"] = label
        rows.append(summary)
        _print_row(f"Seed {seed:2d}", summary)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, title: str):
    """Print mean, std, min, max per strategy."""
    print("\n" + "="*65)
    print(title)
    print("="*65)

    metrics = ["total_return", "sharpe_ratio", "max_drawdown",
               "turnover", "total_cost"]

    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        print(f"\n{strategy} (n={len(subset)}):")
        for metric in metrics:
            vals = subset[metric]
            std  = f"{vals.std():6.4f}" if len(vals) > 1 else f"{'-':>6s}"
            print(f"  {metric:20s} | "
                  f"mean={vals.mean():7.4f} | "
                  f"std={std} | "
                  f"min={vals.min():7.4f} | "
                  f"max={vals.max():7.4f}")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per strategy; std is "-" for a single run (n=1)."""
    summary_rows = []
    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        summary_rows.append({
            "strategy"          : strategy,
            "n"                 : len(subset),
            "mean_return"       : subset["total_return"].mean(),
            "std_return"        : subset["total_return"].std(),
            "mean_sharpe"       : subset["sharpe_ratio"].mean(),
            "std_sharpe"        : subset["sharpe_ratio"].std(),
            "mean_drawdown"     : subset["max_drawdown"].mean(),
            "std_drawdown"      : subset["max_drawdown"].std(),
            "mean_final_value"  : subset["final_value"].mean(),
            "mean_turnover"     : subset["turnover"].mean(),
            "std_turnover"      : subset["turnover"].std(),
            "mean_total_cost"   : subset["total_cost"].mean(),
            "std_total_cost"    : subset["total_cost"].std(),
        })
    out = pd.DataFrame(summary_rows)
    for col in [c for c in out.columns if c.startswith("std_")]:
        out[col] = out[col].astype(object).where(out["n"] > 1, "-")
    return out


def run_full_evaluation(config: dict,
                        train_seeds: list = None,
                        robustness: bool = False,
                        window_seeds: list = None,
                        checkpoint_root: str = None,
                        use_llm: bool = True,
                        tag: str = None):
    """
    1. Primary: fixed full-test-period episode for both benchmarks and
       every agent strategy (per training seed).
       std columns = variance across TRAINING seeds.
    2. Optional robustness: random-start test windows, same windows for
       every strategy, using the first training seed's checkpoints.
    Checkpoints: {checkpoint_root or paths.checkpoints_root}/seed{N}/.
    Every seed folder is checked against the current setup before anything runs.
    use_llm=False skips every MARL+LLM run, so Gemini is never called.
    Save results to results/logs/, or results/logs/{tag}/ when tag is given;
    with use_llm, coordinator calls go to coordinator_calls.jsonl there.
    """
    if train_seeds is None:
        train_seeds = DEFAULT_TRAIN_SEEDS
    if window_seeds is None:
        window_seeds = list(range(10))

    dirs = seed_dirs(config, train_seeds, checkpoint_root)
    # fail before any (LLM) episode runs if one seed folder is unusable
    setup = training_setup(config)
    for ckpt_dir in dirs.values():
        check_setup(ckpt_dir, setup)

    log_dir = os.path.join("results", "logs", tag) if tag else os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    call_log = CallLog(os.path.join(log_dir, "coordinator_calls.jsonl")) if use_llm else None

    test_df = evaluate_test_period(config, dirs, use_llm=use_llm, call_log=call_log)
    test_df.to_csv(os.path.join(log_dir, "test_period_results.csv"), index=False)
    print(f"\nSaved → {os.path.join(log_dir, 'test_period_results.csv')}")
    print_summary(test_df, "FIXED TEST-PERIOD EPISODE (spread = across training seeds)")

    test_summary = summarise(test_df)
    test_summary.to_csv(os.path.join(log_dir, "test_period_summary.csv"), index=False)
    print(f"Saved → {os.path.join(log_dir, 'test_period_summary.csv')}")

    # record the cadence this folder was produced with
    metadata = {"call_frequency": call_frequency(config),
                "call_frequency_source": "config.yaml coordinator.call_frequency"
                    if config.get("coordinator", {}).get("call_frequency") is not None
                    else "DEFAULT_CALL_FREQUENCY fallback",
                "use_llm": use_llm,
                "train_seeds": list(train_seeds)}
    with open(os.path.join(log_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved → {os.path.join(log_dir, 'run_metadata.json')} "
          f"(call_frequency={metadata['call_frequency']})")

    if robustness:
        ckpt_dir = dirs[train_seeds[0]]
        rob_parts = [
            buy_and_hold_multi_seed(config, window_seeds, rebalance=rebalance)
            for rebalance in BENCHMARKS.values()
        ]
        for label, (mode, llm, hold) in agent_strategies(use_llm, call_frequency(config)).items():
            rob_parts.append(evaluate_multi_seed(
                ckpt_dir, config, seeds=window_seeds, use_llm=llm,
                agent_mode=mode, label=label, hold_every=hold,
                call_log=call_log, train_seed=train_seeds[0]
            ))
        rob_df = pd.concat(rob_parts, ignore_index=True)
        rob_df["train_seed"] = train_seeds[0]
        rob_df["note"]       = ROBUSTNESS_NOTE
        rob_df.to_csv(os.path.join(log_dir, "robustness_windows_results.csv"), index=False)
        print(f"\nSaved → {os.path.join(log_dir, 'robustness_windows_results.csv')}")
        print_summary(rob_df, f"ROBUSTNESS: RANDOM TEST WINDOWS ({ROBUSTNESS_NOTE})")

        rob_summary = summarise(rob_df)
        rob_summary["note"] = ROBUSTNESS_NOTE
        rob_summary.to_csv(os.path.join(log_dir, "robustness_windows_summary.csv"), index=False)
        print(f"Saved → {os.path.join(log_dir, 'robustness_windows_summary.csv')}")

    print("\n" + "="*65)
    print("FINAL COMPARISON TABLE (fixed test period)")
    print("="*65)
    print(test_summary.to_string(index=False))

    if call_log is not None:
        call_log.print_outcomes()

    return test_df, test_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on the test split")
    parser.add_argument("--checkpoint-root", type=str, default=None,
                        help="folder holding the seed{N}/ checkpoint folders; "
                             "default: paths.checkpoints_root in config.yaml")
    parser.add_argument("--train-seeds", type=int, nargs="+",
                        default=DEFAULT_TRAIN_SEEDS,
                        help="training seeds to evaluate (default: 0 1 2)")
    parser.add_argument("--robustness", action="store_true",
                        help="also run random-start test windows "
                             "(overlapping, not independent)")
    parser.add_argument("--window-seeds", type=int, default=10,
                        help="number of random windows for --robustness")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip MARL+LLM runs (no Gemini calls)")
    parser.add_argument("--tag", type=str, default=None,
                        help="write results to results/logs/{tag}/ "
                             "instead of results/logs/")
    args = parser.parse_args()

    config = load_config()
    run_full_evaluation(
        config,
        train_seeds     = args.train_seeds,
        robustness      = args.robustness,
        window_seeds    = list(range(args.window_seeds)),
        checkpoint_root = args.checkpoint_root,
        use_llm         = not args.no_llm,
        tag             = args.tag,
    )
