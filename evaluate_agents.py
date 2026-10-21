import time

"""
Evaluation script for trained MARL agents. Test split only (>= 2022-07-01).

Primary evaluation:
    ONE fixed episode covering the full test period, identical for
    buy-and-hold, MARL-no-coordinator and MARL+LLM. Run once per
    training seed ({checkpoint root}/seed{N}/) to report variance
    across training seeds. Each seed folder must hold a
    training_setup.json matching the current setup, or it is refused.

Secondary robustness check (--robustness):
    Random-start windows of config episode_length inside the test split.
    All strategies use exactly the same windows (same seeds). The test split
    is short, so these windows OVERLAP heavily and are NOT independent
    samples — do not treat their spread as a confidence interval.

Run this after downloading checkpoints from Colab.
"""

import argparse
import numpy as np
import pandas as pd
import yaml
import os
from src.environment.stock_env import StockTradingEnv
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.coordinator.llm_coordinator import coordinate_or_average
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


def run_episode(env: StockTradingEnv,
                price_agent: PriceAgent = None,
                sentiment_agent: SentimentAgent = None,
                window_seed: int = None,
                use_llm: bool = False,
                config: dict = None,
                fixed_weights: np.ndarray = None) -> dict:
    """
    Run one evaluation episode on the window selected by reset_window().
    fixed_weights : if given, ignore agents and hold these target weights
                    every step (used for buy-and-hold).
    LLM coordinator called every 20 steps (monthly cadence).
    """
    reset_window(env, window_seed)

    cached_llm_weights = None
    llm_call_counter   = 0
    LLM_FREQUENCY      = 20

    # Fix Bug 2: bypass AEC interface — directly control execution
    for step in range(env.episode_length):
        if not env.agents:
            break

        # Get current date safely
        date_idx = env.start_idx + env.timestep
        if date_idx >= env.n_steps - 1:
            break

        if fixed_weights is not None:
            price_action = sent_action = final_weights = fixed_weights
        else:
            # Get observations for both agents
            price_obs = env.observe("price_agent")
            sent_obs  = env.observe("sentiment_agent")

            # Get both agent suggestions
            price_action = price_agent.predict(price_obs)
            sent_action  = sentiment_agent.predict(sent_obs)

            if use_llm and config is not None:
                # Call LLM every LLM_FREQUENCY steps
                if llm_call_counter % LLM_FREQUENCY == 0:
                    context = {
                        "portfolio_value": env.portfolio_value,
                        "initial_capital": env.initial_capital,
                        "timestep"       : env.timestep,
                        "recent_returns" : env.returns_history[-10:],
                    }
                    cached_llm_weights = coordinate_or_average(
                        price_action, sent_action,
                        context, config, use_llm=True
                    )
                    #time.sleep(1)
                llm_call_counter += 1

                final_weights = cached_llm_weights \
                    if cached_llm_weights is not None \
                    else (price_action + sent_action) / 2.0
            else:
                final_weights = (price_action + sent_action) / 2.0

        # Normalise weights
        final_weights = np.clip(final_weights, 0.0, 1.0)
        final_weights = final_weights / (final_weights.sum() + 1e-8)

        # Fix Bug 2: directly inject final weights into environment
        # bypassing AEC turn structure so LLM decision is actually used
        env.suggestions = {
            "price_agent"    : final_weights,
            "sentiment_agent": final_weights,
        }

        # Execute the step directly
        env._execute_step()
        env.timestep += 1
        env.suggestions = {}

        # Check termination
        if env.terminations.get("price_agent", False):
            break

    summary = env.get_episode_summary()
    summary["start_date"] = str(env.common_dates[env.start_idx].date())
    summary["end_date"]   = str(env.common_dates[env.start_idx + env.timestep].date())
    return summary


def _print_row(prefix: str, summary: dict):
    print(f"  {prefix} | "
          f"{summary['start_date']} → {summary['end_date']} | "
          f"Return: {summary['total_return']*100:6.2f}% | "
          f"Sharpe: {summary['sharpe_ratio']:7.4f} | "
          f"Drawdown: {summary['max_drawdown']*100:5.2f}%")


# ── primary: one fixed full-test-period episode ──────────────────────────────

def evaluate_test_period(config: dict, dirs: dict) -> pd.DataFrame:
    """
    One full-test-period episode per strategy, same window for all.
    MARL strategies are repeated per training seed (dirs maps
    train_seed -> checkpoint folder); buy-and-hold is deterministic so it
    runs once.
    """
    env = make_test_env(config, full_period=True)
    print(f"\nFixed test-period episode: "
          f"{env.common_dates[0].date()} → {env.common_dates[-1].date()} "
          f"({env.n_steps} trading days)")

    rows = []
    summary = run_episode(env, fixed_weights=BAH_WEIGHTS)
    summary.update(strategy="buy_and_hold", train_seed=None)
    rows.append(summary)
    _print_row(f"{'buy_and_hold':20s}", summary)

    for train_seed, ckpt_dir in dirs.items():
        price_agent, sentiment_agent = load_agents(env, config, ckpt_dir)
        for use_llm in (False, True):
            label   = "MARL+LLM" if use_llm else "MARL-no-coordinator"
            summary = run_episode(env, price_agent, sentiment_agent,
                                  use_llm=use_llm, config=config)
            summary.update(strategy=label, train_seed=train_seed)
            rows.append(summary)
            _print_row(f"{label:20s} train_seed={train_seed}", summary)

    return pd.DataFrame(rows)


# ── secondary: random-start windows (overlapping, not independent) ──────────

def evaluate_multi_seed(ckpt_dir: str,
                        config: dict,
                        seeds: list = None,
                        use_llm: bool = False) -> pd.DataFrame:
    """
    Evaluate trained agents on random-start test windows.
    Windows overlap heavily — robustness check only.
    """
    if seeds is None:
        seeds = list(range(10))

    env = make_test_env(config, full_period=False)
    price_agent, sentiment_agent = load_agents(env, config, ckpt_dir)

    label = "MARL+LLM" if use_llm else "MARL-no-coordinator"
    print(f"\nEvaluating {label} across {len(seeds)} windows "
          f"({ROBUSTNESS_NOTE})...")

    rows = []
    for seed in seeds:
        summary = run_episode(
            env, price_agent, sentiment_agent,
            window_seed=seed, use_llm=use_llm, config=config
        )
        summary["seed"]     = seed
        summary["strategy"] = label
        rows.append(summary)
        _print_row(f"Seed {seed:2d}", summary)

    return pd.DataFrame(rows)


def buy_and_hold_multi_seed(config: dict,
                             seeds: list = None) -> pd.DataFrame:
    """
    Buy-and-hold baseline on exactly the same windows as
    evaluate_multi_seed (same seeds, episode length, test split).
    """
    if seeds is None:
        seeds = list(range(10))

    print(f"\nEvaluating buy-and-hold across {len(seeds)} windows "
          f"({ROBUSTNESS_NOTE})...")
    env = make_test_env(config, full_period=False)

    rows = []
    for seed in seeds:
        summary = run_episode(env, window_seed=seed,
                              fixed_weights=BAH_WEIGHTS)
        summary["seed"]     = seed
        summary["strategy"] = "buy_and_hold"
        rows.append(summary)
        _print_row(f"Seed {seed:2d}", summary)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, title: str):
    """Print mean, std, min, max per strategy."""
    print("\n" + "="*65)
    print(title)
    print("="*65)

    metrics = ["total_return", "sharpe_ratio", "max_drawdown"]

    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        print(f"\n{strategy} (n={len(subset)}):")
        for metric in metrics:
            vals = subset[metric]
            print(f"  {metric:20s} | "
                  f"mean={vals.mean():7.4f} | "
                  f"std={vals.std():6.4f} | "
                  f"min={vals.min():7.4f} | "
                  f"max={vals.max():7.4f}")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
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
        })
    return pd.DataFrame(summary_rows)


def run_full_evaluation(config: dict,
                        train_seeds: list = None,
                        robustness: bool = False,
                        window_seeds: list = None,
                        checkpoint_root: str = None):
    """
    1. Primary: fixed full-test-period episode for buy-and-hold,
       MARL-no-coordinator and MARL+LLM (per training seed).
       std columns = variance across TRAINING seeds.
    2. Optional robustness: random-start test windows, same windows for
       every strategy, using the first training seed's checkpoints.
    Checkpoints: {checkpoint_root or paths.checkpoints_root}/seed{N}/.
    Every seed folder is checked against the current setup before anything runs.
    Save results to results/logs/
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

    os.makedirs("results/logs", exist_ok=True)

    test_df = evaluate_test_period(config, dirs)
    test_df.to_csv("results/logs/test_period_results.csv", index=False)
    print("\nSaved → results/logs/test_period_results.csv")
    print_summary(test_df, "FIXED TEST-PERIOD EPISODE (spread = across training seeds)")

    test_summary = summarise(test_df)
    test_summary.to_csv("results/logs/test_period_summary.csv", index=False)
    print("Saved → results/logs/test_period_summary.csv")

    if robustness:
        ckpt_dir = dirs[train_seeds[0]]
        bah_df  = buy_and_hold_multi_seed(config, window_seeds)
        marl_df = evaluate_multi_seed(
            ckpt_dir, config, seeds=window_seeds, use_llm=False
        )
        llm_df  = evaluate_multi_seed(
            ckpt_dir, config, seeds=window_seeds, use_llm=True
        )
        rob_df = pd.concat([bah_df, marl_df, llm_df], ignore_index=True)
        rob_df["train_seed"] = train_seeds[0]
        rob_df["note"]       = ROBUSTNESS_NOTE
        rob_df.to_csv("results/logs/robustness_windows_results.csv", index=False)
        print("\nSaved → results/logs/robustness_windows_results.csv")
        print_summary(rob_df, f"ROBUSTNESS: RANDOM TEST WINDOWS ({ROBUSTNESS_NOTE})")

        rob_summary = summarise(rob_df)
        rob_summary["note"] = ROBUSTNESS_NOTE
        rob_summary.to_csv("results/logs/robustness_windows_summary.csv", index=False)
        print("Saved → results/logs/robustness_windows_summary.csv")

    print("\n" + "="*65)
    print("FINAL COMPARISON TABLE (fixed test period)")
    print("="*65)
    print(test_summary.to_string(index=False))

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
    args = parser.parse_args()

    config = load_config()
    run_full_evaluation(
        config,
        train_seeds     = args.train_seeds,
        robustness      = args.robustness,
        window_seeds    = list(range(args.window_seeds)),
        checkpoint_root = args.checkpoint_root,
    )
