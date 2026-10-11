"""
Multi-seed evaluation script for trained MARL agents.
Runs evaluation across multiple random seeds and reports
mean, std, min and max for all metrics.
Run this after downloading checkpoints from Colab.
"""

import numpy as np
import pandas as pd
import yaml
import os
from src.environment.stock_env import StockTradingEnv
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.coordinator.llm_coordinator import coordinate_or_average

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def run_episode(env: StockTradingEnv,
                price_agent: PriceAgent,
                sentiment_agent: SentimentAgent,
                seed: int,
                use_llm: bool = False,
                config: dict = None) -> dict:
    """
    Run one full evaluation episode.
    Returns episode summary metrics.
    """
    env.mode = "eval"
    obs, _   = env.reset(seed=seed)

    while env.agents:
        current = env.agent_selection

        if env.terminations.get(current, False) or \
           env.truncations.get(current, False):
            env.step(None)
            continue

        # Get both agent observations
        price_obs = env.observe("price_agent")
        sent_obs  = env.observe("sentiment_agent")

        # Get both agent suggestions
        price_action = price_agent.predict(price_obs)
        sent_action  = sentiment_agent.predict(sent_obs)

        if use_llm and config is not None:
            # Use LLM coordinator for final decision
            context = {
                "portfolio_value" : env.portfolio_value,
                "initial_capital" : env.initial_capital,
                "timestep"        : env.timestep,
                "recent_returns"  : env.returns_history[-10:],
            }
            final_action = coordinate_or_average(
                price_action, sent_action,
                context, config, use_llm=True
            )
        else:
            # Simple average (no coordinator)
            final_action = (price_action + sent_action) / 2.0
            final_action = final_action / (final_action.sum() + 1e-8)

        # Step current agent
        if current == "price_agent":
            env.step(price_action)
        else:
            env.step(sent_action)

    return env.get_episode_summary()


def evaluate_multi_seed(price_ckpt: str,
                        sentiment_ckpt: str,
                        config: dict,
                        seeds: list = None,
                        use_llm: bool = False) -> pd.DataFrame:
    """
    Evaluate trained agents across multiple seeds.
    Returns dataframe with results per seed.
    """
    if seeds is None:
        seeds = list(range(10))

    # Initialise environment and agents
    env = StockTradingEnv(config, mode="eval")

    price_agent     = PriceAgent(env, config)
    sentiment_agent = SentimentAgent(env, config)

    # Load checkpoints
    if os.path.exists(price_ckpt):
        price_agent.load(price_ckpt)
        print(f"Loaded price agent from {price_ckpt}")
    else:
        print(f"WARNING: {price_ckpt} not found — using untrained model")

    if os.path.exists(sentiment_ckpt):
        sentiment_agent.load(sentiment_ckpt)
        print(f"Loaded sentiment agent from {sentiment_ckpt}")
    else:
        print(f"WARNING: {sentiment_ckpt} not found — using untrained model")

    label = "MARL+LLM" if use_llm else "MARL-no-coordinator"
    print(f"\nEvaluating {label} across {len(seeds)} seeds...")

    rows = []
    for seed in seeds:
        summary = run_episode(
            env, price_agent, sentiment_agent,
            seed=seed, use_llm=use_llm, config=config
        )
        summary["seed"]     = seed
        summary["strategy"] = label
        rows.append(summary)
        print(f"  Seed {seed:2d} | "
              f"Return: {summary['total_return']*100:6.2f}% | "
              f"Sharpe: {summary['sharpe_ratio']:7.4f} | "
              f"Drawdown: {summary['max_drawdown']*100:5.2f}%")

    return pd.DataFrame(rows)


def buy_and_hold_multi_seed(config: dict,
                             seeds: list = None) -> pd.DataFrame:
    """Buy-and-hold baseline across multiple seeds."""
    if seeds is None:
        seeds = list(range(10))

    print(f"\nEvaluating buy-and-hold across {len(seeds)} seeds...")
    env     = StockTradingEnv(config, mode="eval")
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0], dtype=np.float32)

    rows = []
    for seed in seeds:
        env.mode = "eval"
        obs, _   = env.reset(seed=seed)

        while env.agents:
            current = env.agent_selection
            if env.terminations.get(current, False) or \
               env.truncations.get(current, False):
                env.step(None)
                continue
            env.step(weights)

        summary             = env.get_episode_summary()
        summary["seed"]     = seed
        summary["strategy"] = "buy_and_hold"
        rows.append(summary)
        print(f"  Seed {seed:2d} | "
              f"Return: {summary['total_return']*100:6.2f}% | "
              f"Sharpe: {summary['sharpe_ratio']:7.4f} | "
              f"Drawdown: {summary['max_drawdown']*100:5.2f}%")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Print mean, std, min, max per strategy."""
    print("\n" + "="*65)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("="*65)

    metrics = ["total_return", "sharpe_ratio", "max_drawdown"]

    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        print(f"\n{strategy}:")
        for metric in metrics:
            vals = subset[metric]
            print(f"  {metric:20s} | "
                  f"mean={vals.mean():7.4f} | "
                  f"std={vals.std():6.4f} | "
                  f"min={vals.min():7.4f} | "
                  f"max={vals.max():7.4f}")


def run_full_evaluation(config: dict,
                        price_ckpt: str,
                        sentiment_ckpt: str,
                        seeds: list = None):
    """
    Run complete evaluation:
    1. Buy-and-hold baseline
    2. MARL without coordinator
    3. MARL with LLM coordinator
    Save results to results/logs/
    """
    if seeds is None:
        seeds = list(range(10))

    os.makedirs("results/logs", exist_ok=True)

    # Run all three strategies
    bah_df  = buy_and_hold_multi_seed(config, seeds)
    marl_df = evaluate_multi_seed(
        price_ckpt, sentiment_ckpt, config,
        seeds=seeds, use_llm=False
    )
    llm_df  = evaluate_multi_seed(
        price_ckpt, sentiment_ckpt, config,
        seeds=seeds, use_llm=True
    )

    # Combine
    all_results = pd.concat([bah_df, marl_df, llm_df], ignore_index=True)
    all_results.to_csv("results/logs/multi_seed_results.csv", index=False)
    print("\nSaved → results/logs/multi_seed_results.csv")

    # Summary table
    print_summary(all_results)

    # Save summary
    summary_rows = []
    for strategy in all_results["strategy"].unique():
        subset = all_results[all_results["strategy"] == strategy]
        summary_rows.append({
            "strategy"          : strategy,
            "mean_return"       : subset["total_return"].mean(),
            "std_return"        : subset["total_return"].std(),
            "mean_sharpe"       : subset["sharpe_ratio"].mean(),
            "std_sharpe"        : subset["sharpe_ratio"].std(),
            "mean_drawdown"     : subset["max_drawdown"].mean(),
            "std_drawdown"      : subset["max_drawdown"].std(),
            "mean_final_value"  : subset["final_value"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/logs/multi_seed_summary.csv", index=False)
    print("Saved → results/logs/multi_seed_summary.csv")

    print("\n" + "="*65)
    print("FINAL COMPARISON TABLE")
    print("="*65)
    print(summary_df.to_string(index=False))

    return all_results, summary_df


if __name__ == "__main__":
    config = load_config()

    price_ckpt     = "results/checkpoints/price_agent_final.zip"
    sentiment_ckpt = "results/checkpoints/sentiment_agent_final.zip"
    seeds          = list(range(10))  # seeds 0-9

    all_results, summary = run_full_evaluation(
        config, price_ckpt, sentiment_ckpt, seeds
    )