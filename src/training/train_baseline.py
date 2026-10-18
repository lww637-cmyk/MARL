"""
Trains and evaluates baseline strategies for benchmarking.
Baselines:
1. Buy-and-hold — equal weight across all 5 tickers, held for full period
2. MARL without LLM coordinator — agents act independently, weights averaged
Results saved to results/logs/ for comparison in evaluation phase.
"""

import os
import yaml
import numpy as np
import pandas as pd
from src.environment.stock_env import StockTradingEnv, TICKERS
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def buy_and_hold(config: dict) -> pd.DataFrame:
    """
    Equal-weight portfolio across all 5 tickers.
    Held for the full evaluation period without rebalancing.
    Returns daily portfolio value series.
    """
    print("Running buy-and-hold baseline...")
    env = StockTradingEnv(config, mode="eval", split="test", full_period=True)
    obs, _ = env.reset(seed=0)

    # Equal weights across 5 tickers, no cash
    weights = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.0], dtype=np.float32
    )

    portfolio_values = [env.initial_capital]
    dates = []

    while env.agents:
        current = env.agent_selection
        if env.terminations.get(current, False) or \
           env.truncations.get(current, False):
            env.step(None)
            continue
        env.step(weights)

        if env.infos.get(current, {}).get("portfolio_value"):
            portfolio_values.append(
                env.infos[current]["portfolio_value"]
            )

    summary = env.get_episode_summary()
    print(f"Buy-and-hold — Return: {summary['total_return']*100:.2f}% | "
          f"Sharpe: {summary['sharpe_ratio']:.4f} | "
          f"Drawdown: {summary['max_drawdown']*100:.2f}%")

    return pd.Series(
        env.portfolio_history,
        name="buy_and_hold"
    ), summary


def marl_no_coordinator(config: dict,
                         price_ckpt: str = None,
                         sentiment_ckpt: str = None) -> pd.DataFrame:
    """
    MARL without LLM coordinator — agents act independently,
    final weights are simple average of both suggestions.
    Loads trained agent checkpoints if available.
    Returns daily portfolio value series.
    """
    print("\nRunning MARL-no-coordinator baseline...")
    env = StockTradingEnv(config, mode="eval", split="test", full_period=True)

    price_agent     = PriceAgent(env, config)
    sentiment_agent = SentimentAgent(env, config)

    # Load checkpoints if available
    if price_ckpt and os.path.exists(price_ckpt):
        price_agent.load(price_ckpt)
        print(f"Loaded price agent from {price_ckpt}")
    else:
        print("Warning: No price agent checkpoint found — using untrained model")

    if sentiment_ckpt and os.path.exists(sentiment_ckpt):
        sentiment_agent.load(sentiment_ckpt)
        print(f"Loaded sentiment agent from {sentiment_ckpt}")
    else:
        print("Warning: No sentiment agent checkpoint — using untrained model")

    obs, _ = env.reset(seed=0)

    while env.agents:
        current = env.agent_selection
        if env.terminations.get(current, False) or \
           env.truncations.get(current, False):
            env.step(None)
            continue

        obs = env.observe(current)
        if current == "price_agent":
            action = price_agent.predict(obs)
        else:
            action = sentiment_agent.predict(obs)

        env.step(action)

    summary = env.get_episode_summary()
    print(f"MARL-no-coordinator — Return: {summary['total_return']*100:.2f}% | "
          f"Sharpe: {summary['sharpe_ratio']:.4f} | "
          f"Drawdown: {summary['max_drawdown']*100:.2f}%")

    return pd.Series(
        env.portfolio_history,
        name="marl_no_coordinator"
    ), summary


def run_all_baselines(config: dict,
                      price_ckpt: str = None,
                      sentiment_ckpt: str = None) -> pd.DataFrame:
    """
    Run all baselines and save combined results to CSV.
    """
    os.makedirs("results/logs", exist_ok=True)

    bah_series, bah_summary       = buy_and_hold(config)
    marl_series, marl_summary     = marl_no_coordinator(
        config, price_ckpt, sentiment_ckpt
    )

    # Combine into comparison dataframe
    results = pd.DataFrame({
        "buy_and_hold"        : bah_series.values,
        "marl_no_coordinator" : marl_series.values,
    })
    results.to_csv("results/logs/baseline_results.csv", index=False)
    print("\nBaseline results saved → results/logs/baseline_results.csv")

    # Summary table
    summary_df = pd.DataFrame([
        {
            "strategy"    : "buy_and_hold",
            "total_return": bah_summary["total_return"],
            "sharpe_ratio": bah_summary["sharpe_ratio"],
            "max_drawdown": bah_summary["max_drawdown"],
            "final_value" : bah_summary["final_value"],
        },
        {
            "strategy"    : "marl_no_coordinator",
            "total_return": marl_summary["total_return"],
            "sharpe_ratio": marl_summary["sharpe_ratio"],
            "max_drawdown": marl_summary["max_drawdown"],
            "final_value" : marl_summary["final_value"],
        },
    ])
    summary_df.to_csv("results/logs/baseline_summary.csv", index=False)
    print("Baseline summary saved → results/logs/baseline_summary.csv")

    print("\n=== BASELINE COMPARISON ===")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    config = load_config()
    run_all_baselines(
        config,
        price_ckpt    = "results/checkpoints/price_agent_final.zip",
        sentiment_ckpt= "results/checkpoints/sentiment_agent_final.zip",
    )