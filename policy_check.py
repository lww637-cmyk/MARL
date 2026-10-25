"""
Local version of Step 7 (policy check) in notebooks/03_training_colab.ipynb.

Loads the final price and sentiment checkpoints for one training seed from
{paths.checkpoints_root}/seed{N} with load_checked_agents (setup check on),
runs both agents deterministically on every TRAINING-split day, and prints
each agent's per-ticker allocation mean and std, plus cash. A per-asset std
near 0 means the policy ignores its inputs.

Train split only; no test data, no LLM calls, nothing written.

Usage:
    python policy_check.py --train-seed 4
"""

import argparse
import pandas as pd
import yaml
from src.environment.stock_env import StockTradingEnv
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.checkpoints import checkpoint_dir, load_checked_agents


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def policy_check(config: dict, train_seed: int):
    tickers = config["data"]["tickers"]
    ckpt = checkpoint_dir(config, train_seed)

    env = StockTradingEnv(config, mode="eval", split="train")
    env.reset(seed=0)
    agents = [PriceAgent(env, config), SentimentAgent(env, config)]
    load_checked_agents(ckpt, config, agents)

    print(f"Train seed {train_seed} ({ckpt}) | train split: {env.n_steps} days "
          f"({env.common_dates[0].date()} .. {env.common_dates[-1].date()})")
    cols = tickers + ["cash"]
    for agent in agents:
        weights = []
        for i in range(env.n_steps):
            env.start_idx, env.timestep = i, 0
            weights.append(agent.predict(env.observe(agent.agent_id)))
        w = pd.DataFrame(weights, columns=cols)
        print(f"\n{agent.agent_id} — allocation over {len(w)} train days")
        print(pd.DataFrame({"mean": w.mean(), "std": w.std()}).round(3).T.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()
    policy_check(load_config(), args.train_seed)
