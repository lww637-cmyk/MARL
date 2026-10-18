"""
Trains both price and sentiment agents within the shared stock trading
environment. Logs training metrics to MLflow.
Designed to run on Google Colab with GPU acceleration.
"""

import os
import random
import shutil
import argparse
import yaml
import numpy as np
import mlflow
import mlflow.pytorch
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.stock_env import StockTradingEnv
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mirror_file(path: str, copy_dir: str = None):
    """Copy a saved file into copy_dir (e.g. Google Drive) if one is given."""
    if copy_dir is None:
        return
    os.makedirs(copy_dir, exist_ok=True)
    shutil.copy(path, copy_dir)
    print(f"Copied {path} → {copy_dir}/")


# ── custom callback ──────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Logs episode rewards and portfolio metrics to MLflow
    during training. Saves checkpoints at regular intervals.
    """

    def __init__(self, agent_id: str, config: dict,
                 save_freq: int = 50000, verbose: int = 0,
                 ckpt_dir: str = "results/checkpoints",
                 copy_dir: str = None):
        super().__init__(verbose)
        self.agent_id  = agent_id
        self.config    = config
        self.save_freq = save_freq
        self.ckpt_dir  = ckpt_dir
        self.copy_dir  = copy_dir
        self.episode_rewards = []
        self.episode_count   = 0

    def _on_step(self) -> bool:
        # Log reward every step
        reward = self.locals.get("rewards", [0])[0]
        self.episode_rewards.append(reward)

        # Log to MLflow every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mlflow.log_metric(
                f"{self.agent_id}_mean_reward",
                mean_reward,
                step=self.n_calls
            )

        # Save checkpoint
        if self.n_calls % self.save_freq == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            path = f"{self.ckpt_dir}/{self.agent_id}_step{self.n_calls}"
            self.model.save(path)
            print(f"Checkpoint saved: {path}")
            mirror_file(f"{path}.zip", self.copy_dir)

        return True


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_agent(agent, env: StockTradingEnv,
                   n_episodes: int = 5) -> dict:
    """
    Run agent for n_episodes in eval mode and return
    average performance metrics.
    """
    agent_id = agent.agent_id
    all_returns  = []
    all_sharpes  = []
    all_drawdowns = []

    for ep in range(n_episodes):
        env.mode = "eval"
        obs, _   = env.reset(seed=ep)

        while env.agents:
            current = env.agent_selection
            if env.terminations.get(current, False) or \
               env.truncations.get(current, False):
                env.step(None)
                continue

            if current == agent_id:
                action = agent.predict(env.observe(agent_id))
            else:
                action = env.action_space(current).sample()

            env.step(action)

        summary = env.get_episode_summary()
        all_returns.append(summary["total_return"])
        all_sharpes.append(summary["sharpe_ratio"])
        all_drawdowns.append(summary["max_drawdown"])

    return {
        "mean_return"  : float(np.mean(all_returns)),
        "mean_sharpe"  : float(np.mean(all_sharpes)),
        "mean_drawdown": float(np.mean(all_drawdowns)),
        "std_return"   : float(np.std(all_returns)),
    }


# ── main training ────────────────────────────────────────────────────────────

def checkpoint_dir(seed: int = None) -> str:
    """results/checkpoints for unseeded runs, results/checkpoints/seed{N} otherwise."""
    if seed is None:
        return "results/checkpoints"
    return f"results/checkpoints/seed{seed}"


def set_global_seed(seed: int):
    """Seed numpy, random and torch for a reproducible training run."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def train(config: dict, seed: int = None, copy_dir: str = None):
    """
    Full training pipeline:
    1. Initialise environment and agents (train split only)
    2. Train price agent
    3. Train sentiment agent
    4. Evaluate both agents
    5. Save final models

    seed : optional training seed. When given, all RNGs and PPO are seeded
           and checkpoints go to results/checkpoints/seed{seed}/.
    copy_dir : optional folder (e.g. on Google Drive) that every checkpoint
               and final model is copied to as soon as it is saved.
    """
    total_timesteps = config["training"]["total_timesteps"]
    eval_freq       = config["training"]["eval_frequency"]
    save_freq       = config["training"]["save_frequency"]
    n_eval_episodes = config["training"]["n_eval_episodes"]

    if seed is not None:
        set_global_seed(seed)
        config["training"]["seed"] = seed   # read by BaseTrader -> PPO(seed=...)
    ckpt_dir = checkpoint_dir(seed)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # Initialise environment
    print("Initialising environment...")
    env = StockTradingEnv(config, mode="train", split="train")
    print(f"Environment ready — {env.n_steps} trading days available (train split)")

    # Start MLflow run
    mlflow.set_experiment("agentic_marl_trading")

    run_name = "marl_training" if seed is None else f"marl_training_seed{seed}"
    with mlflow.start_run(run_name=run_name):

        # Log config params
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("episode_length",  config["environment"]["episode_length"])
        mlflow.log_param("initial_capital", config["environment"]["initial_capital"])
        mlflow.log_param("tickers",         config["data"]["tickers"])
        mlflow.log_param("train_seed",      seed)

        # ── Train Price Agent ──────────────────────────────────────────────
        print("\n" + "="*50)
        print("TRAINING PRICE AGENT")
        print("="*50)

        price_agent = PriceAgent(env, config)
        price_callback = TrainingCallback(
            "price_agent", config, save_freq=save_freq, ckpt_dir=ckpt_dir,
            copy_dir=copy_dir
        )

        price_agent.model.learn(
            total_timesteps = total_timesteps,
            callback        = price_callback,
            progress_bar    = True,
        )

        price_agent.save(f"{ckpt_dir}/price_agent_final.zip")
        mirror_file(f"{ckpt_dir}/price_agent_final.zip", copy_dir)
        print("Price agent training complete.")

        # Evaluate price agent
        print("\nEvaluating price agent...")
        price_metrics = evaluate_agent(price_agent, env, n_eval_episodes)
        print(f"Price agent — Return: {price_metrics['mean_return']*100:.2f}% | "
              f"Sharpe: {price_metrics['mean_sharpe']:.4f} | "
              f"Drawdown: {price_metrics['mean_drawdown']*100:.2f}%")

        mlflow.log_metrics({
            "price_mean_return"  : price_metrics["mean_return"],
            "price_mean_sharpe"  : price_metrics["mean_sharpe"],
            "price_mean_drawdown": price_metrics["mean_drawdown"],
        })

        # ── Train Sentiment Agent ──────────────────────────────────────────
        print("\n" + "="*50)
        print("TRAINING SENTIMENT AGENT")
        print("="*50)

        sentiment_agent = SentimentAgent(env, config)
        sentiment_callback = TrainingCallback(
            "sentiment_agent", config, save_freq=save_freq, ckpt_dir=ckpt_dir,
            copy_dir=copy_dir
        )

        sentiment_agent.model.learn(
            total_timesteps = total_timesteps,
            callback        = sentiment_callback,
            progress_bar    = True,
        )

        sentiment_agent.save(f"{ckpt_dir}/sentiment_agent_final.zip")
        mirror_file(f"{ckpt_dir}/sentiment_agent_final.zip", copy_dir)
        print("Sentiment agent training complete.")

        # Evaluate sentiment agent
        print("\nEvaluating sentiment agent...")
        sentiment_metrics = evaluate_agent(
            sentiment_agent, env, n_eval_episodes
        )
        print(f"Sentiment agent — Return: {sentiment_metrics['mean_return']*100:.2f}% | "
              f"Sharpe: {sentiment_metrics['mean_sharpe']:.4f} | "
              f"Drawdown: {sentiment_metrics['mean_drawdown']*100:.2f}%")

        mlflow.log_metrics({
            "sentiment_mean_return"  : sentiment_metrics["mean_return"],
            "sentiment_mean_sharpe"  : sentiment_metrics["mean_sharpe"],
            "sentiment_mean_drawdown": sentiment_metrics["mean_drawdown"],
        })

        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Checkpoints saved to {ckpt_dir}/")

    return price_agent, sentiment_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train price and sentiment agents")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed; checkpoints go to results/checkpoints/seed{N}/")
    parser.add_argument("--copy-dir", type=str, default=None,
                        help="also copy every checkpoint here (e.g. a Google Drive folder)")
    args = parser.parse_args()

    config = load_config()
    price_agent, sentiment_agent = train(config, seed=args.seed,
                                         copy_dir=args.copy_dir)