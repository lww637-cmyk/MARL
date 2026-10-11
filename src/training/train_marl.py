"""
Trains both price and sentiment agents within the shared stock trading
environment. Logs training metrics to MLflow.
Designed to run on Google Colab with GPU acceleration.
"""

import os
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


# ── custom callback ──────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Logs episode rewards and portfolio metrics to MLflow
    during training. Saves checkpoints at regular intervals.
    """

    def __init__(self, agent_id: str, config: dict,
                 save_freq: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_id  = agent_id
        self.config    = config
        self.save_freq = save_freq
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
            os.makedirs("results/checkpoints", exist_ok=True)
            path = f"results/checkpoints/{self.agent_id}_step{self.n_calls}"
            self.model.save(path)
            print(f"Checkpoint saved: {path}")

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

def train(config: dict):
    """
    Full training pipeline:
    1. Initialise environment and agents
    2. Train price agent
    3. Train sentiment agent
    4. Evaluate both agents
    5. Save final models
    """
    total_timesteps = config["training"]["total_timesteps"]
    eval_freq       = config["training"]["eval_frequency"]
    save_freq       = config["training"]["save_frequency"]
    n_eval_episodes = config["training"]["n_eval_episodes"]

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # Initialise environment
    print("Initialising environment...")
    env = StockTradingEnv(config, mode="train")
    print(f"Environment ready — {env.n_steps} trading days available")

    # Start MLflow run
    mlflow.set_experiment("agentic_marl_trading")

    with mlflow.start_run(run_name="marl_training"):

        # Log config params
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("episode_length",  config["environment"]["episode_length"])
        mlflow.log_param("initial_capital", config["environment"]["initial_capital"])
        mlflow.log_param("tickers",         config["data"]["tickers"])

        # ── Train Price Agent ──────────────────────────────────────────────
        print("\n" + "="*50)
        print("TRAINING PRICE AGENT")
        print("="*50)

        price_agent = PriceAgent(env, config)
        price_callback = TrainingCallback(
            "price_agent", config, save_freq=save_freq
        )

        price_agent.model.learn(
            total_timesteps = total_timesteps,
            callback        = price_callback,
            progress_bar    = True,
        )

        price_agent.save("results/checkpoints/price_agent_final.zip")
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
            "sentiment_agent", config, save_freq=save_freq
        )

        sentiment_agent.model.learn(
            total_timesteps = total_timesteps,
            callback        = sentiment_callback,
            progress_bar    = True,
        )

        sentiment_agent.save("results/checkpoints/sentiment_agent_final.zip")
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
        print(f"Checkpoints saved to results/checkpoints/")

    return price_agent, sentiment_agent


if __name__ == "__main__":
    config = load_config()
    price_agent, sentiment_agent = train(config)