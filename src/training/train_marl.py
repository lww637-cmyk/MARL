"""
Trains both price and sentiment agents within the shared stock trading
environment. Logs training metrics to MLflow.
Designed to run on Google Colab with GPU acceleration.
"""

import os
import re
import json
import random
import shutil
import argparse
import yaml
import numpy as np
import mlflow
import mlflow.pytorch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.stock_env import (
    StockTradingEnv, partner_action
)
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.checkpoints import (
    SETUP_FILE, AGENT_IDS, training_setup, check_setup, checkpoint_dir,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mirror_file(path: str, copy_dir: str = None):
    """Copy a saved file into copy_dir (e.g. Google Drive) if one is given."""
    if copy_dir is None:
        return
    os.makedirs(copy_dir, exist_ok=True)
    # copy under a temporary name first so a disconnect never leaves a
    # half-written checkpoint under the real name
    dst = os.path.join(copy_dir, os.path.basename(path))
    shutil.copy(path, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"Copied {path} → {copy_dir}/")


def check_training_setup(setup: dict, dirs: list):
    """
    Refuse to resume checkpoints from a different setup (e.g. the old unscaled
    price features or random partner) in any of dirs, then record setup there.
    """
    for d in dirs:
        if d is None or not os.path.isdir(d):
            continue
        if any(find_checkpoints(a, [d], 0) for a in AGENT_IDS):
            check_setup(d, setup)
    for d in dirs:
        if d is None:
            continue
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, SETUP_FILE), "w") as f:
            json.dump(setup, f, indent=1)


def find_checkpoints(agent_id: str, dirs: list,
                     total_timesteps: int) -> list:
    """
    Checkpoints for agent_id in dirs, newest first, as (step, path).
    {agent_id}_final.zip counts as step total_timesteps.
    """
    pattern = re.compile(rf"^{agent_id}_step(\d+)\.zip$")
    found = {}
    for d in dirs:
        if d is None or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = pattern.match(name)
            if m:
                step = int(m.group(1))
            elif name == f"{agent_id}_final.zip":
                step = total_timesteps
            else:
                continue
            found.setdefault(step, os.path.join(d, name))
    return sorted(found.items(), reverse=True)


def load_latest_checkpoint(agent, dirs: list, total_timesteps: int) -> int:
    """
    Replace agent.model with the newest loadable checkpoint in dirs
    (unreadable files are skipped). Returns the step resumed from, 0 if none.
    """
    for step, path in find_checkpoints(agent.agent_id, dirs, total_timesteps):
        try:
            model = PPO.load(path, env=agent.model.get_env(), device="cpu")
        except Exception as e:
            print(f"WARNING: could not load {path} ({e}); trying an older checkpoint")
            continue
        agent.model = model
        print(f"Resuming {agent.agent_id} from step {model.num_timesteps:,} ({path})")
        return model.num_timesteps
    print(f"No checkpoint for {agent.agent_id} — training from step 0")
    return 0


def learn_with_resume(agent, total_timesteps: int, callback,
                      dirs: list) -> int:
    """
    Train agent up to total_timesteps, continuing from the newest checkpoint
    in dirs if there is one. Returns the step it resumed from.
    """
    start = load_latest_checkpoint(agent, dirs, total_timesteps)
    remaining = total_timesteps - start
    if remaining <= 0:
        print(f"{agent.agent_id} already trained to {start:,} steps — skipping")
        return start
    agent.model.learn(
        total_timesteps     = remaining,
        callback            = callback,
        progress_bar        = True,
        reset_num_timesteps = (start == 0),
    )
    return start


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

        # Log to MLflow every 1000 steps. Uses the model's total step count
        # (not n_calls) so steps stay correct after resuming from a checkpoint.
        if self.num_timesteps % 1000 == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mlflow.log_metric(
                f"{self.agent_id}_mean_reward",
                mean_reward,
                step=self.num_timesteps
            )

        # Save checkpoint
        if self.num_timesteps % self.save_freq == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            path = f"{self.ckpt_dir}/{self.agent_id}_step{self.num_timesteps}"
            self.model.save(path)
            print(f"Checkpoint saved: {path}")
            mirror_file(f"{path}.zip", self.copy_dir)

        return True


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_agent(agent, env: StockTradingEnv,
                   n_episodes: int = 5) -> dict:
    """
    Run agent for n_episodes in eval mode and return
    average performance metrics. The other agent is the same
    fixed partner as in training (config training.partner).
    """
    agent_id = agent.agent_id
    partner  = agent.config["training"].get("partner", "neutral")
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
                action = partner_action(partner, env.action_space(current))

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
           and checkpoints go to {paths.checkpoints_root}/seed{seed}/.
    copy_dir : optional folder (e.g. on Google Drive) that every checkpoint
               and final model is copied to as soon as it is saved.
               If it (or the local checkpoint dir) already holds checkpoints
               for an agent, training resumes from the newest one.
    """
    total_timesteps = config["training"]["total_timesteps"]
    eval_freq       = config["training"]["eval_frequency"]
    save_freq       = config["training"]["save_frequency"]
    n_eval_episodes = config["training"]["n_eval_episodes"]

    if seed is not None:
        set_global_seed(seed)
        config["training"]["seed"] = seed   # read by BaseTrader -> PPO(seed=...)
    ckpt_dir = checkpoint_dir(config, seed)
    # Resume from the newest checkpoint for this seed (Drive copy first)
    resume_dirs = [copy_dir, ckpt_dir]

    os.makedirs(ckpt_dir, exist_ok=True)
    check_training_setup(training_setup(config), resume_dirs)
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
        mlflow.log_param("partner",         config["training"].get("partner", "neutral"))

        # ── Train Price Agent ──────────────────────────────────────────────
        print("\n" + "="*50)
        print("TRAINING PRICE AGENT")
        print("="*50)

        price_agent = PriceAgent(env, config)
        price_callback = TrainingCallback(
            "price_agent", config, save_freq=save_freq, ckpt_dir=ckpt_dir,
            copy_dir=copy_dir
        )

        price_start = learn_with_resume(
            price_agent, total_timesteps, price_callback, resume_dirs
        )
        mlflow.log_param("price_resumed_from_step", price_start)

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

        sentiment_start = learn_with_resume(
            sentiment_agent, total_timesteps, sentiment_callback, resume_dirs
        )
        mlflow.log_param("sentiment_resumed_from_step", sentiment_start)

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
                        help="training seed; checkpoints go to "
                             "{paths.checkpoints_root}/seed{N}/")
    parser.add_argument("--copy-dir", type=str, default=None,
                        help="also copy every checkpoint here (e.g. a Google Drive folder)")
    args = parser.parse_args()

    config = load_config()
    price_agent, sentiment_agent = train(config, seed=args.seed,
                                         copy_dir=args.copy_dir)