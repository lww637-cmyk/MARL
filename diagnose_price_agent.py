"""
TASK4 diagnostics for the price-agent collapse (train split only).

  checkpoints : allocation mean/std over the train split, policy log_std and
                mean |action| before clipping for every price-agent checkpoint
                in a folder (e.g. the seed 0 checkpoints on Drive).
  pilot       : pilot runs of one variant on top of the CURRENT project setup
                (control = project defaults as they are now; vecnorm / softmax /
                both add those changes). Does not change any project default.
                TASK4 ran these before the TASK5 fix (raw features, random partner).

  python diagnose_price_agent.py checkpoints /content/drive/MyDrive/MARL/results/checkpoints_scaled/seed0
  (add --any-setup to inspect checkpoints from an older training setup)
  python diagnose_price_agent.py pilot control --seed 99 --steps 500000 --every 50000
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.base_agent import BaseTrader
from src.checkpoints import check_setup, training_setup
from src.environment.stock_env import StockTradingEnv, TICKERS, normalise_action
from src.training.train_marl import load_config, set_global_seed

AGENT = "price_agent"
COLS  = TICKERS + ["cash"]


def train_split_obs(config: dict) -> np.ndarray:
    """Price observation for every train-split day (same as the notebook policy check)."""
    env = StockTradingEnv(config, mode="eval", split="train")
    env.reset(seed=0)
    obs = []
    for i in range(env.n_steps):
        env.start_idx, env.timestep = i, 0
        obs.append(env.observe(AGENT))
    return np.array(obs, dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def policy_stats(model, obs: np.ndarray, use_softmax: bool = False,
                 vecnorm: VecNormalize = None) -> dict:
    """Deterministic allocation over obs plus the raw Gaussian mean / log_std."""
    if vecnorm is not None:
        obs = vecnorm.normalize_obs(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(obs))
        mu = dist.distribution.mean.numpy()
    if use_softmax:
        w = softmax(mu)
    else:
        w = np.array([normalise_action(a) for a in mu])
    return {
        "alloc_mean"   : w.mean(axis=0),
        "alloc_std"    : w.std(axis=0),
        "alloc_std_avg": float(w.std(axis=0).mean()),
        "log_std"      : model.policy.log_std.detach().numpy().copy(),
        "mu_mean"      : mu.mean(axis=0),
        "mu_std"       : mu.std(axis=0),
        "mean_abs_mu"  : float(np.abs(mu).mean()),
        "frac_outside" : float(((mu < 0) | (mu > 1)).mean()),
    }


def fmt(v) -> str:
    return " ".join(f"{x:+.3f}" for x in v)


# ── B: checkpoint analysis ──────────────────────────────────────────────────

def analyse_checkpoints(ckpt_dir: str, config: dict, any_setup: bool = False):
    if not any_setup:
        check_setup(ckpt_dir, training_setup(config))
    obs = train_split_obs(config)
    pattern = re.compile(rf"^{AGENT}_step(\d+)\.zip$")
    found = []
    for name in os.listdir(ckpt_dir):
        m = pattern.match(name)
        if m:
            found.append((int(m.group(1)), name))
        elif name == f"{AGENT}_final.zip":
            found.append((10**12, name))
    print(f"{len(obs)} train days | columns: {COLS}")
    for step, name in sorted(found):
        model = PPO.load(os.path.join(ckpt_dir, name), device="cpu")
        s = policy_stats(model, obs)
        print(f"\n{name}  (num_timesteps={model.num_timesteps:,})")
        print(f"  alloc mean   {fmt(s['alloc_mean'])}")
        print(f"  alloc std    {fmt(s['alloc_std'])}   avg {s['alloc_std_avg']:.4f}")
        print(f"  mu mean      {fmt(s['mu_mean'])}")
        print(f"  mu std       {fmt(s['mu_std'])}")
        print(f"  log_std      {fmt(s['log_std'])}")
        print(f"  mean |mu| {s['mean_abs_mu']:.3f} | share of mu outside [0,1] {s['frac_outside']:.3f}")


# ── C: pilot runs ───────────────────────────────────────────────────────────

class SoftmaxAction(gym.ActionWrapper):
    """
    Agent outputs logits; the env receives softmax weights. SB3 needs finite
    bounds, so logits are boxed at +-20 (softmax is saturated long before).
    """

    def __init__(self, env):
        super().__init__(env)
        n = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(-20.0, 20.0, (n,), np.float32)

    def action(self, logits):
        return softmax(np.asarray(logits, dtype=np.float64)).astype(np.float32)


def make_vec_env(config: dict, use_softmax: bool, use_vecnorm: bool):
    trader = BaseTrader.__new__(BaseTrader)
    trader.agent_id, trader.config = AGENT, config
    # the project's own SingleAgentWrapper (random partner, same reward)
    inner = trader._make_single_env().envs[0].env
    if use_softmax:
        inner = SoftmaxAction(inner)
    venv = DummyVecEnv([lambda: Monitor(inner)])
    if use_vecnorm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    return venv


class PilotCallback(BaseCallback):
    """Every `every` steps: policy stats on the train split; tracks step rewards."""

    def __init__(self, obs, use_softmax, every=25000):
        super().__init__()
        self.obs, self.use_softmax, self.every = obs, use_softmax, every
        self.rewards, self.history = [], []

    def _snapshot(self):
        vn = self.model.get_vec_normalize_env()
        s = policy_stats(self.model, self.obs, self.use_softmax, vn)
        row = {"step": self.num_timesteps,
               "alloc_std_avg": s["alloc_std_avg"],
               "mean_abs_mu": s["mean_abs_mu"],
               "frac_outside": s["frac_outside"],
               "log_std_mean": float(s["log_std"].mean()),
               "reward_last_window": float(np.mean(self.rewards[-self.every:]))}
        self.history.append(row)
        print(json.dumps(row), flush=True)

    def _on_step(self):
        self.rewards.append(float(self.locals["rewards"][0]))
        if self.num_timesteps % self.every == 0:
            self._snapshot()
        return True


VARIANTS = {"control": (False, False), "vecnorm": (True, False),
            "softmax": (False, True), "both": (True, True)}


def pilot(variant: str, seed: int, steps: int, out_dir: str, config: dict,
          every: int = 25000, window: int = 20000):
    use_vecnorm, use_softmax = VARIANTS[variant]
    set_global_seed(seed)
    venv = make_vec_env(config, use_softmax, use_vecnorm)
    # same PPO settings as BaseTrader.build_model
    model = PPO("MlpPolicy", venv,
                learning_rate=config["agents"][AGENT]["learning_rate"],
                n_steps=512, batch_size=config["agents"][AGENT]["batch_size"],
                gamma=config["agents"][AGENT]["gamma"], gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, device="cpu", verbose=0, seed=seed)
    obs = train_split_obs(config)
    cb = PilotCallback(obs, use_softmax, every)
    model.learn(total_timesteps=steps, callback=cb)

    os.makedirs(out_dir, exist_ok=True)
    model.save(f"{out_dir}/{AGENT}_{variant}_seed{seed}")
    if use_vecnorm:
        venv.save(f"{out_dir}/{AGENT}_{variant}_seed{seed}_vecnormalize.pkl")

    final = policy_stats(model, obs, use_softmax, model.get_vec_normalize_env())
    r = np.array(cb.rewards)
    summary = {
        "variant": variant, "seed": seed, "steps": steps,
        "alloc_std_avg": final["alloc_std_avg"],
        "alloc_mean": final["alloc_mean"].round(3).tolist(),
        "alloc_std": final["alloc_std"].round(3).tolist(),
        "log_std": final["log_std"].round(3).tolist(),
        "mean_abs_mu": final["mean_abs_mu"],
        "frac_outside": final["frac_outside"],
        "partner": config["training"].get("partner", "neutral"),
        "reward_window": window,
        "reward_first": float(r[:window].mean()),
        "reward_last": float(r[-window:].mean()),
        "reward_all": float(r.mean()),
        "history": cb.history,
    }
    with open(f"{out_dir}/{variant}_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "history"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoints")
    c.add_argument("ckpt_dir")
    c.add_argument("--any-setup", action="store_true",
                   help="skip the training_setup.json check (old checkpoints)")
    q = sub.add_parser("pilot")
    q.add_argument("variant", choices=list(VARIANTS))
    q.add_argument("--seed", type=int, default=99)
    q.add_argument("--steps", type=int, default=200000)
    q.add_argument("--every", type=int, default=25000, help="snapshot interval")
    q.add_argument("--window", type=int, default=20000, help="first/last reward window")
    q.add_argument("--out-dir", default="results/pilot_task4")
    args = p.parse_args()

    config = load_config()
    if args.cmd == "checkpoints":
        analyse_checkpoints(args.ckpt_dir, config, args.any_setup)
    else:
        pilot(args.variant, args.seed, args.steps, args.out_dir, config,
              args.every, args.window)
