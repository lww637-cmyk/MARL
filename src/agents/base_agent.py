"""
Base class for all RL trading agents.
Defines shared interface for training, prediction, saving and loading.
Both PriceAgent and SentimentAgent inherit from this class.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class BaseTrader:
    def __init__(self, env, agent_id: str, config: dict):
        """
        env      : the StockTradingEnv instance
        agent_id : 'price_agent' or 'sentiment_agent'
        config   : full project config dict
        """
        self.env      = env
        self.agent_id = agent_id
        self.config   = config
        self.model    = None

        # Training config
        self.lr          = config["agents"][agent_id]["learning_rate"]
        self.batch_size  = config["agents"][agent_id]["batch_size"]
        self.gamma       = config["agents"][agent_id]["gamma"]
        # Optional training seed (None = unseeded, previous behaviour)
        self.seed        = config.get("training", {}).get("seed")

    def _make_single_env(self):
        """
        Creates a single-agent Gym wrapper for SB3 training.
        Bypasses PettingZoo AEC turn structure to avoid dead agent conflicts.
        Each wrapper creates its own environment instance.
        The agent's action is mapped to weights exactly as in StockTradingEnv.step
        (normalise_action). The other agent is a fixed partner set by
        config training.partner: "neutral" (equal weights, default) or "random".
        """
        import gymnasium as gym
        from src.environment.stock_env import (
            StockTradingEnv, PARTNERS, normalise_action, partner_action
        )
        agent_id = self.agent_id
        config   = self.config
        partner  = config.get("training", {}).get("partner", "neutral")
        if partner not in PARTNERS:
            raise ValueError(
                f"training.partner must be one of {PARTNERS}, got {partner!r}"
            )

        class SingleAgentWrapper(gym.Env):
            def __init__(self):
                super().__init__()
                self._env         = StockTradingEnv(config, mode="train")
                self.observation_space = self._env.observation_space(agent_id)
                self.action_space      = self._env.action_space(agent_id)
                self._step_count  = 0
                self._max_steps   = config["environment"]["episode_length"]

            def reset(self, seed=None, options=None):
                self._env.reset(seed=seed)
                self._step_count = 0
                return self._env.observe(agent_id), {}

            def step(self, action):
                self._step_count += 1

                # Store our agent's suggestion, normalised as in env.step
                self._env.suggestions[agent_id] = normalise_action(action)

                # Other agent's suggestion: fixed partner (neutral or random)
                other = [
                    a for a in self._env.possible_agents
                    if a != agent_id
                ][0]
                self._env.suggestions[other] = partner_action(
                    partner, self._env.action_space(other)
                )

                # Execute portfolio step directly (bypass AEC turns)
                self._env._execute_step()
                self._env.timestep += 1
                self._env.suggestions = {}

                reward     = self._env.rewards.get(agent_id, 0.0)
                terminated = self._step_count >= self._max_steps
                truncated  = False
                info       = self._env.infos.get(agent_id, {})

                if terminated:
                    obs = np.zeros(
                        self.observation_space.shape, dtype=np.float32
                    )
                else:
                    obs = self._env.observe(agent_id)

                return obs, reward, terminated, truncated, info

            def render(self):
                pass

            def close(self):
                pass

        def make():
            return Monitor(SingleAgentWrapper())

        return DummyVecEnv([make])

    def build_model(self):
        """Initialise PPO model. Called by subclasses after super().__init__."""
        vec_env    = self._make_single_env()
        self.model = PPO(
            policy        = "MlpPolicy",
            env           = vec_env,
            learning_rate = self.lr,
            n_steps       = 512,
            batch_size    = self.batch_size,
            gamma         = self.gamma,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            device        = "cpu",
            verbose       = 0,
            seed          = self.seed,
        )

    def train(self, total_timesteps: int):
        """Train the PPO model."""
        if self.model is None:
            raise RuntimeError("Call build_model() before train().")
        print(f"Training {self.agent_id} for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print(f"{self.agent_id} training complete.")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Return portfolio weights for the given observation: the deterministic
        action mapped with normalise_action, exactly as in training.
        """
        from src.environment.stock_env import normalise_action
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        obs = np.array(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return normalise_action(np.array(action, dtype=np.float32).flatten())

    def save(self, path: str = None):
        """Save model weights to results/checkpoints/."""
        if path is None:
            os.makedirs("results/checkpoints", exist_ok=True)
            path = f"results/checkpoints/{self.agent_id}.zip"
        self.model.save(path)
        print(f"Saved {self.agent_id} → {path}")

    def load(self, path: str = None):
        """Load model weights from path."""
        if path is None:
            path = f"results/checkpoints/{self.agent_id}.zip"
        self.model = PPO.load(path)
        print(f"Loaded {self.agent_id} from {path}")