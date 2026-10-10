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

    def _make_single_env(self):
        """
        Wraps the trading env into a single-agent Gym-compatible env
        so stable-baselines3 can train one agent at a time.
        """
        import gymnasium as gym
        agent_id = self.agent_id
        aec_env  = self.env

        class SingleAgentWrapper(gym.Env):
            """Adapts PettingZoo AEC env to single-agent Gym interface."""

            def __init__(self):
                super().__init__()
                self.aec_env           = aec_env
                self.observation_space = aec_env.observation_space(agent_id)
                self.action_space      = aec_env.action_space(agent_id)

            def reset(self, seed=None, options=None):
                obs, info = self.aec_env.reset(seed=seed)
                return self.aec_env.observe(agent_id), info

            def step(self, action):
                # Step our agent
                self.aec_env.step(action)
                reward     = self.aec_env.rewards.get(agent_id, 0.0)
                terminated = self.aec_env.terminations.get(agent_id, False)
                truncated  = self.aec_env.truncations.get(agent_id, False)
                info       = self.aec_env.infos.get(agent_id, {})

                # Step other agent with random action
                if self.aec_env.agents and \
                self.aec_env.agent_selection != agent_id:
                    other = self.aec_env.agent_selection
                    if not self.aec_env.terminations.get(other, False) and \
                    not self.aec_env.truncations.get(other, False):
                        other_action = self.aec_env.action_space(other).sample()
                        self.aec_env.step(other_action)

                return self.aec_env.observe(agent_id), reward, terminated, truncated, info

        def render(self):
            pass

        def close(self):
            pass

        def make():
            return Monitor(SingleAgentWrapper())

        return DummyVecEnv([make])

    def build_model(self):
        """Initialise PPO model. Called by subclasses after super().__init__."""
        vec_env     = self._make_single_env()
        self.model  = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = self.lr,
            n_steps         = 512,
            batch_size      = self.batch_size,
            gamma           = self.gamma,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            verbose         = 0,
        )

    def train(self, total_timesteps: int):
        """Train the PPO model."""
        if self.model is None:
            raise RuntimeError("Call build_model() before train().")
        print(f"Training {self.agent_id} for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print(f"{self.agent_id} training complete.")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return action for given observation."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        obs = np.array(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array(action, dtype=np.float32).flatten()

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