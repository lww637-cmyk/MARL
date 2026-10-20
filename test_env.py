"""Quick smoke test for StockTradingEnv."""
import yaml
import numpy as np
from src.environment.stock_env import (
    StockTradingEnv, TICKERS, PRICE_FEATURES, normalise_action,
)

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()
print("Initialising environment...")
env = StockTradingEnv(config, mode="train")

print(f"Agents: {env.possible_agents}")
print(f"Price agent obs size:     {env.observation_space('price_agent').shape}")
print(f"Sentiment agent obs size: {env.observation_space('sentiment_agent').shape}")
print(f"Action space:             {env.action_space('price_agent').shape}")
print(f"Common trading days:      {env.n_steps}")

# Price features are standardised with the train-split scaler and clipped
n_price = len(PRICE_FEATURES) * len(TICKERS)
clip    = env.price_scaler["clip"]
obs = []
for i in range(env.n_steps):
    env.start_idx, env.timestep = i, 0
    obs.append(env.observe("price_agent")[:n_price])
obs = np.array(obs)
print(f"Scaled price features (train split): min {obs.min():.2f} max {obs.max():.2f} "
      f"mean {obs.mean():.3f} std {obs.std():.3f}")
assert np.all(np.abs(obs) <= clip)
assert abs(obs.mean()) < 0.05 and abs(obs.std() - 1.0) < 0.1

# Sentiment observation is not scaled (already in [-1, 1])
env.reset(seed=0)
sent = env.observe("sentiment_agent")
assert np.all(np.abs(sent[:4 * len(TICKERS)]) <= 1.0)

print("\nRunning one episode...")
obs, infos = env.reset(seed=42)

step = 0
while env.agents:
    agent = env.agent_selection
    
    # Pass None if agent is terminated or truncated
    if env.terminations.get(agent, False) or env.truncations.get(agent, False):
        env.step(None)
    else:
        action = env.action_space(agent).sample()
        env.step(action)
        if env.suggestions:
            assert np.allclose(env.suggestions[agent], normalise_action(action))
    step += 1

summary = env.get_episode_summary()
print(f"\nEpisode complete — {summary['n_steps']} steps")
print(f"Total return:  {summary['total_return']*100:.2f}%")
print(f"Sharpe ratio:  {summary['sharpe_ratio']:.4f}")
print(f"Max drawdown:  {summary['max_drawdown']*100:.2f}%")
print(f"Final value:   ${summary['final_value']:,.2f}")
print("\nEnvironment smoke test passed.")
