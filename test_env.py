"""Quick smoke test for StockTradingEnv."""
import yaml
import numpy as np
from src.environment.stock_env import StockTradingEnv

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
    step += 1

summary = env.get_episode_summary()
print(f"\nEpisode complete — {summary['n_steps']} steps")
print(f"Total return:  {summary['total_return']*100:.2f}%")
print(f"Sharpe ratio:  {summary['sharpe_ratio']:.4f}")
print(f"Max drawdown:  {summary['max_drawdown']*100:.2f}%")
print(f"Final value:   ${summary['final_value']:,.2f}")
print("\nEnvironment smoke test passed.")