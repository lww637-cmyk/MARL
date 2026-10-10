"""Smoke test for PriceAgent and SentimentAgent initialisation."""
import yaml
from src.environment.stock_env import StockTradingEnv
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()

print("Initialising environment...")
env = StockTradingEnv(config, mode="train")

print("\nInitialising PriceAgent...")
price_agent = PriceAgent(env, config)

print("\nInitialising SentimentAgent...")
sentiment_agent = SentimentAgent(env, config)

print("\nTesting predictions...")
obs, _ = env.reset(seed=42)

price_obs = env.observe("price_agent")
sent_obs  = env.observe("sentiment_agent")

price_action = price_agent.predict(price_obs)
sent_action  = sentiment_agent.predict(sent_obs)

print(f"Price agent action:     {price_action.round(4)}")
print(f"Sentiment agent action: {sent_action.round(4)}")
print(f"Price weights sum:      {price_action.sum():.4f}")
print(f"Sentiment weights sum:  {sent_action.sum():.4f}")

print("\nAgent smoke test passed.")