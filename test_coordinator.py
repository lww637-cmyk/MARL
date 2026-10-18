"""Smoke test for LLM coordinator."""
import numpy as np
import yaml
from src.coordinator.llm_coordinator import coordinate_or_average
from src.environment.stock_env import TICKERS as ENV_TICKERS

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()

# Simulate agent suggestions
price_suggestion = np.array(
    [0.25, 0.10, 0.30, 0.20, 0.10, 0.05], dtype=np.float32
)
sentiment_suggestion = np.array(
    [0.15, 0.05, 0.20, 0.40, 0.15, 0.05], dtype=np.float32
)

context = {
    "portfolio_value"  : 105000,
    "initial_capital"  : 100000,
    "timestep"         : 50,
    "recent_returns"   : [0.001, -0.002, 0.003, 0.001, -0.001],
}

TICKERS = ENV_TICKERS + ["cash"]

print("Testing fallback (no LLM)...")
fallback = coordinate_or_average(
    price_suggestion, sentiment_suggestion,
    context, config, use_llm=False
)
print("Fallback weights:")
for ticker, w in zip(TICKERS, fallback):
    print(f"  {ticker}: {w*100:.1f}%")
print(f"  Sum: {fallback.sum():.4f}")

print("\nTesting LLM coordination (Gemini)...")
llm_weights = coordinate_or_average(
    price_suggestion, sentiment_suggestion,
    context, config, use_llm=True
)
print("LLM weights:")
for ticker, w in zip(TICKERS, llm_weights):
    print(f"  {ticker}: {w*100:.1f}%")
print(f"  Sum: {llm_weights.sum():.4f}")

print("\nCoordinator smoke test passed.")