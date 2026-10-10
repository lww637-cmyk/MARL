"""
Price agent: observes OHLCV and technical indicators
(RSI, MACD, Bollinger Bands) for all 5 tickers.
Outputs suggested portfolio weights based on price signals only.
Trained using PPO from stable-baselines3.
"""

from src.agents.base_agent import BaseTrader


class PriceAgent(BaseTrader):
    def __init__(self, env, config: dict):
        super().__init__(env, "price_agent", config)
        self.build_model()
        print(f"PriceAgent initialised | obs={env.observation_space('price_agent').shape} | action={env.action_space('price_agent').shape}")