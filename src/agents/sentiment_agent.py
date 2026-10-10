"""
Sentiment agent: observes company-specific FinBERT sentiment scores
for all 5 tickers extracted from financial news headlines.
Outputs suggested portfolio weights based on sentiment signals only.
Trained using PPO from stable-baselines3.
"""

from src.agents.base_agent import BaseTrader


class SentimentAgent(BaseTrader):
    def __init__(self, env, config: dict):
        super().__init__(env, "sentiment_agent", config)
        self.build_model()
        print(f"SentimentAgent initialised | obs={env.observation_space('sentiment_agent').shape} | action={env.action_space('sentiment_agent').shape}")