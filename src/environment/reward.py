"""
Defines reward functions for the MARL trading environment.
Supports risk-adjusted returns (Sharpe-based) and
penalises excessive drawdown.
"""

import numpy as np

def sharpe_reward(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    # TODO: compute Sharpe ratio as reward signal
    pass

def drawdown_penalty(portfolio_values: np.ndarray) -> float:
    # TODO: compute and return drawdown penalty to discourage large losses
    pass

def combined_reward(returns: np.ndarray, portfolio_values: np.ndarray, config: dict) -> float:
    # TODO: combine Sharpe reward and drawdown penalty with configurable weights
    pass