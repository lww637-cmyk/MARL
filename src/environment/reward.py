"""
Standalone reward functions for the stock trading environment.
Used for analysis and alternative reward shaping experiments.
"""

import numpy as np


def sharpe_reward(returns: np.ndarray,
                  window: int = 30,
                  risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from recent returns window."""
    if len(returns) < 2:
        return 0.0
    r   = returns[-window:]
    mean = r.mean() - risk_free_rate / 252
    std  = r.std() + 1e-8
    return float((mean / std) * np.sqrt(252))


def drawdown_penalty(portfolio_values: np.ndarray,
                     threshold: float = 0.1) -> float:
    """Penalty for drawdowns exceeding threshold."""
    if len(portfolio_values) < 2:
        return 0.0
    peak     = portfolio_values.max()
    current  = portfolio_values[-1]
    drawdown = (peak - current) / (peak + 1e-8)
    return float(max(0.0, drawdown - threshold) * 2.0)


def combined_reward(returns: np.ndarray,
                    portfolio_values: np.ndarray,
                    config: dict) -> float:
    """Combine Sharpe reward and drawdown penalty."""
    sharpe  = sharpe_reward(returns)
    penalty = drawdown_penalty(portfolio_values)
    return sharpe - penalty