"""
Custom multi-agent trading environment for AAPL, AMZN, MSFT, NVDA, TSLA.
Built on PettingZoo AEC API.
Two agents share a portfolio: price agent and sentiment agent.
LLM coordinator makes the final allocation decision each timestep.
"""

import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import yaml
import os

# ── constants ────────────────────────────────────────────────────────────────
TICKERS    = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
N_TICKERS  = len(TICKERS)

# Price agent features per ticker
PRICE_FEATURES = [
    "close", "volume", "rsi", "macd",
    "macd_signal", "macd_diff",
    "bb_upper", "bb_lower", "bb_width", "daily_return"
]

# Sentiment agent features per ticker
SENTIMENT_FEATURES = [
    "sentiment_score", "sentiment_std",
    "headline_count_norm", "has_sentiment"
]

N_PRICE_FEATURES     = len(PRICE_FEATURES)      # 10
N_SENTIMENT_FEATURES = len(SENTIMENT_FEATURES)  # 4
N_SHARED             = N_TICKERS + 1            # portfolio weights + norm value

# Total obs sizes
PRICE_OBS_SIZE     = N_PRICE_FEATURES * N_TICKERS + N_SHARED   # 50 + 6 = 56
SENTIMENT_OBS_SIZE = N_SENTIMENT_FEATURES * N_TICKERS + N_SHARED  # 20 + 6 = 26

# Action = portfolio weights for 5 tickers + cash = 6 values
ACTION_SIZE = N_TICKERS + 1


# ── data loader ──────────────────────────────────────────────────────────────

def load_price_data() -> dict:
    """Load preprocessed price data for all tickers."""
    data = {}
    for ticker in TICKERS:
        path = f"data/processed/{ticker}_processed.csv"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        data[ticker] = df
    return data


def load_sentiment_data() -> dict:
    """
    Load sentiment data for all tickers.
    Forward-fills missing trading days and adds has_sentiment flag.
    """
    data = {}
    for ticker in TICKERS:
        path = f"data/sentiment/{ticker}_sentiment.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            data[ticker] = df
        else:
            data[ticker] = None
    return data


def build_merged_dataset(price_data: dict,
                          sentiment_data: dict,
                          episode_length: int = 252) -> dict:
    """
    Align price and sentiment data on the same daily index.
    Forward-fills missing sentiment days.
    Adds has_sentiment flag (1 = fresh signal, 0 = forward-filled).
    Returns dict of {ticker: merged_dataframe}.
    """
    merged = {}

    for ticker in TICKERS:
        price_df = price_data[ticker].copy()
        sent_df  = sentiment_data.get(ticker)

        if sent_df is not None:
            # Mark which dates have real sentiment
            sent_df = sent_df.copy()
            sent_df["has_sentiment"] = 1.0

            # Merge on price index (left join — keeps all trading days)
            merged_df = price_df.join(sent_df, how="left")
            merged_df["has_sentiment"] = merged_df["has_sentiment"].fillna(0.0)

            # Forward-fill sentiment scores
            merged_df["sentiment_score"] = (
                merged_df["sentiment_score"].ffill().fillna(0.0)
            )
            merged_df["sentiment_std"] = (
                merged_df["sentiment_std"].ffill().fillna(0.0)
            )
            merged_df["headline_count"] = (
                merged_df["headline_count"].ffill().fillna(0.0)
            )

            # Normalise headline count to 0-1
            max_count = merged_df["headline_count"].max()
            merged_df["headline_count_norm"] = (
                merged_df["headline_count"] / max_count
                if max_count > 0 else 0.0
            )
        else:
            # No sentiment data — fill with zeros
            merged_df = price_df.copy()
            merged_df["sentiment_score"]    = 0.0
            merged_df["sentiment_std"]      = 0.0
            merged_df["headline_count_norm"] = 0.0
            merged_df["has_sentiment"]       = 0.0

        merged[ticker] = merged_df.dropna()

    return merged


# ── environment ──────────────────────────────────────────────────────────────

class StockTradingEnv(AECEnv):
    """
    PettingZoo AEC environment for two-agent stock trading.

    Agents:
        price_agent    — observes price + technical features
        sentiment_agent — observes company-specific sentiment features

    Each timestep both agents act, then the LLM coordinator
    (called externally) combines their suggestions into a
    final portfolio allocation.
    """

    metadata = {"render_modes": ["human"], "name": "stock_trading_v0"}

    def __init__(self, config: dict, mode: str = "train"):
        super().__init__()

        self.config          = config
        self.mode            = mode   # "train" or "eval"
        self.initial_capital = config["environment"]["initial_capital"]
        self.episode_length  = config["environment"]["episode_length"]
        self.transaction_cost = config["environment"]["transaction_cost"]

        # Load and merge data
        price_data     = load_price_data()
        sentiment_data = load_sentiment_data()
        self.dataset   = build_merged_dataset(
            price_data, sentiment_data, self.episode_length
        )

        # Common date index (intersection across all tickers)
        common_dates = None
        for ticker in TICKERS:
            dates = set(self.dataset[ticker].index)
            common_dates = dates if common_dates is None else common_dates & dates
        self.common_dates = sorted(list(common_dates))
        self.n_steps      = len(self.common_dates)

        # PettingZoo required attributes
        self.possible_agents = ["price_agent", "sentiment_agent"]
        self.agents          = self.possible_agents[:]

        self.action_spaces = {
            agent: spaces.Box(
                low=0.0, high=1.0,
                shape=(ACTION_SIZE,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            "price_agent": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(PRICE_OBS_SIZE,),
                dtype=np.float32
            ),
            "sentiment_agent": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(SENTIMENT_OBS_SIZE,),
                dtype=np.float32
            ),
        }

        # Agent selector handles turn order
        self._agent_selector  = agent_selector(self.possible_agents)
        self.agent_selection  = self.possible_agents[0]

        # Internal state (initialised in reset)
        self.timestep         = 0
        self.start_idx        = 0
        self.portfolio_value  = self.initial_capital
        self.portfolio_weights = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # start 100% in cash

        self.suggestions      = {}   # stores agent action suggestions
        self.rewards          = {a: 0.0 for a in self.possible_agents}
        self.terminations     = {a: False for a in self.possible_agents}
        self.truncations      = {a: False for a in self.possible_agents}
        self.infos            = {a: {} for a in self.possible_agents}

        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset environment to start of a new episode."""
        if seed is not None:
            np.random.seed(seed)

        # Randomly sample episode start for training,
        # use fixed start for evaluation
        max_start = self.n_steps - self.episode_length - 1
        if self.mode == "train":
            self.start_idx = np.random.randint(0, max(1, max_start))
        else:
            self.start_idx = 0

        self.timestep          = 0
        self.portfolio_value   = self.initial_capital
        self.portfolio_weights = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # 100% cash

        self.portfolio_history = [self.initial_capital]
        self.returns_history   = []
        self.suggestions       = {}

        self.agents              = self.possible_agents[:]
        self._agent_selector     = agent_selector(self.possible_agents)
        self.agent_selection     = self._agent_selector.reset()

        self.rewards             = {a: 0.0 for a in self.possible_agents}
        self.terminations        = {a: False for a in self.possible_agents}
        self.truncations         = {a: False for a in self.possible_agents}
        self.infos               = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        return self.observe(self.agent_selection), self.infos

    # ── observe ──────────────────────────────────────────────────────────────

    def observe(self, agent: str) -> np.ndarray:
        """Return current observation for the given agent."""
        date = self.common_dates[self.start_idx + self.timestep]
        shared = np.append(self.portfolio_weights,
                           self.portfolio_value / self.initial_capital)

        if agent == "price_agent":
            features = []
            for ticker in TICKERS:
                row = self.dataset[ticker].loc[date]
                features.extend([row[f] for f in PRICE_FEATURES])
            return np.array(features + shared.tolist(),
                            dtype=np.float32)

        elif agent == "sentiment_agent":
            features = []
            for ticker in TICKERS:
                row = self.dataset[ticker].loc[date]
                features.extend([row[f] for f in SENTIMENT_FEATURES])
            return np.array(features + shared.tolist(),
                            dtype=np.float32)

        return np.zeros(PRICE_OBS_SIZE, dtype=np.float32)

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """
        Process one agent's action (portfolio weight suggestion).
        After both agents have acted, execute the combined decision.
        """
        if self.terminations[self.agent_selection] or \
           self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        # Normalise action to valid portfolio weights (sum to 1)
        action = np.clip(action, 0.0, 1.0)
        action = action / (action.sum() + 1e-8)

        # Store this agent's suggestion
        self.suggestions[self.agent_selection] = action

        # If both agents have acted — execute the step
        if len(self.suggestions) == len(self.possible_agents):
            self._execute_step()
            self.suggestions = {}
            self.timestep   += 1

        # Advance agent selector
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[self.agent_selection] += \
            self.rewards[self.agent_selection]

    # ── execute step ─────────────────────────────────────────────────────────

    def _execute_step(self):
        """
        Execute portfolio rebalancing and compute reward.
        Uses simple average of both agents' suggestions.
        LLM coordinator overrides this externally when active.
        """
        # Default: average both agent suggestions
        final_weights = np.mean(
            list(self.suggestions.values()), axis=0
        )
        final_weights = final_weights / (final_weights.sum() + 1e-8)

        # Get current and next date
        curr_date = self.common_dates[self.start_idx + self.timestep]
        next_idx  = self.start_idx + self.timestep + 1

        if next_idx >= self.n_steps:
            # Episode end
            for agent in self.possible_agents:
                self.terminations[agent] = True
            return

        next_date = self.common_dates[next_idx]

        # Compute portfolio return
        ticker_returns = []
        for i, ticker in enumerate(TICKERS):
            curr_close = self.dataset[ticker].loc[curr_date, "close"]
            next_close = self.dataset[ticker].loc[next_date, "close"]
            ticker_returns.append(
                (next_close - curr_close) / (curr_close + 1e-8)
            )

        ticker_returns = np.array(ticker_returns)

        # Stock weights only (exclude cash weight)
        stock_weights    = final_weights[:N_TICKERS]
        portfolio_return = np.dot(stock_weights, ticker_returns)

        # Apply transaction cost (proportional to weight change)
        weight_change    = np.abs(final_weights - self.portfolio_weights).sum()
        transaction_cost = weight_change * self.transaction_cost
        portfolio_return -= transaction_cost

        # Update portfolio value and weights
        self.portfolio_value  *= (1 + portfolio_return)
        self.portfolio_weights = final_weights.copy()
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(portfolio_return)

        # Compute Sharpe-based reward
        reward = self._compute_reward()

        for agent in self.possible_agents:
            self.rewards[agent] = reward

        # Check episode termination
        if self.timestep + 1 >= self.episode_length:
            for agent in self.possible_agents:
                self.terminations[agent] = True

        # Update infos
        for agent in self.possible_agents:
            self.infos[agent] = {
                "portfolio_value"  : self.portfolio_value,
                "portfolio_return" : portfolio_return,
                "weights"          : final_weights.tolist(),
                "timestep"         : self.timestep,
            }

    # ── reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Sharpe-based reward using recent return window.
        Penalises excessive drawdown.
        """
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history[-30:])
        mean_r  = returns.mean()
        std_r   = returns.std() + 1e-8
        sharpe  = mean_r / std_r

        # Drawdown penalty
        peak     = max(self.portfolio_history)
        drawdown = (peak - self.portfolio_value) / (peak + 1e-8)
        penalty  = max(0.0, drawdown - 0.1) * 2.0  # penalise if >10% drawdown

        return float(sharpe - penalty)

    # ── render ───────────────────────────────────────────────────────────────

    def render(self):
        """Print current portfolio state."""
        print(f"Step {self.timestep:4d} | "
              f"Value: ${self.portfolio_value:,.2f} | "
              f"Return: {((self.portfolio_value/self.initial_capital)-1)*100:.2f}%")
        for i, ticker in enumerate(TICKERS):
            print(f"  {ticker}: {self.portfolio_weights[i]*100:.1f}%")
        print(f"  Cash: {self.portfolio_weights[-1]*100:.1f}%")

    # ── utility ──────────────────────────────────────────────────────────────

    def get_episode_summary(self) -> dict:
        """Return summary metrics for the completed episode."""
        values  = np.array(self.portfolio_history)
        returns = np.array(self.returns_history) if self.returns_history else np.array([0.0])

        total_return = (values[-1] - values[0]) / values[0]
        sharpe       = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
        peak         = values.max()
        max_drawdown = ((peak - values.min()) / (peak + 1e-8))

        return {
            "total_return" : float(total_return),
            "sharpe_ratio" : float(sharpe),
            "max_drawdown" : float(max_drawdown),
            "final_value"  : float(values[-1]),
            "n_steps"      : self.timestep,
        }

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]