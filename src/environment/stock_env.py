"""
Custom multi-agent trading environment for the tickers in config.yaml.
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
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    TICKERS = yaml.safe_load(f)["data"]["tickers"]
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
N_SHARED = N_TICKERS + 2  # 5 stock weights + 1 cash weight + 1 norm value

# Total obs sizes
PRICE_OBS_SIZE     = N_PRICE_FEATURES * N_TICKERS + N_SHARED   # 50 + 7 = 57
SENTIMENT_OBS_SIZE = N_SENTIMENT_FEATURES * N_TICKERS + N_SHARED  # 20 + 7 = 27

# Action = portfolio weights for 5 tickers + cash = 6 values
ACTION_SIZE = N_TICKERS + 1

# Chronological split (inclusive bounds)
TRAIN_END  = pd.Timestamp("2022-06-30")
TEST_START = pd.Timestamp("2022-07-01")
SPLITS     = ("train", "test", "all")

# Max trading days a sentiment value is carried forward after a real headline day
SENTIMENT_FFILL_LIMIT = 5

# Price features are standardised with TRAIN-split mean/std (fit_price_scaler.py),
# then clipped to +-PRICE_CLIP. Committed to git so Colab gets it with the clone.
PRICE_SCALER_PATH = "data/price_scaler.json"
PRICE_CLIP        = 5.0

# Fixed partner used while training one agent (config training.partner)
PARTNERS = ("neutral", "random")


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


def roll_to_trading_days(sent_df: pd.DataFrame,
                         trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Move headline days that are not trading days (weekends, holidays) to the
    next trading day and combine them with that day's headlines.
    Mean is count-weighted, std is the pooled sample std (same as computing
    it over the individual headlines). Days before the first or after the
    last trading day are dropped.
    """
    sent_df = sent_df[(sent_df.index >= trading_days[0]) &
                      (sent_df.index <= trading_days[-1])]
    target = trading_days[trading_days.searchsorted(sent_df.index, side="left")]

    n = sent_df["headline_count"]
    m = sent_df["sentiment_score"]
    s = sent_df["sentiment_std"]
    parts = pd.DataFrame({"n": n.values, "nm": (n * m).values,
                          "nm2": (n * m * m).values,
                          "ss": ((n - 1) * s * s).values}, index=target)
    g = parts.groupby(level=0).sum()

    mean = g["nm"] / g["n"]
    # within-day sum of squares + between-day sum of squares around the new mean
    total_ss = g["ss"] + g["nm2"] - g["n"] * mean * mean
    var = (total_ss / (g["n"] - 1)).where(g["n"] > 1, 0.0).clip(lower=0.0)

    rolled = pd.DataFrame({"sentiment_score": mean,
                           "sentiment_std": np.sqrt(var),
                           "headline_count": g["n"]})
    rolled.index.name = sent_df.index.name
    return rolled


def build_merged_dataset(price_data: dict,
                          sentiment_data: dict,
                          episode_length: int = 252) -> dict:
    """
    Align price and sentiment data on the same daily index.
    Headlines on non-trading days are rolled forward to the next trading day.
    Forward-fills sentiment score/std on missing days for at most
    SENTIMENT_FFILL_LIMIT trading days, then fills with 0.0.
    headline_count is 0 on days without headlines (never forward-filled).
    headline_count_norm = log1p(count) / max over the TRAIN period only,
    clipped to [0, 1].
    Adds has_sentiment flag (1 = real headlines that day, 0 = filled).
    Returns dict of {ticker: merged_dataframe}.
    """
    merged = {}

    for ticker in TICKERS:
        price_df = price_data[ticker].copy()
        sent_df  = sentiment_data.get(ticker)

        if sent_df is not None:
            # Roll weekend / holiday headlines onto the next trading day,
            # then mark which dates have real sentiment
            sent_df = roll_to_trading_days(sent_df, price_df.index)
            sent_df["has_sentiment"] = 1.0

            # Merge on price index (left join — keeps all trading days)
            merged_df = price_df.join(sent_df, how="left")
            merged_df["has_sentiment"] = merged_df["has_sentiment"].fillna(0.0)

            # Forward-fill sentiment scores for at most SENTIMENT_FFILL_LIMIT
            # trading days, then treat as no signal (0.0)
            for col in ["sentiment_score", "sentiment_std"]:
                merged_df[col] = (
                    merged_df[col]
                    .ffill(limit=SENTIMENT_FFILL_LIMIT)
                    .fillna(0.0)
                )
            # Headline count is never carried forward: 0 on days without headlines
            merged_df["headline_count"] = merged_df["headline_count"].fillna(0.0)

            # Normalise log headline count by the TRAIN-period maximum only
            # (no test-period information), clipped to [0, 1]
            log_count = np.log1p(merged_df["headline_count"])
            max_log   = log_count[merged_df.index <= TRAIN_END].max()
            merged_df["headline_count_norm"] = (
                (log_count / max_log).clip(0.0, 1.0) if max_log > 0 else 0.0
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


# ── price scaler ─────────────────────────────────────────────────────────────

def common_trading_dates(dataset: dict, split: str = "all") -> list:
    """Dates present for every ticker, restricted to the chronological split."""
    common = None
    for ticker in TICKERS:
        dates  = set(dataset[ticker].index)
        common = dates if common is None else common & dates
    common = sorted(common)
    if split == "train":
        common = [d for d in common if d <= TRAIN_END]
    elif split == "test":
        common = [d for d in common if d >= TEST_START]
    return common


def compute_price_scaler(dataset: dict) -> dict:
    """
    Mean and std of every (ticker, price feature) over the TRAIN split only.
    A std of 0 is replaced by 1 so a constant feature maps to 0.
    """
    train_dates = common_trading_dates(dataset, "train")
    mean, std = {}, {}
    for ticker in TICKERS:
        values = dataset[ticker].loc[train_dates, PRICE_FEATURES]
        mean[ticker] = values.mean().tolist()
        s = values.std(ddof=0)
        std[ticker]  = s.where(s > 1e-8, 1.0).tolist()
    return {
        "tickers"    : list(TICKERS),
        "features"   : list(PRICE_FEATURES),
        "train_start": str(train_dates[0].date()),
        "train_end"  : str(train_dates[-1].date()),
        "n_days"     : len(train_dates),
        "clip"       : PRICE_CLIP,
        "mean"       : mean,
        "std"        : std,
    }


def load_price_scaler(path: str = PRICE_SCALER_PATH) -> dict:
    """Load the price scaler and check it matches the current tickers/features."""
    import json
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python fit_price_scaler.py` "
            f"(train split only) or pull it from git"
        )
    with open(path) as f:
        scaler = json.load(f)
    if scaler["tickers"] != list(TICKERS) or scaler["features"] != list(PRICE_FEATURES):
        raise ValueError(
            f"{path} was fitted for tickers {scaler['tickers']} / features "
            f"{scaler['features']}; re-run fit_price_scaler.py"
        )
    if pd.Timestamp(scaler["train_end"]) > TRAIN_END:
        raise ValueError(f"{path} uses data after TRAIN_END={TRAIN_END.date()}")
    return scaler


def scale_price_features(ticker: str, values, scaler: dict) -> np.ndarray:
    """Standardise one ticker's PRICE_FEATURES values and clip to +-clip."""
    mean = np.asarray(scaler["mean"][ticker], dtype=np.float64)
    std  = np.asarray(scaler["std"][ticker], dtype=np.float64)
    z    = (np.asarray(values, dtype=np.float64) - mean) / std
    return np.clip(z, -scaler["clip"], scaler["clip"])


# ── action helpers ───────────────────────────────────────────────────────────

def normalise_action(action) -> np.ndarray:
    """
    Map an agent action to portfolio weights: clip to [0, 1] and divide by the sum.
    An all-zero action (sum = 0) means 100% cash.
    """
    action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    total  = action.sum()
    if total <= 1e-8:
        weights = np.zeros(ACTION_SIZE, dtype=np.float32)
        weights[-1] = 1.0
        return weights
    return (action / total).astype(np.float32)


def partner_action(kind: str, action_space) -> np.ndarray:
    """
    Suggestion of the agent that is NOT being trained (config training.partner):
    neutral = equal weights on the stocks and cash, random = uniform sample.
    Both are passed through normalise_action like any agent suggestion.
    """
    if kind == "neutral":
        return np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32)
    if kind == "random":
        return normalise_action(action_space.sample())
    raise ValueError(f"training.partner must be one of {PARTNERS}, got {kind!r}")


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

    def __init__(self, config: dict, mode: str = "train",
                 split: str = "train", full_period: bool = False):
        """
        mode        : "train" (random episode start) or "eval" (start at 0)
        split       : "train" (<= TRAIN_END), "test" (>= TEST_START) or "all"
        full_period : if True, one episode spans the whole split
                      (overrides config episode_length)
        """
        super().__init__()

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.config          = config
        self.mode            = mode   # "train" or "eval"
        self.split           = split
        self.initial_capital = config["environment"]["initial_capital"]
        self.episode_length  = config["environment"]["episode_length"]
        self.transaction_cost = config["environment"]["transaction_cost"]

        # Load and merge data
        price_data     = load_price_data()
        sentiment_data = load_sentiment_data()
        self.dataset   = build_merged_dataset(
            price_data, sentiment_data, self.episode_length
        )

        # Common date index (intersection across all tickers),
        # restricted to the chronological split
        self.common_dates = common_trading_dates(self.dataset, split)
        self.n_steps      = len(self.common_dates)

        # Train-split price scaler (same file for every split and consumer)
        self.price_scaler = load_price_scaler()

        if full_period:
            self.episode_length = self.n_steps - 1
        if self.n_steps <= self.episode_length:
            raise ValueError(
                f"{split} split has {self.n_steps} trading days, "
                f"need > episode_length={self.episode_length}"
            )

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
        date   = self.common_dates[self.start_idx + self.timestep]
        shared = np.append(
            self.portfolio_weights,
            self.portfolio_value / self.initial_capital
        )

        if agent == "price_agent":
            features = []
            for ticker in TICKERS:
                row = self.dataset[ticker].loc[date, PRICE_FEATURES]
                # Standardise with train-split mean/std, clipped to +-PRICE_CLIP
                features.extend(
                    scale_price_features(ticker, row.values, self.price_scaler)
                )
            return np.array(features + shared.tolist(), dtype=np.float32)

        elif agent == "sentiment_agent":
            features = []
            for ticker in TICKERS:
                row = self.dataset[ticker].loc[date]
                features.extend([row[f] for f in SENTIMENT_FEATURES])
            return np.array(features + shared.tolist(), dtype=np.float32)

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
        action = normalise_action(action)

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
        Combined reward: daily return + Sharpe component - drawdown penalty.
        Provides clearer gradient signal than pure Sharpe early in training.
        """
        if len(self.returns_history) < 1:
            return 0.0

        # Daily return component (annualised)
        daily_return = self.returns_history[-1]

        # Sharpe component (only when enough history)
        if len(self.returns_history) >= 10:
            returns = np.array(self.returns_history[-30:])
            sharpe  = returns.mean() / (returns.std() + 1e-8)
        else:
            sharpe  = 0.0

        # Drawdown penalty
        peak     = max(self.portfolio_history)
        drawdown = (peak - self.portfolio_value) / (peak + 1e-8)
        penalty  = max(0.0, drawdown - 0.1) * 2.0

        # Weighted combination
        reward = 0.4 * daily_return * 252 + 0.6 * sharpe - penalty
        return float(reward)

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