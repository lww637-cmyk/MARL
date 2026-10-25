"""Stand-in check of evaluate_agents strategies: synthetic prices, fake agents, no data, checkpoints or Gemini."""
import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import evaluate_agents as ev
from src.environment.stock_env import StockTradingEnv, TICKERS, ACTION_SIZE

N_DAYS, CAPITAL, COST = 30, 100000.0, 0.001
dates = pd.bdate_range("2023-01-02", periods=N_DAYS)
rng   = np.random.default_rng(0)
# each ticker drifts differently so buy-and-hold and rebalancing diverge
closes = {t: 100 * np.cumprod(1 + rng.normal(0.004 * (i - 2), 0.02, N_DAYS))
          for i, t in enumerate(TICKERS)}


class StandInEnv(StockTradingEnv):
    """Real step/summary logic on a synthetic dataset, no files read."""
    def __init__(self, full_period):
        self.mode, self.initial_capital, self.transaction_cost = "eval", CAPITAL, COST
        self.dataset      = {t: pd.DataFrame({"close": closes[t]}, index=dates) for t in TICKERS}
        self.common_dates = list(dates)
        self.n_steps      = N_DAYS
        self.episode_length = N_DAYS - 1 if full_period else 10
        self.possible_agents = ["price_agent", "sentiment_agent"]
        self.reset()

    def observe(self, agent):
        return np.zeros(1, dtype=np.float32)


class FakeAgent:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32)

    def predict(self, obs):
        return self.weights


def one_hot(i):
    w = np.zeros(ACTION_SIZE, dtype=np.float32)
    w[i] = 1.0
    return w


price_agent, sentiment_agent = FakeAgent(one_hot(0)), FakeAgent(one_hot(1))
ratio = {t: closes[t][-1] / closes[t][0] for t in TICKERS}
r1    = {t: closes[t][1] / closes[t][0] - 1 for t in TICKERS}
w     = ev.BAH_WEIGHTS[:len(TICKERS)].astype(np.float64)
entry = 2.0 * COST    # cash -> fully invested: |dw| sums to 2

# 1. True buy-and-hold: one buy, then drift; value matches holding the shares
env = StandInEnv(full_period=True)
bah = ev.run_episode(env, fixed_weights=ev.BAH_WEIGHTS, rebalance=False)
wr1 = sum(wi * r1[t] for wi, t in zip(w, TICKERS))
expected = CAPITAL * (1 + wr1 - entry) / (1 + wr1) * sum(wi * ratio[t] for wi, t in zip(w, TICKERS))
assert np.isclose(bah["final_value"], expected, rtol=1e-6), (bah["final_value"], expected)
assert np.isclose(bah["turnover"], 2.0) and np.isclose(bah["total_cost"], CAPITAL * entry)
print("buy_and_hold OK", round(bah["final_value"], 2))

# 2. Equal-weight daily rebalanced differs from buy-and-hold
ew = ev.run_episode(env, fixed_weights=ev.BAH_WEIGHTS, rebalance=True)
rets = np.array([[closes[t][k + 1] / closes[t][k] - 1 for t in TICKERS] for k in range(N_DAYS - 1)])
daily = rets @ w
daily[0] -= entry
assert np.isclose(ew["final_value"], CAPITAL * np.prod(1 + daily), rtol=1e-6)
assert not np.isclose(ew["final_value"], bah["final_value"], rtol=1e-4)
assert np.isclose(ew["total_cost"], CAPITAL * entry)   # env charges |dw| vs last target
print("equal_weight_daily_rebalanced OK", round(ew["final_value"], 2))

# 3. Single-agent strategies use only that agent's action
t0, t1 = TICKERS[0], TICKERS[1]
for mode, t in [("price", t0), ("sentiment", t1)]:
    s = ev.run_episode(env, price_agent, sentiment_agent, agent_mode=mode)
    exp = CAPITAL * (1 + r1[t] - entry) / (1 + r1[t]) * ratio[t]
    assert np.isclose(s["final_value"], exp, rtol=1e-5), (mode, s["final_value"], exp)
    assert np.isclose(s["turnover"], 2.0)
both = ev.run_episode(env, price_agent, sentiment_agent, agent_mode="both")
half = 0.5 * (rets[:, 0] + rets[:, 1])  # average of both agents, rebalanced daily
half[0] -= entry
assert np.isclose(both["final_value"], CAPITAL * np.prod(1 + half), rtol=1e-5)
try:
    ev.run_episode(env, price_agent, sentiment_agent, agent_mode="neither")
    raise AssertionError("bad agent_mode accepted")
except ValueError:
    pass
print("price-agent-only / sentiment-agent-only OK")


class SwitchAgent:
    """Alternates between two one-hot weights on every predict call."""
    def __init__(self, a, b):
        self.options, self.calls = [one_hot(a), one_hot(b)], 0

    def predict(self, obs):
        self.calls += 1
        return self.options[(self.calls - 1) % 2]


# 4. *-hold20: weights recomputed on steps 0 and 20 only (both even calls),
#    so the held target is the first weight throughout
for mode, ref in [("price", "price"), ("sentiment", "sentiment"), ("both", "both")]:
    expected_run = ev.run_episode(env, price_agent, sentiment_agent, agent_mode=ref)
    held = ev.run_episode(env, SwitchAgent(0, 2), SwitchAgent(1, 3),
                          agent_mode=mode, hold_every=ev.DEFAULT_CALL_FREQUENCY)
    daily_run = ev.run_episode(env, SwitchAgent(0, 2), SwitchAgent(1, 3), agent_mode=mode)
    assert np.isclose(held["final_value"], expected_run["final_value"], rtol=1e-6), mode
    assert np.isclose(held["turnover"], expected_run["turnover"])
    assert np.isclose(held["total_cost"], expected_run["total_cost"])
    assert daily_run["turnover"] > held["turnover"] + 1.0, mode   # switching trades daily
print("*-hold20 OK")

# 5. Full run with --no-llm and --tag: all non-LLM strategies, no Gemini, dash when n=1
ev.make_test_env  = lambda config, full_period: StandInEnv(full_period)
ev.load_agents    = lambda env, config, d: (price_agent, sentiment_agent)
ev.training_setup = lambda config: {}
ev.check_setup    = lambda d, setup: None
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as d:
    os.chdir(d)
    try:
        df, summary = ev.run_full_evaluation({}, train_seeds=[0, 1], robustness=True,
                                             window_seeds=[0, 1, 2], checkpoint_root="x",
                                             use_llm=False, tag="standin")
        out = os.path.join("results", "logs", "standin")
        assert sorted(os.listdir(out)) == sorted([
            "test_period_results.csv", "test_period_summary.csv",
            "robustness_windows_results.csv", "robustness_windows_summary.csv",
            "run_metadata.json"])
        with open(os.path.join(out, "run_metadata.json")) as f:
            meta = json.load(f)
        assert meta["call_frequency"] == ev.DEFAULT_CALL_FREQUENCY, meta
        assert meta["call_frequency_source"] == "DEFAULT_CALL_FREQUENCY fallback", meta
        saved = pd.read_csv(os.path.join(out, "test_period_summary.csv"))
        rob   = pd.read_csv(os.path.join(out, "robustness_windows_summary.csv"))
    finally:
        os.chdir(cwd)

expected_strategies = ["buy_and_hold", "equal_weight_daily_rebalanced", "MARL-no-coordinator",
                       "price-agent-only", "sentiment-agent-only", "MARL-average-hold20",
                       "price-only-hold20", "sentiment-only-hold20"]
assert list(summary["strategy"]) == expected_strategies, list(summary["strategy"])
assert sorted(rob["strategy"]) == sorted(expected_strategies)
assert "src.coordinator.llm_coordinator" not in sys.modules
for col in ["turnover", "total_cost"]:
    assert df[col].notna().all() and (df[col] > 0).all()
std_cols = [c for c in summary.columns if c.startswith("std_")]
assert {"mean_turnover", "std_turnover", "mean_total_cost", "std_total_cost"} <= set(summary.columns)
s = saved.set_index("strategy")
assert (s.loc[["buy_and_hold", "equal_weight_daily_rebalanced"], std_cols] == "-").all().all()
assert saved[std_cols].notna().all().all()          # no NaN anywhere, "-" instead
assert rob[std_cols].notna().all().all()            # n=3 windows -> numbers
print("run_full_evaluation --no-llm --tag OK")

# 6. Cadence comes from coordinator.call_frequency, fallback only when missing
assert ev.call_frequency({}) == ev.DEFAULT_CALL_FREQUENCY
assert ev.call_frequency({"coordinator": {"call_frequency": 7}}) == 7
held = {k: v[2] for k, v in ev.agent_strategies(False, 7).items()}
assert held == {"MARL-no-coordinator": None, "price-agent-only": None,
                "sentiment-agent-only": None, "MARL-average-hold20": 7,
                "price-only-hold20": 7, "sentiment-only-hold20": 7}, held
assert "MARL+LLM" in ev.agent_strategies(True, 7)
assert ev.call_frequency(ev.load_config()) == 20      # config.yaml matches the runs so far
print("call_frequency from config OK")
print(summary.to_string(index=False))
