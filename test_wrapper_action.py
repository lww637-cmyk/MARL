"""Wrapper action mapping matches env.step, neutral/random partner, predict output, resume guard."""
import os
import tempfile
import numpy as np
import yaml
from src.agents.base_agent import BaseTrader
from src.environment.stock_env import (
    StockTradingEnv, ACTION_SIZE, normalise_action, partner_action,
)
from src.training.train_marl import training_setup, check_training_setup

with open("config.yaml") as f:
    config = yaml.safe_load(f)
assert config["training"]["partner"] == "neutral"

# 1. normalise_action: clip to [0, 1], divide by the sum, sum = 0 -> all cash
w = normalise_action([2.0, -1.0, 0.5, 0.0, 0.0, 0.5])
assert np.allclose(w, [0.5, 0.0, 0.25, 0.0, 0.0, 0.25])
assert np.allclose(normalise_action(w), w)          # idempotent
cash = normalise_action([-1.0, 0.0, -3.0, 0.0, 0.0, 0.0])
assert np.allclose(cash, [0, 0, 0, 0, 0, 1])
print("normalise_action OK")

# 2. Partners
neutral = partner_action("neutral", None)
assert np.allclose(neutral, 1.0 / ACTION_SIZE)
env = StockTradingEnv(config, mode="eval", split="train")
r = partner_action("random", env.action_space("sentiment_agent"))
assert np.isclose(r.sum(), 1.0) and r.min() >= 0.0
try:
    partner_action("other", None)
    raise AssertionError("unknown partner accepted")
except ValueError:
    pass
print("partner_action OK")


def wrapper(agent_id, cfg):
    trader = BaseTrader.__new__(BaseTrader)
    trader.agent_id, trader.config = agent_id, cfg
    return trader._make_single_env().envs[0].env    # SingleAgentWrapper


# 3. Wrapper weights == env.step (evaluation path) weights for the same action
action = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)
for agent_id in ["price_agent", "sentiment_agent"]:
    wr = wrapper(agent_id, config)
    wr.reset(seed=0)
    wr._env.start_idx = 10
    _, _, _, _, info = wr.step(action)

    ev = StockTradingEnv(config, mode="eval", split="train")
    ev.reset(seed=0)
    ev.start_idx = 10
    for _ in ev.possible_agents:                     # AEC: both agents act
        current = ev.agent_selection
        ev.step(action if current == agent_id else neutral)

    expected = (normalise_action(action) + neutral) / 2.0
    assert np.allclose(info["weights"], expected, atol=1e-6)
    assert np.allclose(info["weights"], ev.infos[agent_id]["weights"], atol=1e-6)
    assert np.isclose(info["portfolio_return"], ev.infos[agent_id]["portfolio_return"])
print("Wrapper mapping matches env.step OK")

# 4. Unknown partner in config is rejected when the wrapper is built
bad = dict(config, training=dict(config["training"], partner="mirror"))
try:
    wrapper("price_agent", bad)
    raise AssertionError("unknown partner in config accepted")
except ValueError:
    pass

# 5. Agent predict returns valid portfolio weights
from src.agents.price_agent import PriceAgent
agent = PriceAgent(env, config)
env.reset(seed=0)
p = agent.predict(env.observe("price_agent"))
assert p.shape == (ACTION_SIZE,) and np.isclose(p.sum(), 1.0) and p.min() >= 0.0
print("predict returns normalised weights OK")

# 6. Resume guard: checkpoints from another setup are refused
setup = training_setup(config)
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "price_agent_step100000.zip"), "wb").close()
    try:
        check_training_setup(setup, [d])             # old run, no setup file
        raise AssertionError("checkpoint without setup file accepted")
    except RuntimeError:
        pass
with tempfile.TemporaryDirectory() as d:
    check_training_setup(setup, [d])                 # empty dir: records setup
    open(os.path.join(d, "price_agent_step100000.zip"), "wb").close()
    check_training_setup(setup, [d])                 # same setup: resume allowed
    other = dict(setup, partner="random")
    try:
        check_training_setup(other, [d])
        raise AssertionError("different setup accepted")
    except RuntimeError:
        pass
print("Resume guard OK")

print("\nWrapper action test passed.")
