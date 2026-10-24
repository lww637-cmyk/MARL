"""Stand-in check of the coordinator call log: fake Gemini client, synthetic prices, no API calls."""
import contextlib
import io
import json
import os
import tempfile
import numpy as np
import pandas as pd

# a dummy key so importing the coordinator never picks up the real one
os.environ["GEMINI_API_KEY"] = "fake-key-for-stand-in-test"
import src.coordinator.llm_coordinator as llm
import evaluate_agents as ev
from src.environment.stock_env import StockTradingEnv, TICKERS, ACTION_SIZE

CONFIG = {"coordinator": {"model": "fake-model", "temperature": 0.0, "max_tokens": 64}}


class FakeUsage:
    prompt_token_count, candidates_token_count, total_token_count = 100, 20, 120


class FakeResponse:
    def __init__(self, text):
        self.text, self.usage_metadata = text, FakeUsage()


class FakeModels:
    """Plays back scripted replies in a loop; an Exception entry is raised."""
    def __init__(self, replies):
        self.replies, self.calls = replies, 0

    def generate_content(self, model, contents, config):
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    def __init__(self, replies):
        self.models = FakeModels(replies)


def as_json(weights):
    return json.dumps(dict(zip(TICKERS + ["cash"], weights)))


price   = np.array([0.6, 0.2, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
sent    = np.array([0.0, 0.2, 0.6, 0.0, 0.0, 0.2], dtype=np.float32)
average = llm._fallback_weights(price, sent)
context = {"portfolio_value": 100000, "initial_capital": 100000,
           "timestep": 0, "recent_returns": []}
assert ACTION_SIZE == len(TICKERS) + 1


def call(reply):
    llm.client = FakeClient([reply])
    record = {}
    weights = llm.coordinate_or_average(price, sent, context, CONFIG,
                                        use_llm=True, record=record)
    assert np.isclose(weights.sum(), 1.0) and (weights >= 0).all()
    assert record["final_weights"] == [float(w) for w in weights]
    assert record["prompt"].startswith("You are a portfolio coordinator")
    return weights, record


# 1. ok: negative weights clipped to 0, then normalised; tokens recorded
w, r = call(as_json([0.5, -0.2, 0.3, 0.0, 0.0, 0.2]))
assert r["outcome"] == llm.OK and r["error"] is None
assert np.allclose(w, [0.5, 0.0, 0.3, 0.0, 0.0, 0.2])
assert np.isclose(r["parsed_weights"][1], -0.2)                 # logged as given
assert (r["prompt_token_count"], r["candidates_token_count"], r["total_token_count"]) == (100, 20, 120)
assert r["thoughts_token_count"] is None
print("ok + negative clipping OK")

# 2. all-zero (or all-negative) weights -> agent average
for reply in [as_json([0] * 6), as_json([-0.1] * 6)]:
    w, r = call(reply)
    assert r["outcome"] == llm.ALL_ZERO and np.allclose(w, average), reply
print("all_zero_weights -> agent average OK")

# 3. unparseable / empty / non-object / NaN response -> equal weights
for reply in ["not json", None, "[0.2, 0.2]", '{"' + TICKERS[0] + '": NaN}', "```json\n{oops\n```"]:
    w, r = call(reply)
    assert r["outcome"] == llm.PARSE_ERROR and np.allclose(w, llm.EQUAL_WEIGHTS), reply
    assert r["response_text"] == reply and r["error"]
w, r = call("```json\n" + as_json([1, 0, 0, 0, 0, 0]) + "\n```")      # fenced JSON still parses
assert r["outcome"] == llm.OK and np.isclose(w[0], 1.0)
print("parse_error_equal_weights OK")

# 4. API failure -> agent average, no response or tokens
w, r = call(RuntimeError("quota exceeded"))
assert r["outcome"] == llm.API_ERROR and np.allclose(w, average)
assert r["response_text"] is None and r["total_token_count"] is None and "quota" in r["error"]
print("api_error_agent_average OK")


# 5. Full run with --tag: JSONL per call, seed/step, outcome counts per seed
N_DAYS = 30
dates  = pd.bdate_range("2023-01-02", periods=N_DAYS)
rng    = np.random.default_rng(1)
closes = {t: 100 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)) for t in TICKERS}


class StandInEnv(StockTradingEnv):
    """Real step/summary logic on a synthetic dataset, no files read."""
    def __init__(self, full_period):
        self.mode, self.initial_capital, self.transaction_cost = "eval", 100000.0, 0.001
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
        self.weights = weights

    def predict(self, obs):
        return self.weights


checked = []
ev.make_test_env  = lambda config, full_period: StandInEnv(full_period)
ev.load_agents    = lambda env, config, d: (FakeAgent(price), FakeAgent(sent))
ev.training_setup = lambda config: {}
ev.check_setup    = lambda d, setup: checked.append(d)
llm.client = FakeClient([as_json([1, 0, 0, 0, 0, 0]), "garbage",
                         as_json([0] * 6), RuntimeError("down")])
cwd, out = os.getcwd(), io.StringIO()
with tempfile.TemporaryDirectory() as d:
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(out):
            ev.run_full_evaluation(CONFIG, train_seeds=[0, 1], robustness=True,
                                   window_seeds=[0, 1], checkpoint_root="x",
                                   use_llm=True, tag="standin")
        with open(os.path.join("results", "logs", "standin", "coordinator_calls.jsonl"),
                  encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        # --no-llm writes no call log
        with contextlib.redirect_stdout(io.StringIO()):
            ev.run_full_evaluation(CONFIG, train_seeds=[0], checkpoint_root="x",
                                   use_llm=False, tag="nollm")
        assert "coordinator_calls.jsonl" not in os.listdir(os.path.join("results", "logs", "nollm"))
    finally:
        os.chdir(cwd)

assert len(checked) == 3                     # setup check ran for every seed folder, each run
# full period: steps 0 and 20 per training seed; robustness: step 0 per window (seed 0's agents)
assert [(r["phase"], r["seed"], r.get("window_seed"), r["step"]) for r in lines] == [
    ("test_period", 0, None, 0), ("test_period", 0, None, 20),
    ("test_period", 1, None, 0), ("test_period", 1, None, 20),
    ("robustness", 0, 0, 0), ("robustness", 0, 1, 0)]
assert [r["outcome"] for r in lines] == [llm.OK, llm.PARSE_ERROR, llm.ALL_ZERO, llm.API_ERROR,
                                         llm.OK, llm.PARSE_ERROR]
assert all(r["strategy"] == "MARL+LLM" for r in lines)
for key in ["prompt", "response_text", "parsed_weights", "final_weights", "outcome",
            "prompt_token_count", "candidates_token_count", "total_token_count", "date"]:
    assert all(key in r for r in lines), key
text = out.getvalue()
assert "COORDINATOR CALL OUTCOMES" in text
for row in ["test_period seed=0 | calls=   2 | ok=1, parse_error_equal_weights=1",
            "test_period seed=1 | calls=   2 | all_zero_weights=1, api_error_agent_average=1",
            "robustness  seed=0 | calls=   2 | ok=1, parse_error_equal_weights=1"]:
    assert row in text, row
print("coordinator_calls.jsonl + outcome summary OK")
print(text[text.index("COORDINATOR CALL OUTCOMES") - 66:])
