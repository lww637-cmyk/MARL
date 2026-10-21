"""Checkpoint folders are refused unless training_setup.json matches the current setup."""
import os
import json
import tempfile
import yaml
from src.checkpoints import (
    SETUP_FILE, checkpoint_dir, final_checkpoint, training_setup,
    check_setup, load_checked_agents,
)

with open("config.yaml") as f:
    config = yaml.safe_load(f)
setup = training_setup(config)


class FakeAgent:
    def __init__(self, agent_id):
        self.agent_id, self.loaded = agent_id, None

    def load(self, path):
        self.loaded = path


def make_dir(d, saved_setup=None):
    for agent_id in ["price_agent", "sentiment_agent"]:
        open(final_checkpoint(d, agent_id), "wb").close()
    if saved_setup is not None:
        with open(os.path.join(d, SETUP_FILE), "w") as f:
            json.dump(saved_setup, f)


def refused(d, expect):
    agents = [FakeAgent("price_agent"), FakeAgent("sentiment_agent")]
    try:
        load_checked_agents(d, config, agents)
    except RuntimeError as e:
        msg = str(e)
        assert d in msg, msg                     # names the folder
        assert expect in msg, msg                # names the problem
        assert all(a.loaded is None for a in agents)
        return msg
    raise AssertionError(f"{d} accepted")


# 1. Paths: {root}/seed{N}/..._final.zip, root from config or overridden
assert checkpoint_dir(config, 1) == "results/checkpoints_scaled/seed1"
assert checkpoint_dir({}, 2) == "results/checkpoints_scaled/seed2"
assert checkpoint_dir(config, 0, root="x/y") == "x/y/seed0"
assert final_checkpoint("x/seed0", "price_agent") == "x/seed0/price_agent_final.zip"
print("checkpoint paths OK")

# 2. Old folder without training_setup.json is refused
with tempfile.TemporaryDirectory() as d:
    make_dir(d)
    print("  ", refused(d, f"no {SETUP_FILE}"))
print("folder without setup file refused OK")

# 3. Folder with a wrong price scaler hash is refused, naming the field
with tempfile.TemporaryDirectory() as d:
    make_dir(d, dict(setup, price_scaler_sha256="0123456789abcdef"))
    print("  ", refused(d, "price_scaler_sha256"))
print("wrong scaler hash refused OK")

# 4. Wrong partner / action mapping are refused too
for field, value in [("partner", "random"), ("action_mapping", "old")]:
    with tempfile.TemporaryDirectory() as d:
        make_dir(d, dict(setup, **{field: value}))
        refused(d, field)
print("wrong partner / action mapping refused OK")

# 5. Matching folder passes and both agents are loaded from it
with tempfile.TemporaryDirectory() as d:
    make_dir(d, setup)
    check_setup(d, setup)
    agents = [FakeAgent("price_agent"), FakeAgent("sentiment_agent")]
    load_checked_agents(d, config, agents)
    assert [a.loaded for a in agents] == [
        final_checkpoint(d, "price_agent"), final_checkpoint(d, "sentiment_agent")]
print("matching folder accepted OK")

# 6. Matching setup but a missing final checkpoint is an error, not an untrained model
with tempfile.TemporaryDirectory() as d:
    make_dir(d, setup)
    os.remove(final_checkpoint(d, "sentiment_agent"))
    try:
        load_checked_agents(d, config, [FakeAgent("price_agent"),
                                        FakeAgent("sentiment_agent")])
        raise AssertionError("missing checkpoint accepted")
    except FileNotFoundError:
        pass
print("missing checkpoint refused OK")

print("\nCheckpoint setup test passed.")
