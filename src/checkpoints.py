"""
Checkpoint folders: where they live and the guard against loading or
resuming checkpoints trained under a different setup.

Layout: {paths.checkpoints_root}/seed{N}/{agent_id}_final.zip, with a
training_setup.json written by train_marl.py next to the checkpoints.
"""

import os
import json
import hashlib
from src.environment.stock_env import load_price_scaler

SETUP_FILE = "training_setup.json"
DEFAULT_CHECKPOINTS_ROOT = "results/checkpoints_scaled"
AGENT_IDS = ["price_agent", "sentiment_agent"]


def checkpoints_root(config: dict) -> str:
    return config.get("paths", {}).get("checkpoints_root",
                                       DEFAULT_CHECKPOINTS_ROOT)


def checkpoint_dir(config: dict, seed: int = None, root: str = None) -> str:
    """{root}/seed{N}, or {root} itself for an unseeded run."""
    root = root or checkpoints_root(config)
    return root if seed is None else f"{root}/seed{seed}"


def final_checkpoint(ckpt_dir: str, agent_id: str) -> str:
    return f"{ckpt_dir}/{agent_id}_final.zip"


def training_setup(config: dict) -> dict:
    """
    What a checkpoint depends on besides its weights. Loading or resuming
    a checkpoint trained under a different setup is refused.
    """
    scaler = json.dumps(load_price_scaler(), sort_keys=True)
    return {
        "price_scaler_sha256": hashlib.sha256(scaler.encode()).hexdigest()[:16],
        "partner"            : config["training"].get("partner", "neutral"),
        "action_mapping"     : "clip_normalise_in_wrapper",
    }


def check_setup(ckpt_dir: str, setup: dict):
    """
    Raise RuntimeError unless ckpt_dir/training_setup.json exists and
    matches setup in every field. The message names the folder and the
    fields that differ.
    """
    path = os.path.join(ckpt_dir, SETUP_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"{ckpt_dir}: no {SETUP_FILE}, so its checkpoints may come from an "
            f"older training setup (current: {setup}). Copy the "
            f"{SETUP_FILE} written next to these checkpoints during training, "
            f"or retrain."
        )
    with open(path) as f:
        saved = json.load(f)
    diffs = [f"{k}: checkpoint {saved.get(k)!r}, current {setup.get(k)!r}"
             for k in sorted(set(saved) | set(setup))
             if saved.get(k) != setup.get(k)]
    if diffs:
        raise RuntimeError(
            f"{ckpt_dir} was trained under a different setup — "
            + "; ".join(diffs)
            + ". Use checkpoints trained with the current setup."
        )


def load_checked_agents(ckpt_dir: str, config: dict, agents: list):
    """
    Check ckpt_dir against the current setup, then load
    {agent_id}_final.zip into each agent. Missing files are an error.
    """
    check_setup(ckpt_dir, training_setup(config))
    for agent in agents:
        path = final_checkpoint(ckpt_dir, agent.agent_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        agent.load(path)
