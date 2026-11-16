"""
SHAP explainability analysis for trained MARL trading agents.
Computes feature attribution for both price and sentiment agents.
Saves SHAP values and plots to results/logs/shap/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
import yaml
from src.environment.stock_env import StockTradingEnv, TICKERS, PRICE_FEATURES, SENTIMENT_FEATURES
from src.agents.price_agent import PriceAgent
from src.agents.sentiment_agent import SentimentAgent
from src.checkpoints import checkpoint_dir, load_checked_agents

# ── config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ── feature names ─────────────────────────────────────────────────────────────

def get_price_feature_names() -> list:
    """Generate feature names for price agent observation vector."""
    names = []
    for ticker in TICKERS:
        for feat in PRICE_FEATURES:
            names.append(f"{ticker}_{feat}")
    # Shared features
    for ticker in TICKERS:
        names.append(f"w_{ticker}")
    names.append("w_cash")
    names.append("portfolio_norm")
    return names

def get_sentiment_feature_names() -> list:
    """Generate feature names for sentiment agent observation vector."""
    names = []
    for ticker in TICKERS:
        names.append(f"{ticker}_sentiment_score")
        names.append(f"{ticker}_sentiment_std")
        names.append(f"{ticker}_headline_count_norm")
        names.append(f"{ticker}_has_sentiment")
    # Shared features
    for ticker in TICKERS:
        names.append(f"w_{ticker}")
    names.append("w_cash")
    names.append("portfolio_norm")
    return names

# ── observation collector ────────────────────────────────────────────────────

def collect_observations(env: StockTradingEnv,
                          price_agent: PriceAgent,
                          sentiment_agent: SentimentAgent,
                          n_episodes: int = 5,
                          max_obs: int = 200) -> dict:
    """
    Run evaluation episodes and collect observations for both agents.
    Returns dict with price and sentiment observation arrays.
    """
    price_obs_list = []
    sent_obs_list  = []

    for ep in range(n_episodes):
        env.mode = "train"
        env.reset(seed=ep * 7)

        for step in range(env.episode_length):
            if not env.agents:
                break

            date_idx = env.start_idx + env.timestep
            if date_idx >= env.n_steps - 1:
                break

            price_obs = env.observe("price_agent")
            sent_obs  = env.observe("sentiment_agent")

            price_obs_list.append(price_obs.copy())
            sent_obs_list.append(sent_obs.copy())

            # Step both agents
            price_action = price_agent.predict(price_obs)
            sent_action  = sentiment_agent.predict(sent_obs)
            final_weights = (price_action + sent_action) / 2.0
            final_weights = final_weights / (final_weights.sum() + 1e-8)

            env.suggestions = {
                "price_agent"    : final_weights,
                "sentiment_agent": final_weights,
            }
            env._execute_step()
            env.timestep += 1
            env.suggestions = {}

            if len(price_obs_list) >= max_obs:
                break

        if len(price_obs_list) >= max_obs:
            break
            
    return {
        "price"    : np.array(price_obs_list[:max_obs]),
        "sentiment": np.array(sent_obs_list[:max_obs]),
    }


# ── SHAP computation ─────────────────────────────────────────────────────────

def compute_shap_values(model_predict_fn,
                         observations: np.ndarray,
                         n_background: int = 30,
                         agent_label: str = "agent") -> np.ndarray:
    """
    Compute SHAP values using KernelExplainer.
    Background = random sample from observations.
    Returns SHAP values array.
    """
    print(f"Computing SHAP for {agent_label}...")
    print(f"  Observations: {observations.shape}")
    print(f"  Background samples: {n_background}")

    # Sample background dataset
    bg_idx   = np.random.choice(len(observations), n_background, replace=False)
    background = observations[bg_idx]

    # Sample evaluation set (subset for speed)
    eval_idx = np.random.choice(len(observations),
                                min(50, len(observations)), replace=False)
    eval_obs = observations[eval_idx]

    # Create SHAP explainer
    explainer   = shap.KernelExplainer(model_predict_fn, background)
    shap_values = explainer.shap_values(eval_obs, nsamples=100, silent=True)

    print(f"  SHAP computation complete.")
    return shap_values, eval_obs

# ── predict wrappers ─────────────────────────────────────────────────────────

def make_predict_fn(agent, output_idx: int = None):
    """
    Wrap agent predict to return scalar or vector output for SHAP.
    output_idx: if None returns total stock allocation (1-cash)
                if int returns weight for that ticker index
    """
    def predict_fn(obs_array):
        results = []
        for obs in obs_array:
            action = agent.predict(obs.astype(np.float32))
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)
            if output_idx is None:
                # Total stock allocation (1 - cash weight)
                results.append(1.0 - action[-1])
            else:
                results.append(action[output_idx])
        return np.array(results)
    return predict_fn

# ── plotting ─────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray,
                       observations: np.ndarray,
                       feature_names: list,
                       title: str,
                       save_path: str,
                       top_n: int = 15):
    """Plot top-N most important features by mean absolute SHAP value."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[-top_n:][::-1]

    top_features  = [feature_names[i] for i in top_idx]
    top_values    = mean_abs_shap[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_values[::-1],
                   color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_shap_ticker_comparison(shap_values: np.ndarray,
                                 feature_names: list,
                                 title: str,
                                 save_path: str):
    """
    Group SHAP values by ticker and show which ticker's features
    matter most overall.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ticker_importance = {t: 0.0 for t in TICKERS}

    for i, fname in enumerate(feature_names):
        for ticker in TICKERS:
            if fname.startswith(ticker + "_"):
                ticker_importance[ticker] += mean_abs_shap[i]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ax.bar(ticker_importance.keys(),
           ticker_importance.values(),
           color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel("Total |SHAP value|")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def save_shap_csv(shap_values: np.ndarray,
                   feature_names: list,
                   agent_label: str,
                   save_path: str):
    """Save mean absolute SHAP values to CSV for dashboard."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature"       : feature_names,
        "mean_abs_shap" : mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(save_path, index=False)
    print(f"  Saved → {save_path}")
    return df

# ── main ─────────────────────────────────────────────────────────────────────

def run_shap_analysis(config: dict, ckpt_dir: str):
    """
    Full SHAP analysis pipeline for both agents in ckpt_dir
    (refused unless trained with the current setup).
    """

    os.makedirs("results/logs/shap", exist_ok=True)
    np.random.seed(42)

    # Load environment and agents
    print("Initialising environment and agents...")
    env = StockTradingEnv(config, mode="train")

    price_agent     = PriceAgent(env, config)
    sentiment_agent = SentimentAgent(env, config)

    load_checked_agents(ckpt_dir, config, [price_agent, sentiment_agent])

    # Collect observations
    print("\nCollecting observations from evaluation episodes...")
    observations = collect_observations(
        env, price_agent, sentiment_agent,
        n_episodes=5, max_obs=200
    )
    print(f"Collected {len(observations['price'])} price observations")
    print(f"Collected {len(observations['sentiment'])} sentiment observations")

    # Diagnose price agent output variance
    print("\nDiagnosing price agent output variance...")
    sample_actions = []
    for obs in observations["price"][:20]:
        action = price_agent.predict(obs.astype(np.float32))
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        sample_actions.append(action)

    sample_actions = np.array(sample_actions)
    print("Price agent action stats across 20 observations:")
    print(f"  Mean weights: {sample_actions.mean(axis=0).round(4)}")
    print(f"  Std  weights: {sample_actions.std(axis=0).round(4)}")
    print(f"  Min  weights: {sample_actions.min(axis=0).round(4)}")
    print(f"  Max  weights: {sample_actions.max(axis=0).round(4)}")

    price_feature_names    = get_price_feature_names()
    sentiment_feature_names = get_sentiment_feature_names()

    # ── Price Agent SHAP ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("PRICE AGENT SHAP ANALYSIS")
    print("="*50)

    # Total stock allocation
    price_predict_fn = make_predict_fn(price_agent, output_idx=None)
    price_shap, price_eval_obs = compute_shap_values(
        price_predict_fn, observations["price"],
        n_background=30, agent_label="price_agent"
    )

    # Save CSV
    price_df = save_shap_csv(
        price_shap, price_feature_names,
        "price_agent",
        "results/logs/shap/price_agent_shap.csv"
    )

    # Plot feature importance
    plot_shap_summary(
        price_shap, price_eval_obs, price_feature_names,
        title="Price Agent — Top Features by SHAP Importance\n(driving total stock allocation)",
        save_path="results/logs/shap/price_agent_feature_importance.png"
    )

    # Plot ticker comparison
    plot_shap_ticker_comparison(
        price_shap, price_feature_names,
        title="Price Agent — Feature Importance by Ticker",
        save_path="results/logs/shap/price_agent_ticker_importance.png"
    )

    print("\nTop 10 price agent features:")
    print(price_df.head(10).to_string(index=False))

    # ── Sentiment Agent SHAP ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SENTIMENT AGENT SHAP ANALYSIS")
    print("="*50)

    sentiment_predict_fn = make_predict_fn(sentiment_agent, output_idx=None)
    sentiment_shap, sent_eval_obs = compute_shap_values(
        sentiment_predict_fn, observations["sentiment"],
        n_background=30, agent_label="sentiment_agent"
    )

    # Save CSV
    sentiment_df = save_shap_csv(
        sentiment_shap, sentiment_feature_names,
        "sentiment_agent",
        "results/logs/shap/sentiment_agent_shap.csv"
    )

    # Plot feature importance
    plot_shap_summary(
        sentiment_shap, sent_eval_obs, sentiment_feature_names,
        title="Sentiment Agent — Top Features by SHAP Importance\n(driving total stock allocation)",
        save_path="results/logs/shap/sentiment_agent_feature_importance.png"
    )

    # Plot ticker comparison
    plot_shap_ticker_comparison(
        sentiment_shap, sentiment_feature_names,
        title="Sentiment Agent — Feature Importance by Ticker",
        save_path="results/logs/shap/sentiment_agent_ticker_importance.png"
    )

    print("\nTop 10 sentiment agent features:")
    print(sentiment_df.head(10).to_string(index=False))

    # ── Combined summary ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SHAP ANALYSIS COMPLETE")
    print("="*50)
    print("Files saved to results/logs/shap/:")
    for f in os.listdir("results/logs/shap"):
        print(f"  {f}")


# ── multi-seed SHAP on the test period ───────────────────────────────────────

SHAP_SEED    = 12345   # draws the explained / background steps and seeds KernelExplainer
N_EXPLAIN    = 200     # test-period observations explained per seed and agent
N_BACKGROUND = 100     # training-split observations used as background
TABLES_DIR   = "report/tables"
GROUPS       = ["price_indicators", "sentiment_features", "portfolio_cash_state"]
SEED_SETS    = {"seeds 0-3": [0, 1, 2, 3], "seeds 4-5": [4, 5]}


def feature_group(name: str) -> str:
    """price_indicators, sentiment_features or portfolio_cash_state."""
    if name.startswith("w_") or name == "portfolio_norm":
        return "portfolio_cash_state"
    if any(name.endswith("_" + f) for f in SENTIMENT_FEATURES):
        return "sentiment_features"
    return "price_indicators"


class RecordingAgent:
    """Delegates predict() to agent and keeps every observation it is given."""
    def __init__(self, agent):
        self.agent = agent
        self.obs   = []

    def predict(self, observation):
        self.obs.append(np.array(observation, dtype=np.float32))
        return self.agent.predict(observation)


def period_observations(env: StockTradingEnv, price_agent, sentiment_agent) -> tuple:
    """
    Observations of both agents, one per step, along the full-period
    MARL-no-coordinator episode (agent weights averaged every step), run by
    evaluate_agents.run_episode. Returns (price_obs, sent_obs, summary).
    """
    from evaluate_agents import run_episode
    price_rec, sent_rec = RecordingAgent(price_agent), RecordingAgent(sentiment_agent)
    summary = run_episode(env, price_rec, sent_rec)
    return np.array(price_rec.obs), np.array(sent_rec.obs), summary


def make_batch_predict_fn(agent):
    """
    All six portfolio weights of the deterministic policy for a batch of
    observations, mapped as in BaseAgent.predict (normalise_action):
    clip to [0, 1] and divide by the sum, all-zero -> 100% cash.
    """
    def predict_fn(obs_array):
        actions, _ = agent.model.predict(np.asarray(obs_array, dtype=np.float32),
                                         deterministic=True)
        actions = np.clip(actions.astype(np.float32), 0.0, 1.0)
        total   = actions.sum(axis=1, keepdims=True)
        cash    = np.zeros_like(actions)
        cash[:, -1] = 1.0
        return np.where(total > 1e-8, actions / np.maximum(total, 1e-8), cash)
    return predict_fn


def explain_agent(agent, background: np.ndarray, explained: np.ndarray,
                  label: str) -> np.ndarray:
    """
    KernelExplainer SHAP values of the six output weights, shape
    (n_explained, n_features, 6). Checks the batched predict against
    agent.predict and SHAP additivity.
    """
    predict_fn = make_batch_predict_fn(agent)
    outputs    = predict_fn(explained)
    single     = np.array([agent.predict(o) for o in explained])
    # batched and single-row float32 forward passes differ by ~1e-6
    if not np.allclose(outputs, single, atol=1e-5):
        raise RuntimeError(f"{label}: batched predict differs from agent.predict")
    print(f"  {label}: output weight std over explained obs "
          f"{outputs.std(axis=0).round(4)}")

    np.random.seed(SHAP_SEED)
    explainer = shap.KernelExplainer(predict_fn, background)
    values    = explainer.shap_values(explained, silent=True)
    values    = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)

    additivity = np.abs(values.sum(axis=1) + np.asarray(explainer.expected_value)
                        - outputs).max()
    print(f"  {label}: SHAP {values.shape}, max additivity error {additivity:.2e}")
    return values


def importance_rows(values: np.ndarray, feature_names: list,
                    seed: int, agent: str) -> tuple:
    """
    Long-format rows. Feature: mean over observations and the six output
    weights of |SHAP|. Group: SHAP summed over the group's features (SHAP
    is additive), then the same mean of |.|.
    """
    by_feature = np.abs(values).mean(axis=(0, 2))
    feat_rows  = [{"seed": seed, "agent": agent, "feature": f, "value": v}
                  for f, v in zip(feature_names, by_feature)]
    groups     = np.array([feature_group(f) for f in feature_names])
    group_rows = [{"seed": seed, "agent": agent, "feature": g,
                   "value": np.abs(values[:, groups == g, :].sum(axis=1)).mean()}
                  for g in GROUPS if (groups == g).any()]
    return feat_rows, group_rows


def check_return(summary: dict, results_csv: str, seed: int):
    """The recorded episode must reproduce the logged MARL-no-coordinator return."""
    df     = pd.read_csv(results_csv)
    row    = df[(df["strategy"] == "MARL-no-coordinator") & (df["train_seed"] == seed)]
    logged = float(row["total_return"].iloc[0])
    if not np.isclose(summary["total_return"], logged, rtol=0, atol=1e-9):
        raise RuntimeError(f"seed {seed}: recorded return {summary['total_return']} "
                           f"!= logged {logged} in {results_csv}")
    print(f"  return {summary['total_return']:.6f} matches {results_csv}")


def print_seed_stats(df: pd.DataFrame, title: str, top_n: int = None):
    """Per-seed values, then mean and std over seeds: all seeds and each SEED_SETS entry."""
    wide = df.pivot_table(index=["agent", "feature"], columns="seed", values="value")
    sets = {"all": sorted(df["seed"].unique()), **SEED_SETS}
    for name, seeds in sets.items():
        seeds = [s for s in seeds if s in wide.columns]
        wide[f"mean {name}"] = wide[seeds].mean(axis=1)
        wide[f"std {name}"]  = wide[seeds].std(axis=1, ddof=1)
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", "{:.5f}".format):
        for agent, part in wide.groupby(level="agent"):
            part = part.droplevel("agent").sort_values("mean all", ascending=False)
            print(f"\n[{agent}]" + (f" top {top_n} by mean over all seeds" if top_n else ""))
            print(part.head(top_n) if top_n else part)


def print_rank_correlation(feat_df: pd.DataFrame):
    """Spearman correlation of per-feature importance between seeds, per agent."""
    print("\nRank correlation (Spearman) of per-feature importance between seeds")
    low, high = SEED_SETS["seeds 0-3"], SEED_SETS["seeds 4-5"]
    for agent, part in feat_df.groupby("agent"):
        corr  = part.pivot(index="feature", columns="seed", values="value").corr(method="spearman")
        seeds = list(corr.columns)

        def mean_pairs(a, b):
            pairs = {tuple(sorted((i, j))) for i in a for j in b if i != j}
            return np.mean([corr.loc[i, j] for i, j in pairs])

        print(f"\n[{agent}]")
        print(corr.round(3).to_string())
        print(f"  mean pairwise, all seeds        : {mean_pairs(seeds, seeds):.3f}")
        print(f"  mean pairwise, within seeds 0-3 : {mean_pairs(low, low):.3f}")
        print(f"  seed 4 vs seed 5                : {mean_pairs(high, high):.3f}")
        print(f"  mean, seeds 0-3 vs seeds 4-5    : {mean_pairs(low, high):.3f}")
        for s in high:
            print(f"  mean, seed {s} vs seeds 0-3      : {mean_pairs([s], low):.3f}")


def run_multi_seed_shap(config: dict, train_seeds: list, root: str = None):
    """
    SHAP of the deterministic price and sentiment policies for every seed
    (checkpoints refused unless trained with the current setup):
    N_EXPLAIN test-period observations explained against N_BACKGROUND
    training-split observations, the steps drawn once with SHAP_SEED (same
    steps for every seed). Observations come from each seed's full-period
    MARL-no-coordinator episode. Writes report/tables/shap_by_feature.csv
    and shap_by_group.csv.
    """
    from evaluate_agents import make_test_env, make_train_env

    test_env  = make_test_env(config, full_period=True)
    train_env = make_train_env(config)
    rng       = np.random.default_rng(SHAP_SEED)
    test_idx  = np.sort(rng.choice(test_env.episode_length, N_EXPLAIN, replace=False))
    bg_idx    = np.sort(rng.choice(train_env.episode_length, N_BACKGROUND, replace=False))
    print(f"Explaining {N_EXPLAIN} of {test_env.episode_length} test steps, "
          f"background {N_BACKGROUND} of {train_env.episode_length} train steps "
          f"(seed {SHAP_SEED})")

    names = {"price": get_price_feature_names(), "sentiment": get_sentiment_feature_names()}
    feat_rows, group_rows = [], []

    for seed in train_seeds:
        print(f"\n=== seed {seed} ===")
        price_agent     = PriceAgent(test_env, config)
        sentiment_agent = SentimentAgent(test_env, config)
        load_checked_agents(checkpoint_dir(config, seed, root), config,
                            [price_agent, sentiment_agent])

        test_price, test_sent, summary = period_observations(test_env, price_agent, sentiment_agent)
        check_return(summary, "results/logs/dd_test/test_period_results.csv", seed)
        train_price, train_sent, summary = period_observations(train_env, price_agent, sentiment_agent)
        check_return(summary, "results/logs/dd_train/train_period_results.csv", seed)

        obs = {"price"    : (price_agent, train_price, test_price),
               "sentiment": (sentiment_agent, train_sent, test_sent)}
        for agent_name, (agent, train_obs, test_obs) in obs.items():
            values = explain_agent(agent, train_obs[bg_idx], test_obs[test_idx],
                                   f"seed {seed} {agent_name}")
            f, g = importance_rows(values, names[agent_name], seed, agent_name)
            feat_rows  += f
            group_rows += g

    feat_df  = pd.DataFrame(feat_rows)
    group_df = pd.DataFrame(group_rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    feat_df.to_csv(f"{TABLES_DIR}/shap_by_feature.csv", index=False)
    group_df.to_csv(f"{TABLES_DIR}/shap_by_group.csv", index=False)
    print(f"\nSaved → {TABLES_DIR}/shap_by_feature.csv, {TABLES_DIR}/shap_by_group.csv")

    print_seed_stats(group_df, "Mean |SHAP| per feature group (per seed, then mean/std over seeds)")
    print_seed_stats(feat_df, "Mean |SHAP| per feature", top_n=15)
    print_rank_correlation(feat_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP analysis of one training seed, "
                                                 "or of several with --seeds")
    parser.add_argument("--checkpoint-root", type=str, default=None,
                        help="folder holding the seed{N}/ checkpoint folders; "
                             "default: paths.checkpoints_root in config.yaml")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="multi-seed test-period SHAP (run_multi_seed_shap), "
                             "writing report/tables/shap_by_{feature,group}.csv")
    args = parser.parse_args()

    config = load_config()
    if args.seeds is not None:
        run_multi_seed_shap(config, args.seeds, args.checkpoint_root)
    else:
        run_shap_analysis(
            config,
            checkpoint_dir(config, args.train_seed, args.checkpoint_root),
        )