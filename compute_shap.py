"""
SHAP explainability analysis for trained MARL trading agents.
Computes feature attribution for both price and sentiment agents.
Saves SHAP values and plots to results/logs/shap/
"""

import os
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

def run_shap_analysis(config: dict,
                       price_ckpt: str,
                       sentiment_ckpt: str):
    """Full SHAP analysis pipeline for both agents."""

    os.makedirs("results/logs/shap", exist_ok=True)
    np.random.seed(42)

    # Load environment and agents
    print("Initialising environment and agents...")
    env = StockTradingEnv(config, mode="train")

    price_agent     = PriceAgent(env, config)
    sentiment_agent = SentimentAgent(env, config)

    if os.path.exists(price_ckpt):
        price_agent.load(price_ckpt)
    else:
        print(f"WARNING: {price_ckpt} not found")

    if os.path.exists(sentiment_ckpt):
        sentiment_agent.load(sentiment_ckpt)
    else:
        print(f"WARNING: {sentiment_ckpt} not found")

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


if __name__ == "__main__":
    config = load_config()
    run_shap_analysis(
        config,
        price_ckpt     = "results/checkpoints/price_agent_final.zip",
        sentiment_ckpt = "results/checkpoints/sentiment_agent_final.zip",
    )