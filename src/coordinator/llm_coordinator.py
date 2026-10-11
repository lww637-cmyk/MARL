"""
LLM coordination layer using Google Gemini API.
Receives suggested portfolio weights from both price and sentiment agents,
plus current market context, and outputs the final portfolio allocation.
Operates after both agents have acted each timestep.
"""

import os
import json
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]

# Configure Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(price_suggestion: np.ndarray,
                 sentiment_suggestion: np.ndarray,
                 context: dict) -> str:
    """
    Format agent suggestions and market context into a prompt for Gemini.

    price_suggestion      : array of 6 weights [AAPL,AMZN,MSFT,NVDA,TSLA,cash]
    sentiment_suggestion  : array of 6 weights [AAPL,AMZN,MSFT,NVDA,TSLA,cash]
    context               : dict with portfolio_value, timestep, returns_history
    """
    def format_weights(weights: np.ndarray) -> str:
        lines = []
        for i, ticker in enumerate(TICKERS):
            lines.append(f"  {ticker}: {weights[i]*100:.1f}%")
        lines.append(f"  Cash: {weights[-1]*100:.1f}%")
        return "\n".join(lines)

    portfolio_value   = context.get("portfolio_value", 100000)
    initial_capital   = context.get("initial_capital", 100000)
    portfolio_return  = (portfolio_value - initial_capital) / initial_capital * 100
    timestep          = context.get("timestep", 0)
    recent_returns    = context.get("recent_returns", [])

    if len(recent_returns) >= 5:
        recent_perf = np.mean(recent_returns[-5:]) * 100
        volatility  = np.std(recent_returns[-5:]) * 100
    else:
        recent_perf = 0.0
        volatility  = 0.0

    prompt = f"""You are a portfolio coordinator for an algorithmic trading system.
Two RL trading agents have analysed the market and suggested portfolio allocations.
Your role is to combine their suggestions into a final allocation.

PORTFOLIO STATUS:
- Current value: ${portfolio_value:,.2f}
- Total return: {portfolio_return:.2f}%
- Trading day: {timestep}
- Recent 5-day avg return: {recent_perf:.3f}%
- Recent volatility: {volatility:.3f}%

PRICE AGENT SUGGESTION (based on technical indicators):
{format_weights(price_suggestion)}

SENTIMENT AGENT SUGGESTION (based on company-specific news sentiment):
{format_weights(sentiment_suggestion)}

COORDINATION RULES:
1. If both agents agree on a ticker (both > 20%), increase its weight slightly
2. If agents strongly disagree on a ticker (one > 30%, other < 5%), be conservative — use the average
3. If recent volatility is high (> 2%), increase cash allocation
4. Always maintain some diversification — no single stock > 50%
5. Weights must sum to exactly 1.0

Respond ONLY with a valid JSON object in this exact format, no other text:
{{
  "AAPL": 0.20,
  "AMZN": 0.20,
  "MSFT": 0.20,
  "NVDA": 0.20,
  "TSLA": 0.20,
  "cash": 0.00
}}"""

    return prompt


def parse_response(response_text: str) -> np.ndarray:
    """
    Parse Gemini JSON response into portfolio weight array.
    Returns weights in order: [AAPL, AMZN, MSFT, NVDA, TSLA, cash]
    Falls back to equal weights if parsing fails.
    """
    try:
        # Strip markdown code blocks if present
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        weights_dict = json.loads(text.strip())

        weights = np.array([
            float(weights_dict.get("AAPL", 0.0)),
            float(weights_dict.get("AMZN", 0.0)),
            float(weights_dict.get("MSFT", 0.0)),
            float(weights_dict.get("NVDA", 0.0)),
            float(weights_dict.get("TSLA", 0.0)),
            float(weights_dict.get("cash", 0.0)),
        ], dtype=np.float32)

        # Normalise to sum to 1
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    except Exception as e:
        print(f"Warning: Failed to parse Gemini response: {e}")
        print(f"Response was: {response_text}")
        # Fall back to equal weights
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                        dtype=np.float32)


def coordinate(price_suggestion: np.ndarray,
               sentiment_suggestion: np.ndarray,
               context: dict,
               config: dict) -> np.ndarray:
    """
    Main coordination function.
    Calls Gemini API and returns final portfolio weights.
    Falls back to agent average if API call fails.
    """
    try:
        prompt   = build_prompt(price_suggestion, sentiment_suggestion, context)
        response = client.models.generate_content(
            model   = config["coordinator"]["model"],
            contents= prompt,
            config  = types.GenerateContentConfig(
                temperature        = config["coordinator"]["temperature"],
                max_output_tokens  = config["coordinator"]["max_tokens"],
                thinking_config    = types.ThinkingConfig(thinking_budget=0)
            )
        )
        weights = parse_response(response.text)
        return weights

    except Exception as e:
        print(f"Warning: Gemini API call failed: {e}")
        print("Falling back to agent average.")
        return _fallback_weights(price_suggestion, sentiment_suggestion)

def _fallback_weights(price_suggestion: np.ndarray,
                      sentiment_suggestion: np.ndarray) -> np.ndarray:
    """Average both agent suggestions as a safe fallback."""
    weights = (price_suggestion + sentiment_suggestion) / 2.0
    weights = np.clip(weights, 0.0, 1.0)
    total   = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.astype(np.float32)


def coordinate_or_average(price_suggestion: np.ndarray,
                           sentiment_suggestion: np.ndarray,
                           context: dict,
                           config: dict,
                           use_llm: bool = True) -> np.ndarray:
    """
    Wrapper that switches between LLM coordination and simple average.
    use_llm=False is used for the MARL-no-coordinator baseline.
    """
    if use_llm:
        return coordinate(price_suggestion, sentiment_suggestion,
                          context, config)
    else:
        return _fallback_weights(price_suggestion, sentiment_suggestion)