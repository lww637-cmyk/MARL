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
from src.environment.stock_env import TICKERS

load_dotenv()

# Configure Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(price_suggestion: np.ndarray,
                 sentiment_suggestion: np.ndarray,
                 context: dict) -> str:
    """
    Format agent suggestions and market context into a prompt for Gemini.

    price_suggestion      : array of weights [*TICKERS, cash]
    sentiment_suggestion  : array of weights [*TICKERS, cash]
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

    equal_weight  = round(1.0 / len(TICKERS), 2)
    json_template = ",\n".join(
        [f'  "{t}": {equal_weight:.2f}' for t in TICKERS] + ['  "cash": 0.00']
    )

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
{json_template}
}}"""

    return prompt


# outcome of one coordinator call, as written to the call log
OK            = "ok"
API_ERROR     = "api_error_agent_average"
PARSE_ERROR   = "parse_error_equal_weights"
ALL_ZERO      = "all_zero_weights"   # falls back to the agent average

EQUAL_WEIGHTS = np.array([1.0 / len(TICKERS)] * len(TICKERS) + [0.0],
                         dtype=np.float32)


def parse_response(response_text: str) -> np.ndarray:
    """
    Parse Gemini JSON response into portfolio weight array.
    Returns the weights as given, in order [*TICKERS, cash], not clipped
    or normalised. Raises ValueError if the text is not a JSON object of
    finite numbers.
    """
    # Strip markdown code blocks if present
    text = response_text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    weights_dict = json.loads(text.strip())
    if not isinstance(weights_dict, dict):
        raise ValueError(f"expected a JSON object, got {type(weights_dict).__name__}")

    weights = np.array(
        [float(weights_dict.get(t, 0.0)) for t in TICKERS]
        + [float(weights_dict.get("cash", 0.0))],
        dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("non-finite weight")
    return weights


def _token_counts(response) -> dict:
    """Token counts from response.usage_metadata (None where missing)."""
    usage = getattr(response, "usage_metadata", None)
    return {name: getattr(usage, name, None) for name in (
        "prompt_token_count", "candidates_token_count",
        "thoughts_token_count", "total_token_count")}


def coordinate(price_suggestion: np.ndarray,
               sentiment_suggestion: np.ndarray,
               context: dict,
               config: dict,
               record: dict = None) -> np.ndarray:
    """
    Main coordination function.
    Calls Gemini API and returns final portfolio weights.
    API failure            -> agent average      (API_ERROR)
    unparseable response   -> equal weights      (PARSE_ERROR)
    weights all <= 0       -> agent average      (ALL_ZERO)
    otherwise negatives are clipped to 0 and the weights normalised (OK).
    record: if given, filled with prompt, raw response text, parsed and
    final weights, outcome and token counts for the call log.
    """
    if record is None:
        record = {}
    prompt = build_prompt(price_suggestion, sentiment_suggestion, context)
    record.update(prompt=prompt, response_text=None, parsed_weights=None,
                  error=None, **_token_counts(None))

    try:
        response = client.models.generate_content(
            model   = config["coordinator"]["model"],
            contents= prompt,
            config  = types.GenerateContentConfig(
                temperature        = config["coordinator"]["temperature"],
                max_output_tokens  = config["coordinator"]["max_tokens"],
                thinking_config    = types.ThinkingConfig(thinking_budget=0)
            )
        )
        record.update(_token_counts(response))
        record["response_text"] = response.text
    except Exception as e:
        print(f"Warning: Gemini API call failed: {e}")
        print("Falling back to agent average.")
        weights = _fallback_weights(price_suggestion, sentiment_suggestion)
        record.update(outcome=API_ERROR, error=repr(e))
        record["final_weights"] = [float(w) for w in weights]
        return weights

    try:
        parsed = parse_response(record["response_text"])
    except Exception as e:
        print(f"Warning: Failed to parse Gemini response: {e}")
        print(f"Response was: {record['response_text']}")
        weights = EQUAL_WEIGHTS.copy()
        record.update(outcome=PARSE_ERROR, error=repr(e))
    else:
        record["parsed_weights"] = [float(w) for w in parsed]
        weights = np.clip(parsed, 0.0, None)
        total   = weights.sum()
        if total > 0:
            weights = (weights / total).astype(np.float32)
            record["outcome"] = OK
        else:
            print("Warning: Gemini returned no positive weight. "
                  "Falling back to agent average.")
            weights = _fallback_weights(price_suggestion, sentiment_suggestion)
            record["outcome"] = ALL_ZERO

    record["final_weights"] = [float(w) for w in weights]
    return weights

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
                           use_llm: bool = True,
                           record: dict = None) -> np.ndarray:
    """
    Wrapper that switches between LLM coordination and simple average.
    use_llm=False is used for the MARL-no-coordinator baseline.
    record is passed to coordinate() (filled only when use_llm=True).
    """
    if use_llm:
        return coordinate(price_suggestion, sentiment_suggestion,
                          context, config, record=record)
    else:
        return _fallback_weights(price_suggestion, sentiment_suggestion)