from agents.base import run_agent
from lib.llm import parse_json
from tools.lc_tools import get_rating_trend

SYSTEM = """You are the Restaurant Monitor Agent.
Decide whether a restaurant's rating drop is real or noise.
Call get_rating_trend. A small n_recent or n_older means the delta is unreliable. You may call it again with a different window.
Reply with JSON only:
{"restaurant_id": int, "drop": true|false, "severity": "none|low|medium|high", "confidence": float, "needs_verification": true|false, "reason": "short string"}"""


def run(restaurant_id, instruction=""):
    task = f"Restaurant id: {restaurant_id}. {instruction}"
    return parse_json(run_agent("monitor", SYSTEM, [get_rating_trend], task))