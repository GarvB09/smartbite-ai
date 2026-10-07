from agents.base import run_agent
from lib.llm import parse_json
from tools.lc_tools import get_reviews, set_penalty

SYSTEM = """You are the Review Sentiment Agent for a restaurant platform.
Judge whether quality is declining from recent review text, before star ratings show it.
Call get_reviews. If you get fewer than 8 reviews or the picture is mixed, call it again with a larger n.
If sentiment is clearly negative with confidence >= 0.7, call set_penalty with a value between 0.1 and 0.8 (stronger decline, larger value) and a short reason. Otherwise do not set a penalty.
Reply with JSON only:
{"restaurant_id": int, "sentiment": "negative|neutral|positive", "trend": "declining|stable|improving", "confidence": float, "penalty_applied": float or null, "evidence": [short strings]}"""


def run(restaurant_id, instruction=""):
    task = f"Restaurant id: {restaurant_id}. {instruction}"
    return parse_json(run_agent("sentiment", SYSTEM, [get_reviews, set_penalty], task))