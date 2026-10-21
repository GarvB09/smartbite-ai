from agents.base import run_agent
from lib.llm import parse_json
from tools.lc_tools import get_price_drift, save_alert_draft

SYSTEM = """You are the Menu Price Intelligence Agent for SmartBite AI.
Your job:
1. Call get_price_drift for the given restaurant_id to check price history.
2. If n_points < 2, report that there is insufficient data and stop.
3. If |drift_pct| > 15, this is a significant change. Call save_alert_draft
   with audience "owner", describing the price change, the old and new price,
   and suggesting the owner verify or update their listed prices. Mark
   needs_retraining as true so the recommender can recalibrate.
4. If |drift_pct| <= 15, report that prices are stable.

Reply with JSON only:
{
  "restaurant_id": int,
  "drift_pct": float,
  "significant": true|false,
  "needs_retraining": true|false,
  "alert_id": int or null,
  "reason": "short explanation of your decision"
}"""


def run(restaurant_id: int, instruction: str = "") -> dict:
    task = (
        f"Restaurant id: {restaurant_id}. "
        f"{instruction} "
        "Check the menu price history and act if the drift exceeds 15%."
    )
    return parse_json(run_agent("price", SYSTEM, [get_price_drift, save_alert_draft], task))