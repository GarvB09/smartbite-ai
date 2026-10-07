import json

from agents.base import run_agent
from lib.llm import parse_json
from tools.lc_tools import save_alert_draft

SYSTEM = """You are the Alert Agent. You turn confirmed findings from the Monitor and Sentiment agents into a short alert for the restaurant owner.
Only act when the findings confirm a real quality decline. If there is no decline, or the evidence is weak or contradictory, do not draft anything.
Write one concise draft (max 3 sentences): what changed, the evidence, and one suggested action. Then call save_alert_draft with audience "owner". Drafts stay pending until a human approves them.
Reply with JSON only:
{"restaurant_id": int, "alert_drafted": true|false, "alert_id": int or null, "summary": "short string"}"""


def run(restaurant_id, instruction="", findings=None):
    task = (
        f"Restaurant id: {restaurant_id}. {instruction}\n"
        f"Findings so far: {json.dumps(findings or {}, default=str)}"
    )
    return parse_json(run_agent("alert", SYSTEM, [save_alert_draft], task))
