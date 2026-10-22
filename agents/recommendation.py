from agents.base import run_agent
from lib.llm import parse_json
from tools.lc_tools import get_recommendations, get_user_profile

"""
Agent 1 — Customer Recommendation Agent
========================================
Triggered by the Supervisor in Recommendation or Combined mode.

Workflow:
  1. Calls get_user_profile to fetch the user's favourite cuisines.
  2. Calls get_recommendations with the NL query; retries with a broader
     query if fewer than 3 results are returned.
  3. Returns structured JSON with a ranked list and a plain-English
     explanation of the top pick, factoring in any active penalties
     written by the Sentiment agent.
"""



SYSTEM = """You are the Customer Recommendation Agent for SmartBite AI.
Your job has three steps:

1. Call get_user_profile to understand the user's favourite cuisines.
2. Call get_recommendations with the structured query you extract from the
   user's natural-language request and the user_id. Parameters:
     - user_id (int, required)
     - query (str, optional)
     - cuisine_hint (str, optional)
     - max_results (int, default 5)
   If fewer than 3 restaurants are returned, call it again with a broader query.
3. Write a short, friendly explanation (2-3 sentences) for why the top
   result is a great match, referencing the user's taste profile and any
   penalty notes in the results.

Reply with JSON only:
{
  "user_id": int,
  "query_interpreted": "what you understood the user to want",
  "recommendations": [],
  "explanation": "friendly 2-3 sentence summary for the top pick"
}"""


def run(user_id: int, query: str = "") -> dict:
    task = (
        f"User id: {user_id}. "
        f"Their request: \"{query or 'show me good restaurants'}\". "
        "Fetch their profile, get personalised recommendations, and explain the top pick."
    )
    return parse_json(run_agent("recommendation", SYSTEM, [get_user_profile, get_recommendations], task))