"""
Supervisor Agent — Orchestrator for SmartBite AI's 5-agent system.

Workers it can dispatch:
  - recommendation  (Agent 1) – personalised restaurant recommendations
  - monitor         (Agent 2) – rating drop detection
  - price           (Agent 3) – menu price drift detection
  - sentiment       (Agent 4) – review text sentiment + penalty
  - alert           (Agent 5) – draft owner/customer notification

Routing is decided by an LLM call (not hard-coded edges), so the
Supervisor genuinely reasons about what to do next based on findings.
"""

import json
from typing import Literal

from pydantic import BaseModel

from lib.llm import get_llm

SYSTEM = """You are the Supervisor of SmartBite AI, a 5-agent restaurant intelligence system.

Workers available:
  "recommendation"  – Agent 1: personalised restaurant recommendations using user profile + NL query
  "monitor"         – Agent 2: detects whether a rating drop is real or noise (numeric)
  "price"           – Agent 3: detects significant menu price changes (>15%)
  "sentiment"       – Agent 4: analyses recent review text, applies ranking penalty if declining
  "alert"           – Agent 5: drafts an owner alert from confirmed findings, requires human approval

Routing rules:
  QUALITY CHECK mode (restaurant_id provided, query is empty or None):
    - Start with monitor (Agent 2), then sentiment (Agent 4).
    - Run price (Agent 3) in parallel reasoning — after monitor or after sentiment.
    - If monitor and sentiment disagree or confidence < 0.6, re-run the less certain one once
      with a more specific instruction (e.g. fetch more reviews).
    - Run alert (Agent 5) only when monitor AND sentiment agree a real decline exists.
    - Never run alert if there is no confirmed decline.

  RECOMMENDATION mode (query is non-empty):
    - Run recommendation (Agent 1) directly. No need for monitor/sentiment/alert.
    - Finish after recommendation returns results.

  COMBINED mode (both restaurant_id and query are meaningful):
    - Run recommendation first for the user.
    - Then run monitor + sentiment on the restaurant to give the user a quality signal.

General rules:
  - Never run the same worker more than twice.
  - If a worker result contains "error", retry once with a simpler instruction; finish if it fails again.
  - Finish once the task is complete, findings are consistent, or MAX_STEPS is near.
  - Always give a concrete instruction when dispatching a worker.
  - Give a short reason for every decision.
"""


class Decision(BaseModel):
    next: Literal["recommendation", "monitor", "price", "sentiment", "alert", "finish"]
    reason: str
    instruction: str = ""


def decide(restaurant_id, findings, steps, query=""):
    llm = get_llm().with_structured_output(Decision, method="json_schema")
    prompt = (
        f"{SYSTEM}\n\n"
        f"Restaurant id: {restaurant_id}\n"
        f"User query: {query!r}\n"
        f"Steps so far: {steps}\n"
        f"Findings: {json.dumps(findings, default=str)}"
    )
    return llm.invoke(prompt)
