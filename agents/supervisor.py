import json
from typing import Literal

from pydantic import BaseModel

from lib.llm import get_llm

SYSTEM = """You are the Supervisor of a multi-agent restaurant intelligence system.
Workers: "monitor" (rating drop detection), "sentiment" (review text analysis, can apply a score penalty) and "alert" (drafts an owner alert from confirmed findings).
Decide who runs next based on the findings so far, or "finish".
Rules:
- Start with monitor, then sentiment.
- If the two disagree or either has low confidence, re-run the less certain one once with a specific instruction (for example fetch 40 reviews and re-verify).
- If a worker result contains an "error", retry it once with a simpler instruction, or finish if it fails again.
- Run "alert" only once, and only after monitor and sentiment agree that a real decline exists.
- Never run the same worker more than twice.
- Finish once the findings are consistent, the alert has run (or no decline was found), or the disagreement has been investigated.
Give a short reason and, when dispatching, a concrete instruction for the worker."""


class Decision(BaseModel):
    next: Literal["monitor", "sentiment", "alert", "finish"]
    reason: str
    instruction: str = ""


def decide(restaurant_id, findings, steps):
    llm = get_llm().with_structured_output(Decision, method="json_schema")
    prompt = f"{SYSTEM}\n\nRestaurant id: {restaurant_id}\nSteps so far: {steps}\nFindings: {json.dumps(findings, default=str)}"
    return llm.invoke(prompt)