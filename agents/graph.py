import json
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents import alert, monitor, sentiment, supervisor
from lib import trace
from lib.db import get_conn

MAX_STEPS = 8


class State(TypedDict):
    restaurant_id: int
    findings: dict
    steps: int
    next: str
    instruction: str


def supervisor_node(state):
    try:
        d = supervisor.decide(state["restaurant_id"], state["findings"], state["steps"])
    except Exception as e:
        d = supervisor.Decision(next="finish", reason=f"supervisor error: {e}")
    nxt = "finish" if state["steps"] >= MAX_STEPS else d.next
    trace.log("supervisor", nxt, {"instruction": d.instruction}, d.reason)
    return {"next": nxt, "instruction": d.instruction, "steps": state["steps"] + 1}


def make_worker(name, module):
    def node(state):
        try:
            if name == "alert":
                result = module.run(state["restaurant_id"], state["instruction"], state["findings"])
            else:
                result = module.run(state["restaurant_id"], state["instruction"])
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        trace.log(name, "supervisor", result, "result")
        findings = dict(state["findings"])
        findings.setdefault(name, []).append(result)
        return {"findings": findings}

    return node


def build():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor_node)
    g.add_node("monitor", make_worker("monitor", monitor))
    g.add_node("sentiment", make_worker("sentiment", sentiment))
    g.add_node("alert", make_worker("alert", alert))
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: END if s["next"] == "finish" else s["next"],
        {"monitor": "monitor", "sentiment": "sentiment", "alert": "alert", END: END},
    )
    g.add_edge("monitor", "supervisor")
    g.add_edge("sentiment", "supervisor")
    g.add_edge("alert", "supervisor")
    return g.compile()


def run(restaurant_id):
    conn = get_conn()
    exists = conn.execute("SELECT 1 FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    conn.close()
    if not exists:
        raise ValueError(f"unknown restaurant id {restaurant_id}")
    trace.new_run()
    return build().invoke(
        {"restaurant_id": restaurant_id, "findings": {}, "steps": 0, "next": "", "instruction": ""}
    )


if __name__ == "__main__":
    out = run(int(sys.argv[1]))
    print(json.dumps(out["findings"], indent=2))