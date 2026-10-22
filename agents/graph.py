"""
LangGraph graph definition — wires all 5 agents under the Supervisor.

State fields:
  restaurant_id   int   – the restaurant being analysed (0 = none)
  user_id         int   – the user making a request (0 = none)
  query           str   – natural-language user request (empty = quality-check mode)
  findings        dict  – agent name → list of result dicts
  steps           int   – how many supervisor decisions have been made
  next            str   – which worker runs next
  instruction     str   – instruction passed to the next worker
"""

import json
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents import alert, monitor, price, recommendation, sentiment, supervisor
from lib import trace
from lib.db import get_conn

MAX_STEPS = 12   # raised from 8 to allow 5-agent runs


class State(TypedDict):
    restaurant_id: int
    user_id: int
    query: str
    findings: dict
    steps: int
    next: str
    instruction: str


# ── Supervisor node ───────────────────────────────────────────────────

def supervisor_node(state):
    try:
        d = supervisor.decide(
            state["restaurant_id"],
            state["findings"],
            state["steps"],
            query=state.get("query", ""),
        )
    except Exception as e:
        d = supervisor.Decision(next="finish", reason=f"supervisor error: {e}")
    nxt = "finish" if state["steps"] >= MAX_STEPS else d.next
    trace.log("supervisor", nxt, {"instruction": d.instruction}, d.reason)
    return {"next": nxt, "instruction": d.instruction, "steps": state["steps"] + 1}


# ── Generic worker factory ────────────────────────────────────────────

def make_worker(name, module):
    def node(state):
        try:
            if name == "alert":
                # Alert receives all findings for context
                result = module.run(
                    state["restaurant_id"],
                    state["instruction"],
                    state["findings"],
                )
            elif name == "recommendation":
                # Recommendation agent needs user_id and the NL query
                result = module.run(
                    state["user_id"],
                    state.get("query", ""),
                )
            else:
                result = module.run(state["restaurant_id"], state["instruction"])
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        trace.log(name, "supervisor", result, "result")
        findings = dict(state["findings"])
        findings.setdefault(name, []).append(result)
        return {"findings": findings}

    return node


# ── Build and compile the graph ───────────────────────────────────────

def build():
    g = StateGraph(State)

    # Nodes
    g.add_node("supervisor", supervisor_node)
    g.add_node("recommendation", make_worker("recommendation", recommendation))
    g.add_node("monitor",        make_worker("monitor",        monitor))
    g.add_node("price",          make_worker("price",          price))
    g.add_node("sentiment",      make_worker("sentiment",      sentiment))
    g.add_node("alert",          make_worker("alert",          alert))

    # Entry
    g.set_entry_point("supervisor")

    # Supervisor routes to whichever worker it chooses, or END
    g.add_conditional_edges(
        "supervisor",
        lambda s: END if s["next"] == "finish" else s["next"],
        {
            "recommendation": "recommendation",
            "monitor":        "monitor",
            "price":          "price",
            "sentiment":      "sentiment",
            "alert":          "alert",
            END:              END,
        },
    )

    # Every worker returns to supervisor
    for worker in ("recommendation", "monitor", "price", "sentiment", "alert"):
        g.add_edge(worker, "supervisor")

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────

def run(restaurant_id: int = 0, user_id: int = 0, query: str = ""):
    """
    Run the multi-agent graph.

    Quality-check mode  : run(restaurant_id=2)
    Recommendation mode : run(user_id=1, query="cheap Indian food")
    Combined mode       : run(restaurant_id=2, user_id=1, query="is this place still good?")
    """
    conn = get_conn()
    if restaurant_id:
        exists = conn.execute(
            "SELECT 1 FROM restaurants WHERE id=?", (restaurant_id,)
        ).fetchone()
        if not exists:
            conn.close()
            raise ValueError(f"unknown restaurant id {restaurant_id}")
    if user_id:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not exists:
            conn.close()
            raise ValueError(f"unknown user id {user_id}")
    conn.close()

    trace.new_run()
    return build().invoke({
        "restaurant_id": restaurant_id,
        "user_id":       user_id,
        "query":         query,
        "findings":      {},
        "steps":         0,
        "next":          "",
        "instruction":   "",
    })


# ── CLI helper ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--restaurant", type=int, default=0)
    p.add_argument("--user",       type=int, default=0)
    p.add_argument("--query",      default="")
    a = p.parse_args()
    out = run(a.restaurant, a.user, a.query)
    print(json.dumps(out["findings"], indent=2, default=str))
