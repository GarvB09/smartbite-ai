# SmartBite AI — System Design Document

**CA3 · Multi-Agent AI · Symbiosis Institute of Technology**  
**Team:** Jeel Patel (23070122110) · Ishan Sinha (23070122106) · Jayant Puri (23070122109) · Jiya Tyagi (23070122111)

---

## 1. Problem Statement

Restaurant platforms like Zomato and Swiggy surface stale ratings and show declining restaurants equally alongside healthy ones. Customers book based on outdated signals; restaurant owners learn about quality drops only through complaints. SmartBite AI is a real-time, multi-agent intelligence layer that continuously monitors restaurant health, personalises recommendations by factoring in live quality signals, and alerts human operators before problems escalate — with a mandatory human approval step before any alert is sent.

---

## 2. Architecture Overview

SmartBite AI uses an **orchestrator-worker multi-agent pattern** implemented with **LangGraph** (cyclic graph, not a linear pipeline). A central Supervisor agent uses an LLM call to decide at each step which specialist worker to invoke next, based on accumulated findings. This is genuine agentic reasoning — the routing decision is not hard-coded.

```
                         ┌─────────────────────────────────┐
                         │          SUPERVISOR              │
                         │   (LLM structured-output call)   │
                         │  Decision: next | reason |       │
                         │           instruction            │
                         └──────┬──────────────────────────┘
                                │ dispatches to one worker
              ┌─────────────────┼──────────────────────────────────┐
              │                 │                                  │
    ┌─────────▼──────┐ ┌────────▼───────┐  ┌──────────────────────▼──┐
    │   Agent 1      │ │   Agent 2      │  │      Agent 3            │
    │ Recommendation │ │   Monitor      │  │   Price Intelligence    │
    │ (user_id+query)│ │ (rating trend) │  │   (menu price drift)    │
    └────────────────┘ └────────────────┘  └─────────────────────────┘
              │                 │                                  │
              │        ┌────────▼────────┐  ┌────────────────────▼──┐
              │        │   Agent 4       │  │      Agent 5          │
              │        │   Sentiment     │  │   Alert (human-in-    │
              │        │ (reviews + pen) │  │    the-loop draft)    │
              │        └─────────────────┘  └───────────────────────┘
              │                 │                        │
              └─────────────────┴──── all return to ────┘
                                           SUPERVISOR
```

Every worker node returns findings to the Supervisor, which decides the next step. The graph runs until the Supervisor decides to "finish" or MAX\_STEPS (12) is reached.

---

## 3. Agent Specifications

### Agent 1 — Customer Recommendation Agent
- **Trigger:** User submits a natural-language food query (e.g. "cheap Indian food").
- **Tools:** `get_user_profile`, `get_recommendations`
- **Loop:** LLM → fetch profile → get recs → if <3 results, broaden query and retry once → write personalised explanation → return.
- **Key behaviour:** Recommendations are **penalty-aware** — restaurants penalised by Agent 4 appear lower in rankings automatically, without any additional logic in Agent 1. The Perceive→Decide→Act loop is: perceive the user query + profile, decide what cuisine/params to pass, act by calling the tool.
- **Output JSON:** `{ user_id, query_interpreted, recommendations: [...], explanation }`

### Agent 2 — Rating Monitor Agent
- **Trigger:** Supervisor dispatches after restaurant_id is provided.
- **Tools:** `get_rating_trend`
- **Loop:** LLM → fetch trend → decide if drop is real (confidence threshold 0.6) → return severity/confidence JSON.
- **Key behaviour:** Distinguishes noise from real decline using `delta` vs `n_reviews`. If confidence is low, Supervisor may re-dispatch with more context.
- **Output JSON:** `{ restaurant_id, drop, severity, confidence, reason }`

### Agent 3 — Menu Price Intelligence Agent
- **Trigger:** Supervisor dispatches (often parallel to or after Agent 2).
- **Tools:** `get_price_drift`, `save_alert_draft`
- **Loop:** LLM → fetch price history → if |drift_pct| > 15%, decide to save alert draft → return.
- **Key behaviour:** LLM can reason contextually — a 16% rise over a year is acceptable; a 40% rise in a week is not. The decision to alert is LLM-driven, not purely threshold-based.
- **Output JSON:** `{ restaurant_id, drift_pct, significant, needs_retraining, alert_id, reason }`

### Agent 4 — Sentiment & Penalty Agent
- **Trigger:** Supervisor dispatches, usually after Agent 2 confirms a numeric drop.
- **Tools:** `get_reviews`, `set_penalty`
- **Loop:** LLM → fetch 10 latest reviews → analyse text sentiment → if negative trend with confidence > 0.6, call `set_penalty` → return.
- **Key behaviour:** The penalty persists in SQLite and is automatically picked up by `get_recommendations` in future Agent 1 runs. This creates a genuine **feedback loop between agents**.
- **Output JSON:** `{ restaurant_id, sentiment, trend, penalty_applied, confidence, themes }`

### Agent 5 — Alert & Notification Agent (Human-in-the-Loop)
- **Trigger:** Supervisor dispatches only when Agents 2 and 4 both confirm a real decline.
- **Tools:** `save_alert_draft`
- **Loop:** LLM → synthesise findings from all previous agents → write a structured alert for the restaurant owner → save as `pending` (never auto-sends).
- **Key behaviour:** Implements **mandatory human-in-the-loop**. The draft is surfaced in the Streamlit UI where a human operator approves or rejects it. Agent 5 never sends anything autonomously.
- **Output JSON:** `{ alert_id, status: "pending", audience, summary }`

---

## 4. Supervisor — LLM-Driven Routing

The Supervisor is not a rule-based router. At each step it makes an LLM call using Groq (`openai/gpt-oss-120b`) with `with_structured_output(Decision)` where:

```python
class Decision(BaseModel):
    next: Literal["recommendation", "monitor", "price", "sentiment", "alert", "finish"]
    reason: str
    instruction: str = ""
```

The LLM receives: restaurant\_id, user query, step count, and the full findings dict. It decides which agent to call next, or "finish". This satisfies the rubric requirement for genuine LLM-based orchestration, not hard-coded edges.

Three routing modes are encoded in the Supervisor's system prompt:
- **Quality Check mode** (restaurant\_id only): Monitor → Price/Sentiment → Alert if confirmed.
- **Recommendation mode** (query only): Recommendation → Finish.
- **Combined mode** (both): Recommendation + Monitor + Sentiment.

---

## 5. Human-in-the-Loop Design

Agent 5's alerts are stored with `status = "pending"` in the `alerts` table. The Streamlit UI surfaces all pending alerts in the "🔔 Alerts" tab. A human operator reads the draft and clicks **Approve** or **Reject**. `set_alert_status()` updates the row. This pattern ensures no automated message reaches a restaurant owner without explicit human sign-off — a requirement the CA3 faculty specifically asked for.

---

## 6. Agent Loop (Perceive → Decide → Act → Learn)

Each specialist agent runs inside `agents/base.py`'s `run_agent()` loop:
1. **Perceive** — receives task string from Supervisor (restaurant\_id, instruction, prior findings).
2. **Decide** — LLM selects which tool to call, with what parameters.
3. **Act** — tool executes (DB query, penalty write, alert save).
4. **Learn** — findings are appended to the shared State `findings` dict, which is passed back to the Supervisor. Future Supervisor decisions are conditioned on all prior findings.

The feedback loop between Agent 4 (penalty write) and Agent 1 (penalty-aware scoring) is the clearest example of learning across the multi-agent system.

---

## 7. Data Store

**SQLite** (`data/smartbite.db`), 6 tables:

| Table | Purpose |
|-------|---------|
| `restaurants` | 300 restaurants from Zomato dataset |
| `reviews` | Text reviews with timestamps |
| `menu_prices` | Price history with timestamps (enables drift calculation) |
| `users` | 20 synthetic users with cuisine preferences and order history |
| `penalties` | Active ranking penalties per restaurant (written by Agent 4, read by Agent 1) |
| `alerts` | Alert drafts from Agent 5, status: pending/approved/rejected |

---

## 8. Execution Trace

Every agent message and tool call is appended to `trace.jsonl` by `lib/trace.py`. Each record contains:
```json
{
  "run_id": "uuid",
  "ts":     "ISO-8601",
  "from":   "supervisor | agent-name | tool",
  "to":     "agent-name | supervisor | tool",
  "content": { ... },
  "reason": "why this message was sent"
}
```
The Streamlit "Agent Trace" tab renders all runs and individual messages. Sample traces for all 5-agent scenarios are in `docs/`.

---

## 9. Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph (cyclic StateGraph) |
| LLM | Groq API — `openai/gpt-oss-120b` (free tier) |
| Tool protocol | LangChain `@tool` decorator |
| Structured output | Pydantic v2 `BaseModel` + `with_structured_output` |
| Recommender | scikit-learn TF-IDF + cosine similarity, penalty-weighted |
| Database | SQLite (via Python `sqlite3`) |
| UI | Streamlit |
| Testing | pytest |
| Data | Zomato open dataset (300 restaurants, 20 synthetic users) |

---

## 10. Individual Contributions

| Member | Role | Contribution |
|--------|------|-------------|
| Ishan Sinha (23070122106) | Core infrastructure | `lib/` (db, llm, trace), `tools/` (reviews, prices, penalties, alerts, lc_tools), `agents/base.py`, `data/` pipeline, `simulator.py` |
| Jayant Puri (23070122109) | Worker agents | `agents/monitor.py` (Agent 2), `agents/sentiment.py` (Agent 4), `agents/alert.py` (Agent 5), `tests/test_core.py` |
| Jeel Patel (23070122110) | Orchestration + UI | `agents/supervisor.py`, `agents/graph.py` (5-agent wiring), `app.py` (Recommendation tab, Quality tab, Alerts tab, Trace tab) |
| Jiya Tyagi (23070122111) | Recommendation system | `agents/recommendation.py` (Agent 1), `agents/price.py` (Agent 3), `tools/recommender.py` (TF-IDF + penalty scoring), `tests/test_agents.py`, documentation |
