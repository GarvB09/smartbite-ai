# SmartBite AI

**Multi-Agent Restaurant Intelligence Platform**  
CA3 Submission · Symbiosis Institute of Technology · AI-1 · Batch 2023-27  
Team: Jeel Patel (23070122110), Ishan Sinha (23070122106), Jayant Puri (23070122109), Jiya Tyagi (23070122111)

---

## What This Is

SmartBite AI uses **5 LLM-powered agents** coordinated by a **LangGraph supervisor** to:
1. Give customers personalised restaurant recommendations that automatically account for live quality signals.
2. Detect rating drops, price hikes, and review sentiment decline in real time.
3. Draft owner alerts — but **never send them without a human approving first**.

---

## 5 Agents

| # | Agent | Trigger | Tools |
|---|-------|---------|-------|
| 1 | Customer Recommendation | User query | `get_user_profile`, `get_recommendations` |
| 2 | Rating Monitor | Restaurant quality check | `get_rating_trend` |
| 3 | Price Intelligence | Restaurant quality check | `get_price_drift`, `save_alert_draft` |
| 4 | Sentiment & Penalty | After Agent 2 confirms drop | `get_reviews`, `set_penalty` |
| 5 | Alert (Human-in-Loop) | After Agents 2+4 confirm decline | `save_alert_draft` |

The **Supervisor** routes between agents using a Groq LLM call — not hard-coded edges.

---

## Quick Start

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd smartbite-ai
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_...your_key_here...
```
Get a free key at https://console.groq.com

### 5. Load the database
```bash
python -m lib.load_data
```
This copies `data/seed.db` into `data/smartbite.db`. If `seed.db` is present, no CSV parsing is needed.

### 6. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Running Individual Agents (CLI)

### Agent 1 — Recommendation
```bash
python -m agents.graph --user 3 --query "cheap Indian food near me"
```

### Agent 2, 3, 4, 5 — Quality Check
```bash
python -m agents.graph --restaurant 7
```

### Combined mode (both restaurant + user)
```bash
python -m agents.graph --restaurant 7 --user 3 --query "is this place still good?"
```

### Demo a full decline scenario
```bash
# Step 1: inject bad reviews to trigger the monitoring pipeline
python -c "import simulator; simulator.bad_reviews(7, 12)"

# Step 2: inject a price hike to trigger Agent 3
python -c "import simulator; simulator.change_price(7, 40)"

# Step 3: run all agents on restaurant 7
python -m agents.graph --restaurant 7

# Step 4: check the trace
cat trace.jsonl | python -m json.tool | head -100
```

---

## How to Demo for Faculty (Live Demo — Rubric Section C)

**Scenario 1 — Recommendation Agent (Agent 1)**
1. Open the app → "Recommendations" tab.
2. Select User 3 (Priya Sharma — likes Indian food).
3. Type "cheap spicy biryani" → click **Get Recommendations**.
4. Show the ranked list with score and penalty columns.
5. Explain: "Agent 1 fetched her profile, called the KNN recommender, and wrote this explanation. The penalty on Curry Palace was set by Agent 4 in a previous run."

**Scenario 2 — Quality Check Pipeline (Agents 2, 3, 4, 5)**
1. First inject bad reviews: sidebar → select restaurant 7 → "Inject bad reviews (12)".
2. Also inject a price change: "Inject price change (40%)".
3. Go to "Quality Check" tab → click **Run quality check**.
4. Show the four columns: Monitor found a drop, Price found a 40% hike, Sentiment applied a penalty, Alert drafted and waiting for approval.
5. Go to "Alerts" tab → show the pending alert → click **Approve**.
6. Explain: "This alert was never sent automatically. A human operator approved it first."

**Scenario 3 — Agent Trace (Rubric: Execution Traces)**
1. After any run, open the "Agent Trace" tab.
2. Select the latest run_id.
3. Walk through the trace: Supervisor → Monitor → Supervisor → Price → Supervisor → Sentiment → Supervisor → Alert → Finish.
4. Expand individual records to show the JSON structure.

---

## Project Structure

```
smartbite-ai/
├── agents/
│   ├── base.py            # Generic LLM tool-calling loop (max 6 steps)
│   ├── supervisor.py      # ← UPDATED: 5-agent LLM routing
│   ├── graph.py           # ← UPDATED: LangGraph with user_id/query State
│   ├── recommendation.py  # Agent 1
│   ├── monitor.py         # Agent 2
│   ├── price.py           # Agent 3
│   ├── sentiment.py       # Agent 4
│   └── alert.py           # Agent 5
├── tools/
│   ├── lc_tools.py        # All @tool wrappers (used by agents)
│   ├── recommender.py     # TF-IDF + penalty-aware scoring
│   ├── reviews.py         # get_reviews, get_rating_trend
│   ├── prices.py          # get_price_drift
│   ├── penalties.py       # set/get/clear penalty
│   └── alerts.py          # save/list/set_status alert
├── lib/
│   ├── db.py              # SQLite connection, init_db()
│   ├── llm.py             # get_llm() → ChatGroq
│   ├── trace.py           # trace.jsonl writer
│   └── load_data.py       # seed.db → smartbite.db
├── data/
│   ├── seed.db            # Pre-seeded SQLite database
│   └── zomato.csv         # Source data (300 restaurants)
├── docs/
│   ├── DESIGN.md                        # This document
│   ├── sample_trace_decline.jsonl        # Existing: full 5-step decline trace
│   ├── sample_trace_healthy.jsonl        # Existing: healthy restaurant (no alert)
│   ├── sample_trace_recommendation.jsonl # NEW: Agent 1 recommendation trace
│   └── sample_trace_price.jsonl          # NEW: Price + all 5 agents trace
├── tests/
│   ├── test_agents.py     # Integration tests (supervisor, crash handling)
│   └── test_core.py       # Unit tests (penalty ranking, rating trend, drift)
├── app.py                 # ← UPDATED: Streamlit UI with all 4 tabs
├── simulator.py           # Inject bad reviews / price changes
├── requirements.txt       # ← FIXED: no corrupted spacing
└── .env.example           # GROQ_API_KEY placeholder
```

---

## Files Changed vs Teammate's Base

Only these 3 files were updated to wire in Agents 1 and 3:

| File | What changed |
|------|-------------|
| `agents/supervisor.py` | Added `recommendation`, `price` to routing literal; added RECOMMENDATION and COMBINED mode routing rules to system prompt; added `query` parameter to `decide()` |
| `agents/graph.py` | Added `user_id` and `query` to `State`; added `recommendation` and `price` nodes; `make_worker` handles recommendation's different signature; MAX_STEPS raised to 12 |
| `app.py` | Added "Recommendations (Agent 1)" tab; added Agent 3 results column in Quality Check tab; added full Agent Trace tab |

Everything else (`agents/base.py`, all `tools/`, all `lib/`, `agents/monitor.py`, `agents/sentiment.py`, `agents/alert.py`, `simulator.py`, `tests/`) is **unchanged** from the teammate's original code.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_core.py::test_penalty_lowers_rank PASSED
tests/test_core.py::test_unknown_user_raises PASSED
tests/test_core.py::test_rating_trend PASSED
tests/test_core.py::test_price_drift PASSED
tests/test_agents.py::test_unknown_restaurant_rejected PASSED
tests/test_agents.py::test_worker_crash_handled PASSED
tests/test_agents.py::test_supervisor_finish_on_error PASSED
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Get free at console.groq.com |
| `MODEL` | No | `openai/gpt-oss-120b` | Any Groq-hosted model ID |

---

## Deployment (Streamlit Cloud)

1. Push to a public GitHub repo.
2. Go to https://share.streamlit.io → New app → select your repo → `app.py`.
3. In "Advanced settings → Secrets", add:
   ```toml
   GROQ_API_KEY = "gsk_...your_key..."
   ```
4. Click Deploy. The app will be live at `https://<your-app>.streamlit.app`.

Note: `data/seed.db` must be committed to the repo for the deployed app to have data.

---

## Rubric Checklist

| Section | Marks | Covered by |
|---------|-------|-----------|
| A — Multi-Agent Architecture | 8 | 5 agents + LangGraph supervisor + cyclic graph with LLM routing |
| B — Design | 8 | `docs/DESIGN.md`, agent loop diagram, supervisor routing modes |
| C — Live Demo | 6 | Streamlit tabs: Recommendation, Quality Check, Alerts, Trace |
| D — Presentation | 5 | Demo scenarios above |
| E — Documentation | 5 | This README + DESIGN.md + in-code docstrings |
| F — Code Review | 5 | Clean separation: agents/, tools/, lib/; pytest; SOLID principles |
| G — Individual Contributions | 5 | GitHub commits per member; DESIGN.md contribution table |
