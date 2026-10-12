import json
import os
import shutil

import pandas as pd
import streamlit as st

import simulator
from agents import graph
from lib import trace
from lib.db import DB_PATH, get_conn, init_db
from tools import alerts, penalties, recommender
from tools.reviews import get_rating_trend

st.set_page_config(page_title="SmartBite AI", page_icon="🍽️", layout="wide")


def ensure_data():
    """On a fresh deployment the DB is empty; start from the bundled seed copy."""
    seed = os.path.join(os.path.dirname(DB_PATH), "seed.db")
    empty = True
    if os.path.exists(DB_PATH):
        conn = get_conn()
        try:
            empty = conn.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0] == 0
        except Exception:
            empty = True
        conn.close()
    if empty and os.path.exists(seed):
        shutil.copy(seed, DB_PATH)


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
ensure_data()
init_db()


def q(sql, args=()):
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    conn.close()
    return rows


def read_trace(run_id):
    if not os.path.exists(trace.PATH):
        return []
    with open(trace.PATH, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs if r["run_id"] == run_id]


restaurants = q("SELECT id, name, cuisine, city, rating FROM restaurants ORDER BY id")
if not restaurants:
    st.error("Database is empty. Run `python -m lib.load_data` first.")
    st.stop()

st.title("🍽️ SmartBite AI")
st.caption("Multi-agent restaurant quality monitoring: Supervisor, Monitor, Sentiment and Alert agents (LangGraph)")

with st.sidebar:
    st.header("Restaurant")
    labels = {r["id"]: f"{r['id']} · {r['name']} ({r['cuisine']})" for r in restaurants}
    rid = st.selectbox("Select", list(labels), format_func=labels.get)
    row = next(r for r in restaurants if r["id"] == rid)
    st.metric("Listed rating", row["rating"])
    t = get_rating_trend(rid)
    st.metric("Recent avg (10 reviews)", t["recent_avg"], t["delta"])
    pen = penalties.get_penalty(rid)
    st.metric("Ranking penalty", pen["value"] if pen else 0.0)

    st.header("Simulate an incident")
    n = st.slider("Bad reviews to inject", 1, 20, 8)
    if st.button("Inject bad reviews"):
        simulator.bad_reviews(rid, n)
        recommender._state.clear()
        st.rerun()
    if st.button("Clear penalty"):
        penalties.clear_penalty(rid)
        recommender._state.clear()
        st.rerun()

tab_run, tab_alerts, tab_rec = st.tabs(["Run agents", "Alerts (human approval)", "Recommendations"])

with tab_run:
    st.write("The Supervisor decides which agent runs next, based on what the others found.")
    if st.button("▶ Run multi-agent analysis", type="primary"):
        try:
            with st.spinner("Agents working..."):
                out = graph.run(rid)
            st.session_state["result"] = {"rid": rid, "findings": out["findings"], "run_id": trace._run["id"]}
            recommender._state.clear()
        except Exception as e:
            st.error(f"Run failed: {e}")
    res = st.session_state.get("result")
    if res and res["rid"] == rid:
        st.subheader("Findings")
        cols = st.columns(3)
        for col, name in zip(cols, ["monitor", "sentiment", "alert"]):
            with col:
                st.markdown(f"**{name.title()} agent**")
                for item in res["findings"].get(name, []):
                    st.json(item)
        st.subheader("Agent message trace")
        recs = read_trace(res["run_id"])
        st.dataframe(
            pd.DataFrame(
                [
                    {"from": r["from"], "to": r["to"], "reason": r["reason"], "content": json.dumps(r["content"], default=str)[:200]}
                    for r in recs
                ]
            ),
            width="stretch",
        )

with tab_alerts:
    pending = alerts.list_alerts()
    if not pending:
        st.info("No alerts yet. Run the analysis on a restaurant with declining reviews.")
    for a in pending:
        with st.container(border=True):
            st.markdown(f"**Alert #{a['id']}** · restaurant {a['restaurant_id']} · for {a['audience']} · status: `{a['status']}`")
            st.write(a["draft"])
            if a["status"] == "pending":
                c1, c2, _ = st.columns([1, 1, 6])
                if c1.button("Approve", key=f"ok{a['id']}"):
                    alerts.set_alert_status(a["id"], "approved")
                    st.rerun()
                if c2.button("Reject", key=f"no{a['id']}"):
                    alerts.set_alert_status(a["id"], "rejected")
                    st.rerun()

with tab_rec:
    users = q("SELECT id, name, favorite_cuisines FROM users ORDER BY id")
    uid = st.selectbox("User", [u["id"] for u in users], format_func=lambda i: f"user {i}")
    query = st.text_input("Search (optional)", "")
    recs = recommender.recommend(uid, query or None, 8)
    st.dataframe(pd.DataFrame(recs), width="stretch")
    st.caption("A penalty set by the Sentiment agent lowers a restaurant's score and shows up in the reasons.")
