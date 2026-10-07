import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lib.db import get_conn

_state = {}


def fit():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT r.id, r.name, r.cuisine, r.city, r.rating, "
        "(SELECT price FROM menu_prices m WHERE m.restaurant_id=r.id ORDER BY id DESC LIMIT 1) AS price "
        "FROM restaurants r",
        conn,
    )
    conn.close()
    span = df["price"].max() - df["price"].min()
    df["cheapness"] = 1 - (df["price"] - df["price"].min()) / span if span else 0.5
    df["cheapness"] = df["cheapness"].fillna(0.5)
    text = df["name"] + " " + df["cuisine"] + " " + df["city"]
    vec = TfidfVectorizer(stop_words="english")
    _state.update(df=df, vec=vec, matrix=vec.fit_transform(text))
    return {"restaurants": len(df)}


def recommend(user_id, query=None, k=5):
    if not _state:
        fit()
    conn = get_conn()
    user = conn.execute("SELECT favorite_cuisines FROM users WHERE id=?", (user_id,)).fetchone()
    penalties = {r["restaurant_id"]: r["value"] for r in conn.execute("SELECT restaurant_id, value FROM penalties")}
    conn.close()
    if user is None:
        raise ValueError(f"unknown user {user_id}")

    df = _state["df"].copy()
    favs = [c.strip() for c in user["favorite_cuisines"].split(",")]
    df["taste"] = df["cuisine"].isin(favs).astype(float) * 0.3
    if query:
        sims = cosine_similarity(_state["vec"].transform([query]), _state["matrix"]).ravel()
        df["match"] = sims * 0.8
    else:
        df["match"] = 0.0
    df["penalty"] = df["id"].map(penalties).fillna(0.0)
    df["score"] = df["rating"] / 5 + df["taste"] + df["match"] + df["cheapness"] * 0.1 - df["penalty"]

    out = []
    for _, r in df.sort_values("score", ascending=False).head(k).iterrows():
        reasons = []
        if r["taste"]:
            reasons.append("matches your favourite cuisine")
        if r["match"] > 0.05:
            reasons.append("matches your search")
        if r["penalty"]:
            reasons.append(f"penalty {r['penalty']:.2f} applied for recent quality decline")
        out.append(
            {
                "restaurant_id": int(r["id"]),
                "name": r["name"],
                "cuisine": r["cuisine"],
                "rating": float(r["rating"]),
                "score": round(float(r["score"]), 3),
                "penalty": float(r["penalty"]),
                "reasons": reasons,
            }
        )
    return out