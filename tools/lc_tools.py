from langchain_core.tools import tool

from tools import alerts as _alerts
from tools import penalties as _penalties
from tools import prices as _prices
from tools import reviews as _reviews
from tools import recommender as _recommender
from lib.db import get_conn


@tool
def get_reviews(restaurant_id: int, n: int = 20) -> list:
    """Fetch the n most recent reviews (stars, text, created_at) for a restaurant."""
    return _reviews.get_reviews(restaurant_id, n)


@tool
def get_rating_trend(restaurant_id: int, window: int = 10) -> dict:
    """Compare the average stars of the newest `window` reviews against all older ones."""
    return _reviews.get_rating_trend(restaurant_id, window)


@tool
def get_price_drift(restaurant_id: int) -> dict:
    """Percent change between the first and latest recorded price of a restaurant."""
    return _prices.get_price_drift(restaurant_id)


@tool
def set_penalty(restaurant_id: int, value: float, reason: str) -> dict:
    """Apply a recommendation score penalty (0.1 to 0.8) to a restaurant with a short reason."""
    value = max(0.1, min(0.8, float(value)))  # clamp to safe range
    return _penalties.set_penalty(restaurant_id, value, reason)


@tool
def save_alert_draft(restaurant_id: int, audience: str, draft: str) -> dict:
    """Save a pending alert draft for human approval. audience is 'owner' or 'customers'."""
    return _alerts.save_alert_draft(restaurant_id, audience, draft)


@tool
def get_user_profile(user_id: int) -> dict:
    """Fetch a user's name and favourite cuisines from the database."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, favorite_cuisines FROM users WHERE id=?", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"unknown user {user_id}"}
    return {"user_id": row["id"], "name": row["name"], "favorite_cuisines": row["favorite_cuisines"]}


@tool
def get_recommendations(user_id: int, query: str = "", cuisine_hint: str = "", max_results: int = 5) -> list:
    """Return personalised restaurant recommendations for a user, penalty-aware."""
    results = _recommender.recommend(user_id, query or cuisine_hint or None, max_results)
    return results