from langchain_core.tools import tool

from tools import penalties as _penalties
from tools import prices as _prices
from tools import reviews as _reviews


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
    return _penalties.set_penalty(restaurant_id, value, reason)