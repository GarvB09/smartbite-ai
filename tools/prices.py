from lib.db import get_conn


def get_price_history(restaurant_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT item, price, recorded_at FROM menu_prices WHERE restaurant_id=? ORDER BY id",
        (restaurant_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_price_drift(restaurant_id):
    history = get_price_history(restaurant_id)
    if len(history) < 2:
        return {"restaurant_id": restaurant_id, "drift_pct": 0.0, "n_points": len(history)}
    first, last = history[0]["price"], history[-1]["price"]
    return {
        "restaurant_id": restaurant_id,
        "first": first,
        "last": last,
        "drift_pct": round((last - first) / first * 100, 2),
        "n_points": len(history),
    }