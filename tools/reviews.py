from lib.db import get_conn


def get_reviews(restaurant_id, n=20):
    conn = get_conn()
    rows = conn.execute(
        "SELECT stars, text, created_at FROM reviews WHERE restaurant_id=? "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (restaurant_id, n),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_rating_trend(restaurant_id, window=10):
    conn = get_conn()
    stars = [
        r["stars"]
        for r in conn.execute(
            "SELECT stars FROM reviews WHERE restaurant_id=? ORDER BY created_at DESC, id DESC",
            (restaurant_id,),
        )
    ]
    conn.close()
    recent, older = stars[:window], stars[window:]
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    r, o = avg(recent), avg(older)
    return {
        "restaurant_id": restaurant_id,
        "recent_avg": r,
        "older_avg": o,
        "delta": round(r - o, 2) if r is not None and o is not None else None,
        "n_recent": len(recent),
        "n_older": len(older),
    }