from lib.db import get_conn


def search_restaurants(query, limit=10):
    like = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, cuisine, city, rating FROM restaurants "
        "WHERE name LIKE ? OR cuisine LIKE ? OR city LIKE ? ORDER BY rating DESC LIMIT ?",
        (like, like, like, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]