from lib.db import get_conn


def set_penalty(restaurant_id, value, reason):
    conn = get_conn()
    conn.execute(
        "INSERT INTO penalties (restaurant_id, value, reason) VALUES (?,?,?) "
        "ON CONFLICT(restaurant_id) DO UPDATE SET value=excluded.value, "
        "reason=excluded.reason, updated_at=CURRENT_TIMESTAMP",
        (restaurant_id, value, reason),
    )
    conn.commit()
    conn.close()
    return {"restaurant_id": restaurant_id, "value": value, "reason": reason}


def get_penalty(restaurant_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM penalties WHERE restaurant_id=?", (restaurant_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def clear_penalty(restaurant_id):
    conn = get_conn()
    conn.execute("DELETE FROM penalties WHERE restaurant_id=?", (restaurant_id,))
    conn.commit()
    conn.close()