from lib.db import get_conn


def save_alert_draft(restaurant_id, audience, draft):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO alerts (restaurant_id, audience, draft) VALUES (?,?,?)",
        (restaurant_id, audience, draft),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return {"alert_id": alert_id, "status": "pending"}


def list_alerts(status=None):
    conn = get_conn()
    if status:
        rows = conn.execute("SELECT * FROM alerts WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_alert_status(alert_id, status):
    conn = get_conn()
    conn.execute("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))
    conn.commit()
    conn.close()