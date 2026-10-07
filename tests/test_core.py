import pytest
from lib.db import get_conn, init_db
from tools import recommender
from tools.penalties import set_penalty
from tools.prices import get_price_drift
from tools.reviews import get_rating_trend


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.db.DB_PATH", str(tmp_path / "t.db"))
    init_db()
    conn = get_conn()
    conn.executemany(
        "INSERT INTO restaurants VALUES (?,?,?,?,?)",
        [(1, "A Cafe", "Cafe", "X", 4.5), (2, "B Grill", "Grill", "X", 4.0)],
    )
    conn.execute("INSERT INTO users VALUES (1,'u','Cafe,Grill')")
    conn.executemany(
        "INSERT INTO menu_prices (restaurant_id, item, price) VALUES (?,?,?)",
        [(1, "m", 500), (2, "m", 500)],
    )
    conn.commit()
    conn.close()
    recommender._state.clear()
    yield
    recommender._state.clear()


def test_penalty_changes_ranking(db):
    assert recommender.recommend(1)[0]["restaurant_id"] == 1
    set_penalty(1, 0.8, "bad reviews")
    assert recommender.recommend(1)[0]["restaurant_id"] == 2


def test_unknown_user_raises(db):
    with pytest.raises(ValueError):
        recommender.recommend(99)


def test_rating_trend_detects_drop(db):
    conn = get_conn()
    for _ in range(10):
        conn.execute(
            "INSERT INTO reviews (restaurant_id, user_id, stars, text, created_at) "
            "VALUES (1,1,4.5,'good', datetime('now','-30 days'))"
        )
    for _ in range(10):
        conn.execute("INSERT INTO reviews (restaurant_id, user_id, stars, text) VALUES (1,1,1.5,'bad')")
    conn.commit()
    conn.close()
    t = get_rating_trend(1)
    assert t["recent_avg"] == 1.5
    assert t["delta"] == -3.0


def test_price_drift(db):
    conn = get_conn()
    conn.execute("INSERT INTO menu_prices (restaurant_id, item, price) VALUES (1,'m',625)")
    conn.commit()
    conn.close()
    assert get_price_drift(1)["drift_pct"] == 25.0