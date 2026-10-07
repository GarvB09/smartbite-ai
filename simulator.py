import argparse
import random
from lib.db import get_conn

BAD_REVIEWS = [
    "Food was cold and the service was very slow.",
    "Found a hair in my dish. Never coming back.",
    "Quality has dropped badly, used to be so much better.",
    "Rude staff and overpriced for what you get.",
    "Stale food, got a stomach ache afterwards.",
    "Waited an hour and the order was wrong.",
    "Very disappointing, dirty tables and bland food.",
]


def bad_reviews(rid, n):
    conn = get_conn()
    for _ in range(n):
        conn.execute(
            "INSERT INTO reviews (restaurant_id, user_id, stars, text) VALUES (?,?,?,?)",
            (rid, random.randint(1, 20), random.choice([1.0, 1.5, 2.0]), random.choice(BAD_REVIEWS)),
        )
    conn.commit()
    conn.close()
    print(f"Injected {n} bad reviews for restaurant {rid}")


def set_rating(rid, value):
    conn = get_conn()
    conn.execute("UPDATE restaurants SET rating=? WHERE id=?", (value, rid))
    conn.commit()
    conn.close()
    print(f"Rating of restaurant {rid} set to {value}")


def change_price(rid, pct):
    conn = get_conn()
    row = conn.execute(
        "SELECT item, price FROM menu_prices WHERE restaurant_id=? ORDER BY id DESC LIMIT 1",
        (rid,),
    ).fetchone()
    if not row:
        print("No price found")
        return
    new = round(row["price"] * (1 + pct / 100), 2)
    conn.execute(
        "INSERT INTO menu_prices (restaurant_id, item, price) VALUES (?,?,?)",
        (rid, row["item"], new),
    )
    conn.commit()
    conn.close()
    print(f"Price for restaurant {rid}: {row['price']} -> {new}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["bad_reviews", "rating", "price"])
    p.add_argument("restaurant_id", type=int)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--to", type=float, default=3.0)
    p.add_argument("--pct", type=float, default=25)
    a = p.parse_args()
    if a.action == "bad_reviews":
        bad_reviews(a.restaurant_id, a.n)
    elif a.action == "rating":
        set_rating(a.restaurant_id, a.to)
    else:
        change_price(a.restaurant_id, a.pct)