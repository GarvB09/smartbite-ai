import ast
import random
import pandas as pd
from lib.db import get_conn, init_db

random.seed(42)
N_RESTAURANTS = 300
N_USERS = 20
MAX_REVIEWS = 30


def parse_rate(x):
    try:
        return float(str(x).split("/")[0])
    except ValueError:
        return None


def parse_cost(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def parse_reviews(raw):
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    out = []
    for rating, text in items:
        try:
            stars = float(rating.replace("Rated", "").strip())
        except (ValueError, AttributeError):
            continue
        if not isinstance(text, str):
            continue
        text = text.replace("RATED", "").replace("\n", " ").strip()
        if text:
            out.append((stars, text))
    return out[:MAX_REVIEWS]


def load():
    init_db()
    df = pd.read_csv("data/zomato.csv")
    df["rating"] = df["rate"].apply(parse_rate)
    df = df.dropna(subset=["rating"]).drop_duplicates(subset=["name", "address"])
    df = df.sort_values("votes", ascending=False).head(N_RESTAURANTS)

    conn = get_conn()
    for t in ["restaurants", "reviews", "menu_prices", "users", "penalties", "alerts"]:
        conn.execute(f"DELETE FROM {t}")

    cuisines_pool = set()
    for rid, (_, row) in enumerate(df.iterrows(), start=1):
        cuisine = str(row["cuisines"]).split(",")[0].strip()
        cuisines_pool.add(cuisine)
        conn.execute(
            "INSERT INTO restaurants VALUES (?,?,?,?,?)",
            (rid, row["name"], cuisine, row["location"], row["rating"]),
        )
        cost = parse_cost(row["approx_cost(for two people)"])
        if cost:
            conn.execute(
                "INSERT INTO menu_prices (restaurant_id, item, price, recorded_at) "
                "VALUES (?,?,?, datetime('now','-30 days'))",
                (rid, "Meal for two", cost),
            )
        for stars, text in parse_reviews(row["reviews_list"]):
            days = random.randint(0, 60)
            conn.execute(
                "INSERT INTO reviews (restaurant_id, user_id, stars, text, created_at) "
                "VALUES (?,?,?,?, datetime('now', ?))",
                (rid, random.randint(1, N_USERS), stars, text, f"-{days} days"),
            )

    pool = sorted(cuisines_pool)
    for uid in range(1, N_USERS + 1):
        favs = ",".join(random.sample(pool, 2))
        conn.execute("INSERT INTO users VALUES (?,?,?)", (uid, f"user{uid}", favs))

    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} restaurants and {N_USERS} users")


if __name__ == "__main__":
    load()