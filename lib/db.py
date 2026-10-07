import sqlite3

DB_PATH = "data/smartbite.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS restaurants (
        id INTEGER PRIMARY KEY,
        name TEXT,
        cuisine TEXT,
        city TEXT,
        rating REAL
    );
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        restaurant_id INTEGER,
        user_id INTEGER,
        stars REAL,
        text TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS menu_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        restaurant_id INTEGER,
        item TEXT,
        price REAL,
        recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        favorite_cuisines TEXT
    );
    CREATE TABLE IF NOT EXISTS penalties (
        restaurant_id INTEGER PRIMARY KEY,
        value REAL,
        reason TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        restaurant_id INTEGER,
        audience TEXT,
        draft TEXT,
        status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("DB ready")