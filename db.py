import os
import sqlite3
import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "logs.db")

def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        stage TEXT,
        status TEXT,
        message TEXT
    )""")
    conn.commit()
    return conn

def insert_log(stage: str, status: str, message: str):
    ts = datetime.datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute("INSERT INTO logs (timestamp, stage, status, message) VALUES (?, ?, ?, ?)",
                 (ts, stage, status, message))
    conn.commit()
    conn.close()

def get_last_logs(limit: int = 100):
    conn = _get_conn()
    cur = conn.execute("SELECT id, timestamp, stage, status, message FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows