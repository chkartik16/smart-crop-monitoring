"""
database.py
============
Lightweight SQLite persistence layer for the Smart Crop Monitoring
System. Replaces the previous users.json file with a real embedded
database that survives redeploys (as long as the underlying volume
persists) and supports:

- User accounts (signup / login) with salted password hashing
- Per-user Telegram chat_id binding (no more session-only state)
- A full prediction history log per user, used by the Analytics tab

No external dependencies — uses Python's built-in sqlite3 and hashlib.
"""

import sqlite3
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "smart_crop.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not already exist. Safe to call on every app start."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            telegram_chat_id TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            crop_type TEXT,
            growth_stage TEXT,
            soil_moisture REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            prediction INTEGER,
            probability REAL,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# PASSWORD HASHING (PBKDF2 via hashlib — no external deps)
# ============================================================

def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()


def create_user(username, password):
    """Returns True on success, False if the username already exists."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        conn.close()
        return False

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    cur.execute(
        "INSERT INTO users (username, password_hash, salt, telegram_chat_id, created_at) "
        "VALUES (?, ?, ?, NULL, ?)",
        (username, password_hash, salt, datetime.now(timezone.utc).isoformat())
    )

    conn.commit()
    conn.close()
    return True


def verify_user(username, password):
    """Returns True if username/password match a stored account."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT password_hash, salt FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    expected_hash = _hash_password(password, row["salt"])
    return hmac.compare_digest(expected_hash, row["password_hash"])


# ============================================================
# TELEGRAM CHAT ID (persisted per user, not just per session)
# ============================================================

def set_telegram_chat_id(username, chat_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET telegram_chat_id = ? WHERE username = ?",
        (str(chat_id), username)
    )
    conn.commit()
    conn.close()


def get_telegram_chat_id(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT telegram_chat_id FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()
    return row["telegram_chat_id"] if row else None


# ============================================================
# PREDICTION HISTORY
# ============================================================

def log_prediction(username, crop_type, growth_stage, soil_moisture,
                    temperature, humidity, rainfall, prediction, probability):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions
           (username, timestamp, crop_type, growth_stage, soil_moisture,
            temperature, humidity, rainfall, prediction, probability)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            username,
            datetime.now(timezone.utc).isoformat(),
            crop_type,
            growth_stage,
            soil_moisture,
            temperature,
            humidity,
            rainfall,
            int(prediction),
            float(probability),
        )
    )
    conn.commit()
    conn.close()


def get_user_history(username, limit=200):
    """Returns a list of sqlite3.Row objects, most recent first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM predictions
           WHERE username = ?
           ORDER BY timestamp DESC
           LIMIT ?""",
        (username, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows
