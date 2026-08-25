"""
Database connection and schema initialization using SQLite.
"""
import os
import sqlite3
import config

def get_db_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Incidents Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_code TEXT UNIQUE NOT NULL,
        worker_id INTEGER NOT NULL,
        worker_label TEXT NOT NULL,
        violation_type TEXT NOT NULL,
        description TEXT,
        severity TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL,
        duration REAL DEFAULT 0.0,
        source TEXT NOT NULL,
        snapshot_path TEXT,
        status TEXT DEFAULT 'CONFIRMED'
    )
    """)

    # 2. Restricted & Hazard Zones Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        coordinates TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # 3. System Settings Key-Value Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    # 4. Monitoring Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        source_type TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        total_workers INTEGER DEFAULT 0,
        total_violations INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized SQLite database at: {config.DB_PATH}")

if __name__ == "__main__":
    init_db()
