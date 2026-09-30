"""
Database connection, schema initialization, and safe migrations using SQLite.
"""
import os
import secrets
import sqlite3
import datetime
from werkzeug.security import generate_password_hash
import config

def get_db_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    # Autocommit: every statement commits on its own, so a statement that fails (e.g. a UNIQUE or
    # FOREIGN KEY violation) can never leave a write transaction open and lock the database.
    conn = sqlite3.connect(config.DB_PATH, timeout=20.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users & RBAC Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'VIEWER',
        email TEXT,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    """)

    # 2. Sites Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        location TEXT,
        project_name TEXT,
        status TEXT DEFAULT 'ACTIVE',
        safety_officer TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """)

    # 3. Cameras Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        site_id INTEGER,
        zone_id INTEGER,
        stream_source TEXT,
        stream_type TEXT DEFAULT 'webcam',
        status TEXT DEFAULT 'OFFLINE',
        resolution TEXT DEFAULT '640x480',
        fps REAL DEFAULT 0.0,
        last_seen TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(site_id) REFERENCES sites(id) ON DELETE SET NULL
    )
    """)

    # 4. Restricted & Hazard Zones Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site_id INTEGER,
        name TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        coordinates TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(site_id) REFERENCES sites(id) ON DELETE SET NULL
    )
    """)

    # 5. Violations Table (Separated from Incidents)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        violation_code TEXT UNIQUE NOT NULL,
        worker_id INTEGER NOT NULL,
        worker_label TEXT NOT NULL,
        violation_type TEXT NOT NULL,
        description TEXT,
        severity TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL,
        duration_sec REAL DEFAULT 0.0,
        source TEXT NOT NULL,
        site_id INTEGER,
        zone_id INTEGER,
        camera_id INTEGER,
        snapshot_path TEXT,
        status TEXT DEFAULT 'OPEN',
        review_decision TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        review_notes TEXT,
        incident_id INTEGER,
        FOREIGN KEY(site_id) REFERENCES sites(id) ON DELETE SET NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE SET NULL,
        FOREIGN KEY(zone_id) REFERENCES zones(id) ON DELETE SET NULL
    )
    """)

    # 6. Incidents Table (Reviewed Safety Cases)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_code TEXT UNIQUE NOT NULL,
        violation_id INTEGER,
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
        status TEXT DEFAULT 'OPEN',
        site_id INTEGER,
        camera_id INTEGER,
        zone_id INTEGER,
        assigned_to TEXT,
        root_cause TEXT,
        corrective_action TEXT,
        resolution_note TEXT,
        created_at TEXT,
        acknowledged_at TEXT,
        resolved_at TEXT,
        closed_by TEXT,
        FOREIGN KEY(violation_id) REFERENCES violations(id) ON DELETE SET NULL,
        FOREIGN KEY(site_id) REFERENCES sites(id) ON DELETE SET NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE SET NULL
    )
    """)

    # Check and add any missing columns in incidents table (for seamless migration)
    cursor.execute("PRAGMA table_info(incidents)")
    existing_incident_cols = {row["name"] for row in cursor.fetchall()}
    migration_cols = [
        ("violation_id", "INTEGER"),
        ("site_id", "INTEGER"),
        ("camera_id", "INTEGER"),
        ("zone_id", "INTEGER"),
        ("assigned_to", "TEXT"),
        ("root_cause", "TEXT"),
        ("corrective_action", "TEXT"),
        ("resolution_note", "TEXT"),
        ("created_at", "TEXT"),
        ("acknowledged_at", "TEXT"),
        ("resolved_at", "TEXT"),
        ("closed_by", "TEXT")
    ]
    for col_name, col_type in migration_cols:
        if col_name not in existing_incident_cols:
            cursor.execute(f"ALTER TABLE incidents ADD COLUMN {col_name} {col_type}")

    # Check violations table columns
    cursor.execute("PRAGMA table_info(violations)")
    existing_viol_cols = {row["name"] for row in cursor.fetchall()}
    viol_migration_cols = [
        ("site_id", "INTEGER"),
        ("zone_id", "INTEGER"),
        ("camera_id", "INTEGER"),
        ("duration_sec", "REAL DEFAULT 0.0"),
        ("status", "TEXT DEFAULT 'OPEN'"),
        ("review_decision", "TEXT"),
        ("reviewed_by", "TEXT"),
        ("reviewed_at", "TEXT"),
        ("review_notes", "TEXT"),
        ("incident_id", "INTEGER")
    ]
    for col_name, col_type in viol_migration_cols:
        if col_name not in existing_viol_cols:
            cursor.execute(f"ALTER TABLE violations ADD COLUMN {col_name} {col_type}")

    # Check alerts table columns
    cursor.execute("PRAGMA table_info(alerts)")
    existing_alert_cols = {row["name"] for row in cursor.fetchall()}
    alert_migration_cols = [
        ("violation_id", "INTEGER"),
        ("site_id", "INTEGER"),
        ("camera_id", "INTEGER"),
        ("assigned_to", "TEXT"),
        ("acknowledged_at", "TEXT"),
        ("resolved_at", "TEXT"),
        ("resolution_note", "TEXT")
    ]
    # Only migrate an existing (legacy) table; a fresh database gets the full schema below
    for col_name, col_type in alert_migration_cols:
        if existing_alert_cols and col_name not in existing_alert_cols:
            cursor.execute(f"ALTER TABLE alerts ADD COLUMN {col_name} {col_type}")

    # Also check zones table for site_id
    cursor.execute("PRAGMA table_info(zones)")
    existing_zone_cols = {row["name"] for row in cursor.fetchall()}
    if "site_id" not in existing_zone_cols:
        cursor.execute("ALTER TABLE zones ADD COLUMN site_id INTEGER")

    # 7. Alerts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_code TEXT UNIQUE NOT NULL,
        incident_id INTEGER,
        violation_id INTEGER,
        violation_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        status TEXT DEFAULT 'OPEN',
        camera_id INTEGER,
        site_id INTEGER,
        assigned_to TEXT,
        timestamp TEXT NOT NULL,
        acknowledged_at TEXT,
        resolved_at TEXT,
        resolution_note TEXT,
        FOREIGN KEY(incident_id) REFERENCES incidents(id) ON DELETE CASCADE,
        FOREIGN KEY(violation_id) REFERENCES violations(id) ON DELETE CASCADE,
        FOREIGN KEY(site_id) REFERENCES sites(id) ON DELETE SET NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE SET NULL
    )
    """)

    # 8. Worker Tracking Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS workers_track (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_code TEXT UNIQUE NOT NULL,
        label TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        total_observations INTEGER DEFAULT 0,
        compliant_observations INTEGER DEFAULT 0,
        violation_count INTEGER DEFAULT 0
    )
    """)

    # 9. In-App Notifications Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        severity TEXT DEFAULT 'INFO',
        is_read INTEGER DEFAULT 0,
        incident_id INTEGER,
        violation_id INTEGER,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(incident_id) REFERENCES incidents(id) ON DELETE SET NULL
    )
    """)

    # 10. Audit Logs Table
    cursor.execute("PRAGMA table_info(audit_logs)")
    existing_audit_cols = {row["name"] for row in cursor.fetchall()}
    if "ip_address" not in existing_audit_cols and existing_audit_cols:
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN ip_address TEXT")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_role TEXT NOT NULL,
        action TEXT NOT NULL,
        category TEXT NOT NULL,
        details TEXT,
        target_id TEXT,
        ip_address TEXT
    )
    """)

    # 11. System Settings Key-Value Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    # 12. Monitoring Sessions Table
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

    # Create Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)")

    # Seed Default Settings if missing
    default_settings = [
        ("conf_thresh", str(config.CONFIDENCE_THRESHOLD)),
        ("iou_thresh", str(config.IOU_THRESHOLD)),
        ("min_violation_frames", str(config.MIN_VIOLATION_FRAMES)),
        ("min_zone_frames", str(config.MIN_ZONE_FRAMES))
    ]
    for k, v in default_settings:
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))

    # Seed an initial admin account if there are no users.
    # Password comes from SITEGUARD_ADMIN_PASSWORD, or a random one is generated and printed once.
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("SELECT COUNT(*) as count FROM users")
    if cursor.fetchone()["count"] == 0:
        admin_password = os.environ.get("SITEGUARD_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, role, email, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("admin", generate_password_hash(admin_password), "Site Administrator", "ADMIN", "", now_str))
        if os.environ.get("SITEGUARD_ADMIN_PASSWORD"):
            print("[DB] Created initial user 'admin' with the password from SITEGUARD_ADMIN_PASSWORD")
        else:
            print(f"[DB] Created initial user 'admin' with generated password: {admin_password}")
            print("[DB] Store it now; add further users with: python manage_users.py add <username> <role>")

    # Migrate legacy incidents (created before violations existed) into violations table
    cursor.execute("""
        INSERT OR IGNORE INTO violations 
        (violation_code, worker_id, worker_label, violation_type, description, severity, confidence, timestamp, source, snapshot_path, status, incident_id)
        SELECT 'VIO-' || substr(incident_code, 5), worker_id, worker_label, violation_type, description, severity, confidence, timestamp, source, snapshot_path, 
               CASE WHEN status = 'RESOLVED' THEN 'RESOLVED' ELSE 'OPEN' END, id
        FROM incidents
        WHERE violation_id IS NULL
          AND 'VIO-' || substr(incident_code, 5) NOT IN (SELECT violation_code FROM violations)
    """)
    cursor.execute("""
        UPDATE incidents
        SET violation_id = (SELECT v.id FROM violations v WHERE v.violation_code = 'VIO-' || substr(incidents.incident_code, 5))
        WHERE violation_id IS NULL
    """)

    # Sync any violations/incidents into alerts table
    cursor.execute("""
        INSERT OR IGNORE INTO alerts (alert_code, incident_id, violation_type, severity, status, timestamp)
        SELECT 'ALT-' || substr(incident_code, 5), id, violation_type, severity, 
               CASE WHEN status = 'RESOLVED' THEN 'RESOLVED' ELSE 'OPEN' END, 
               timestamp
        FROM incidents
        WHERE id NOT IN (SELECT incident_id FROM alerts WHERE incident_id IS NOT NULL)
          AND violation_id NOT IN (SELECT violation_id FROM alerts WHERE violation_id IS NOT NULL)
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized SiteGuard AI database schema at: {config.DB_PATH}")

if __name__ == "__main__":
    init_db()
