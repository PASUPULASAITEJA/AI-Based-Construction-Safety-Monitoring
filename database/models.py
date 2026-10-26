"""
Database models and queries for Incidents, Zones, and Analytics.
"""
import json
import datetime
from database.db import get_db_connection

class IncidentModel:
    @staticmethod
    def create(incident_code, worker_id, worker_label, violation_type, description, severity, confidence, source, snapshot_path=None, duration=0.0):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT OR REPLACE INTO incidents 
            (incident_code, worker_id, worker_label, violation_type, description, severity, confidence, timestamp, duration, source, snapshot_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CONFIRMED')
        """, (incident_code, worker_id, worker_label, violation_type, description, severity, confidence, now_str, duration, source, snapshot_path))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all(limit=100, offset=0, violation_type=None, severity=None, search=None):
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM incidents WHERE 1=1"
        params = []

        if violation_type and violation_type != "ALL":
            query += " AND violation_type = ?"
            params.append(violation_type)

        if severity and severity != "ALL":
            query += " AND severity = ?"
            params.append(severity)

        if search:
            query += " AND (worker_label LIKE ? OR incident_code LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_recent(limit=10):
        return IncidentModel.get_all(limit=limit)

    @staticmethod
    def get_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM incidents")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0

    @staticmethod
    def get_analytics_summary():
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total counts
        cursor.execute("SELECT COUNT(*) as total FROM incidents")
        total_incidents = cursor.fetchone()["total"]

        # Severity breakdown
        cursor.execute("SELECT severity, COUNT(*) as count FROM incidents GROUP BY severity")
        severity_dist = {r["severity"]: r["count"] for r in cursor.fetchall()}

        # Violation type breakdown
        cursor.execute("SELECT violation_type, COUNT(*) as count FROM incidents GROUP BY violation_type")
        type_dist = {r["violation_type"]: r["count"] for r in cursor.fetchall()}

        # Violations by source
        cursor.execute("SELECT source, COUNT(*) as count FROM incidents GROUP BY source")
        source_dist = {r["source"]: r["count"] for r in cursor.fetchall()}

        # Recent 7 days timeline
        cursor.execute("""
            SELECT substr(timestamp, 1, 10) as day, COUNT(*) as count 
            FROM incidents 
            GROUP BY day 
            ORDER BY day DESC 
            LIMIT 7
        """)
        timeline_rows = cursor.fetchall()
        timeline = {r["day"]: r["count"] for r in reversed(timeline_rows)}

        conn.close()

        return {
            "total_incidents": total_incidents,
            "severity_distribution": severity_dist,
            "type_distribution": type_dist,
            "source_distribution": source_dist,
            "timeline": timeline
        }

class ZoneModel:
    @staticmethod
    def create(name, zone_type, coordinates):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coords_json = json.dumps(coordinates)

        cursor.execute("""
            INSERT INTO zones (name, zone_type, coordinates, created_at)
            VALUES (?, ?, ?, ?)
        """, (name, zone_type.upper(), coords_json, now_str))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM zones ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()

        zones = []
        for r in rows:
            z_dict = dict(r)
            try:
                z_dict["coordinates"] = json.loads(z_dict["coordinates"])
            except Exception:
                z_dict["coordinates"] = []
            zones.append(z_dict)
        return zones

    @staticmethod
    def delete(zone_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

class SettingsModel:
    @staticmethod
    def get(key, default_val=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row["value"] if row else default_val

    @staticmethod
    def set(key, value):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, str(value)))
        conn.commit()
        conn.close()

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        conn.close()
        return {r["key"]: r["value"] for r in rows}
