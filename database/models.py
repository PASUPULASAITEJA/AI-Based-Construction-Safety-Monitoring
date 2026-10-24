"""
Database models and queries for Users, Sites, Cameras, Zones, Violations, Incidents, Alerts, Worker Tracking, Notifications, and Audit Logs.
"""
import json
import datetime
import threading
from werkzeug.security import check_password_hash, generate_password_hash
from database.db import get_db_connection

# Serializes allocation of the shared INC-/VIO-/ALT- sequence numbers within this process
RECORD_CODE_LOCK = threading.Lock()

PPE_CATEGORY_IDS = ["HEAD", "EYE", "HEARING", "VISIBILITY", "HAND", "FALL_PROTECTION", "LEG_PROTECTION", "FOOT"]

def next_record_number(cursor):
    """Next free number of the sequence shared by incident, violation and alert codes (INC-0007, VIO-0007, ...)."""
    cursor.execute("""
        SELECT MAX(n) as max_n FROM (
            SELECT MAX(CAST(substr(incident_code, 5) AS INTEGER)) as n FROM incidents
            UNION ALL SELECT MAX(CAST(substr(violation_code, 5) AS INTEGER)) FROM violations
            UNION ALL SELECT MAX(CAST(substr(alert_code, 5) AS INTEGER)) FROM alerts
        )
    """)
    row = cursor.fetchone()
    return (row["max_n"] or 0) + 1

class UserModel:
    @staticmethod
    def authenticate(username, password):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row["password_hash"], password):
            UserModel.update_last_login(row["id"])
            return dict(row)
        return None

    @staticmethod
    def get_by_id(user_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def update_last_login(user_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_str, user_id))
        conn.commit()
        conn.close()

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, role, email, created_at, last_login FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def create(username, password, full_name, role="VIEWER", email=""):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        phash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, role, email, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username.strip(), phash, full_name.strip(), role.strip(), email.strip(), now_str))
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id


class SiteModel:
    @staticmethod
    def create(site_code, name, location="", project_name="", safety_officer=""):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO sites (site_code, name, location, project_name, status, safety_officer, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
        """, (site_code.strip(), name.strip(), location.strip(), project_name.strip(), safety_officer.strip(), now_str, now_str))
        conn.commit()
        site_id = cursor.lastrowid
        conn.close()
        return site_id

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, 
                   (SELECT COUNT(*) FROM cameras c WHERE c.site_id = s.id) as camera_count,
                   (SELECT COUNT(*) FROM zones z WHERE z.site_id = s.id) as zone_count,
                   (SELECT COUNT(*) FROM violations v WHERE v.site_id = s.id) as violation_count,
                   (SELECT COUNT(*) FROM incidents i WHERE i.site_id = s.id) as incident_count
            FROM sites s
            ORDER BY s.id ASC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_by_id(site_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sites WHERE id = ?", (site_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def update(site_id, name, location, project_name, status, safety_officer):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE sites 
            SET name = ?, location = ?, project_name = ?, status = ?, safety_officer = ?, updated_at = ?
            WHERE id = ?
        """, (name, location, project_name, status, safety_officer, now_str, site_id))
        conn.commit()
        conn.close()

    @staticmethod
    def delete(site_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sites WHERE id = ?", (site_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    @staticmethod
    def get_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM sites")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0


class CameraModel:
    @staticmethod
    def create(camera_code, name, site_id=None, zone_id=None, stream_source="0", stream_type="webcam", resolution="640x480", fps=30.0):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO cameras (camera_code, name, site_id, zone_id, stream_source, stream_type, status, resolution, fps, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'OFFLINE', ?, ?, ?, ?)
        """, (camera_code.strip(), name.strip(), site_id, zone_id, str(stream_source), stream_type, resolution, fps, None, now_str))
        conn.commit()
        cam_id = cursor.lastrowid
        conn.close()
        return cam_id

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, s.name as site_name, z.name as zone_name,
                   (SELECT COUNT(*) FROM violations v WHERE v.camera_id = c.id) as violation_count,
                   (SELECT COUNT(*) FROM incidents i WHERE i.camera_id = c.id) as incident_count
            FROM cameras c
            LEFT JOIN sites s ON c.site_id = s.id
            LEFT JOIN zones z ON c.zone_id = z.id
            ORDER BY c.id ASC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_by_id(camera_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def code_exists(camera_code):
        conn = get_db_connection()
        row = conn.execute("SELECT 1 FROM cameras WHERE camera_code = ?", (camera_code,)).fetchone()
        conn.close()
        return row is not None

    @staticmethod
    def update_status(camera_id, status, fps=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if fps is not None:
            cursor.execute("UPDATE cameras SET status = ?, fps = ?, last_seen = ? WHERE id = ?", (status, fps, now_str, camera_id))
        else:
            cursor.execute("UPDATE cameras SET status = ?, last_seen = ? WHERE id = ?", (status, now_str, camera_id))
        conn.commit()
        conn.close()

    @staticmethod
    def delete(camera_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    @staticmethod
    def get_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM cameras")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0

    @staticmethod
    def get_active_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM cameras WHERE status = 'ONLINE'")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0


class ZoneModel:
    @staticmethod
    def create(name, zone_type, coordinates, site_id=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coords_json = json.dumps(coordinates)

        cursor.execute("""
            INSERT INTO zones (name, zone_type, coordinates, site_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (name, zone_type.upper(), coords_json, site_id, now_str))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all(site_id=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        if site_id:
            cursor.execute("SELECT z.*, s.name as site_name FROM zones z LEFT JOIN sites s ON z.site_id = s.id WHERE z.site_id = ? ORDER BY z.id ASC", (site_id,))
        else:
            cursor.execute("SELECT z.*, s.name as site_name FROM zones z LEFT JOIN sites s ON z.site_id = s.id ORDER BY z.id ASC")
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


class ViolationModel:
    @staticmethod
    def create(violation_code, worker_id, worker_label, violation_type, description, severity, confidence, source, snapshot_path=None, duration_sec=0.0, site_id=None, zone_id=None, camera_id=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO violations
            (violation_code, worker_id, worker_label, violation_type, description, severity, confidence, timestamp, duration_sec, source, snapshot_path, status, site_id, zone_id, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
        """, (violation_code, worker_id, worker_label, violation_type, description, severity, confidence, now_str, duration_sec, source, snapshot_path, site_id, zone_id, camera_id))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all(limit=100, offset=0, violation_type=None, severity=None, status=None, site_id=None, camera_id=None, search=None, start_date=None, end_date=None):
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT v.*, s.name as site_name, c.name as camera_name, z.name as zone_name 
            FROM violations v
            LEFT JOIN sites s ON v.site_id = s.id
            LEFT JOIN cameras c ON v.camera_id = c.id
            LEFT JOIN zones z ON v.zone_id = z.id
            WHERE 1=1
        """
        params = []

        if violation_type and violation_type != "ALL":
            query += " AND v.violation_type = ?"
            params.append(violation_type)

        if severity and severity != "ALL":
            query += " AND v.severity = ?"
            params.append(severity)

        if status and status != "ALL":
            query += " AND v.status = ?"
            params.append(status)

        if site_id and site_id != "ALL":
            query += " AND v.site_id = ?"
            params.append(site_id)

        if camera_id and camera_id != "ALL":
            query += " AND v.camera_id = ?"
            params.append(camera_id)

        if start_date:
            query += " AND substr(v.timestamp, 1, 10) >= ?"
            params.append(start_date)

        if end_date:
            query += " AND substr(v.timestamp, 1, 10) <= ?"
            params.append(end_date)

        if search:
            query += " AND (v.worker_label LIKE ? OR v.violation_code LIKE ? OR v.description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        query += " ORDER BY v.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_by_id(violation_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT v.*, s.name as site_name, c.name as camera_name, z.name as zone_name 
            FROM violations v
            LEFT JOIN sites s ON v.site_id = s.id
            LEFT JOIN cameras c ON v.camera_id = c.id
            LEFT JOIN zones z ON v.zone_id = z.id
            WHERE v.id = ? OR v.violation_code = ?
        """, (violation_id, violation_id))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def confirm_violation(violation_id, reviewer, notes=""):
        """Human-in-the-loop: confirms violation and creates corresponding formal Incident."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("SELECT * FROM violations WHERE id = ? OR violation_code = ?", (violation_id, violation_id))
        v = cursor.fetchone()
        if not v:
            conn.close()
            return None

        v_dict = dict(v)

        # The AI pipeline already opened an incident for this violation: confirm that one instead of duplicating it
        if v_dict.get("incident_id"):
            existing = cursor.execute("SELECT id, incident_code FROM incidents WHERE id = ?", (v_dict["incident_id"],)).fetchone()
            if existing:
                cursor.execute("""
                    UPDATE incidents SET assigned_to = COALESCE(assigned_to, ?), acknowledged_at = COALESCE(acknowledged_at, ?)
                    WHERE id = ?
                """, (reviewer, now_str, existing["id"]))
                cursor.execute("""
                    UPDATE violations
                    SET status = 'CONFIRMED', review_decision = 'CONFIRMED', reviewed_by = ?, reviewed_at = ?, review_notes = ?
                    WHERE id = ?
                """, (reviewer, now_str, notes, v_dict["id"]))
                conn.commit()
                conn.close()
                return existing["incident_code"]

        with RECORD_CODE_LOCK:
            inc_code = f"INC-{next_record_number(cursor):04d}"
            # Create Incident record
            cursor.execute("""
            INSERT INTO incidents 
            (incident_code, violation_id, worker_id, worker_label, violation_type, description, severity, confidence, timestamp, duration, source, snapshot_path, status, site_id, camera_id, zone_id, assigned_to, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
        """, (inc_code, v_dict["id"], v_dict["worker_id"], v_dict["worker_label"], v_dict["violation_type"], v_dict["description"], v_dict["severity"], v_dict["confidence"], v_dict["timestamp"], v_dict["duration_sec"], v_dict["source"], v_dict["snapshot_path"], v_dict["site_id"], v_dict["camera_id"], v_dict["zone_id"], reviewer, now_str))
            inc_id = cursor.lastrowid

            # Update Violation
            cursor.execute("""
                UPDATE violations
                SET status = 'CONFIRMED', review_decision = 'CONFIRMED', reviewed_by = ?, reviewed_at = ?, review_notes = ?, incident_id = ?
                WHERE id = ?
            """, (reviewer, now_str, notes, inc_id, v_dict["id"]))

            # Commit before releasing the lock so the next allocation sees this code
            conn.commit()
        conn.close()
        return inc_code

    @staticmethod
    def mark_false_positive(violation_id, reviewer, reason=""):
        """Human-in-the-loop: flags detection as False Positive to tune CV and analytics."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            UPDATE violations 
            SET status = 'FALSE_POSITIVE', review_decision = 'FALSE_POSITIVE', reviewed_by = ?, reviewed_at = ?, review_notes = ?
            WHERE id = ? OR violation_code = ?
        """, (reviewer, now_str, reason, violation_id, violation_id))
        found = cursor.rowcount > 0

        # Close the incident the pipeline opened for it
        cursor.execute("""
            UPDATE incidents
            SET status = 'CLOSED', closed_by = ?, resolved_at = ?,
                resolution_note = COALESCE(resolution_note, ?)
            WHERE violation_id IN (SELECT id FROM violations WHERE id = ? OR violation_code = ?)
              AND status NOT IN ('RESOLVED', 'CLOSED')
        """, (reviewer, now_str, f"False positive: {reason}", violation_id, violation_id))

        # Close any alert
        cursor.execute("""
            UPDATE alerts 
            SET status = 'RESOLVED', resolution_note = ?
            WHERE violation_id = ? OR alert_code = ?
        """, (f"Marked False Positive by {reviewer}: {reason}", violation_id, violation_id))

        conn.commit()
        conn.close()
        return found

    @staticmethod
    def get_distinct_types():
        conn = get_db_connection()
        rows = conn.execute("SELECT DISTINCT violation_type FROM violations ORDER BY violation_type").fetchall()
        conn.close()
        return [r["violation_type"] for r in rows]

    @staticmethod
    def get_stats():
        """Computes human review and false positive metrics."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM violations")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as confirmed FROM violations WHERE review_decision = 'CONFIRMED'")
        confirmed = cursor.fetchone()["confirmed"]

        cursor.execute("SELECT COUNT(*) as false_pos FROM violations WHERE review_decision = 'FALSE_POSITIVE'")
        false_pos = cursor.fetchone()["false_pos"]

        reviewed = confirmed + false_pos
        pending = total - reviewed

        fp_rate = round((false_pos / reviewed) * 100, 1) if reviewed > 0 else 0.0
        conf_rate = round((confirmed / reviewed) * 100, 1) if reviewed > 0 else 0.0

        conn.close()
        return {
            "total_violations": total,
            "confirmed_violations": confirmed,
            "false_positives": false_pos,
            "pending_reviews": pending,
            "false_positive_rate": fp_rate,
            "confirmation_rate": conf_rate,
            "review_rate": round((reviewed / max(1, total)) * 100, 1) if total > 0 else 0.0
        }

    @staticmethod
    def get_count(today_only=False):
        conn = get_db_connection()
        cursor = conn.cursor()
        if today_only:
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT COUNT(*) as total FROM violations WHERE substr(timestamp, 1, 10) = ?", (today_str,))
        else:
            cursor.execute("SELECT COUNT(*) as total FROM violations")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0


class IncidentModel:
    @staticmethod
    def create(incident_code, worker_id, worker_label, violation_type, description, severity, confidence, source, snapshot_path=None, duration=0.0, site_id=None, camera_id=None, zone_id=None, assigned_to=None, violation_id=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO incidents
            (incident_code, violation_id, worker_id, worker_label, violation_type, description, severity, confidence, timestamp, duration, source, snapshot_path, status, site_id, camera_id, zone_id, assigned_to, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
        """, (incident_code, violation_id, worker_id, worker_label, violation_type, description, severity, confidence, now_str, duration, source, snapshot_path, site_id, camera_id, zone_id, assigned_to, now_str))
        incident_id = cursor.lastrowid
        if violation_id is not None:
            cursor.execute("UPDATE violations SET incident_id = ? WHERE id = ?", (incident_id, violation_id))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all(limit=100, offset=0, violation_type=None, severity=None, status=None, site_id=None, camera_id=None, search=None, start_date=None, end_date=None):
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT i.*, s.name as site_name, c.name as camera_name, z.name as zone_name 
            FROM incidents i
            LEFT JOIN sites s ON i.site_id = s.id
            LEFT JOIN cameras c ON i.camera_id = c.id
            LEFT JOIN zones z ON i.zone_id = z.id
            WHERE 1=1
        """
        params = []

        if violation_type and violation_type != "ALL":
            query += " AND i.violation_type = ?"
            params.append(violation_type)

        if severity and severity != "ALL":
            query += " AND i.severity = ?"
            params.append(severity)

        if status and status != "ALL":
            query += " AND i.status = ?"
            params.append(status)

        if site_id and site_id != "ALL":
            query += " AND i.site_id = ?"
            params.append(site_id)

        if camera_id and camera_id != "ALL":
            query += " AND i.camera_id = ?"
            params.append(camera_id)

        if start_date:
            query += " AND substr(i.timestamp, 1, 10) >= ?"
            params.append(start_date)

        if end_date:
            query += " AND substr(i.timestamp, 1, 10) <= ?"
            params.append(end_date)

        if search:
            query += " AND (i.worker_label LIKE ? OR i.incident_code LIKE ? OR i.description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        query += " ORDER BY i.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_by_id(incident_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT i.*, s.name as site_name, c.name as camera_name, z.name as zone_name 
            FROM incidents i
            LEFT JOIN sites s ON i.site_id = s.id
            LEFT JOIN cameras c ON i.camera_id = c.id
            LEFT JOIN zones z ON i.zone_id = z.id
            WHERE i.id = ? OR i.incident_code = ?
        """, (incident_id, incident_id))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_recent(limit=10):
        return IncidentModel.get_all(limit=limit)

    @staticmethod
    def get_distinct_types():
        conn = get_db_connection()
        rows = conn.execute("SELECT DISTINCT violation_type FROM incidents ORDER BY violation_type").fetchall()
        conn.close()
        return [r["violation_type"] for r in rows]

    @staticmethod
    def get_count(today_only=False, status=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        if today_only:
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT COUNT(*) as total FROM incidents WHERE substr(timestamp, 1, 10) = ?", (today_str,))
        elif status:
            cursor.execute("SELECT COUNT(*) as total FROM incidents WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) as total FROM incidents")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0

    @staticmethod
    def update_resolution(incident_id, status, assigned_to=None, root_cause=None, corrective_action=None, resolution_note=None, closed_by=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            UPDATE incidents
            SET status = ?, assigned_to = COALESCE(?, assigned_to),
                root_cause = COALESCE(?, root_cause),
                corrective_action = COALESCE(?, corrective_action),
                resolution_note = COALESCE(?, resolution_note),
                closed_by = COALESCE(?, closed_by),
                resolved_at = CASE WHEN ? IN ('RESOLVED', 'CLOSED') THEN ? ELSE resolved_at END
            WHERE id = ? OR incident_code = ?
        """, (status, assigned_to, root_cause, corrective_action, resolution_note, closed_by, status, now_str, incident_id, incident_id))

        # Also update corresponding alert and violation
        cursor.execute("""
            UPDATE alerts
            SET status = ?, assigned_to = COALESCE(?, assigned_to),
                resolved_at = CASE WHEN ? IN ('RESOLVED', 'CLOSED') THEN ? ELSE resolved_at END,
                resolution_note = COALESCE(?, resolution_note)
            WHERE incident_id = ? OR alert_code = ?
        """, (status, assigned_to, status, now_str, resolution_note, incident_id, incident_id))

        conn.commit()
        conn.close()

    @staticmethod
    def get_analytics_summary():
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total counts
        cursor.execute("SELECT COUNT(*) as total FROM violations")
        total_violations = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM incidents")
        total_incidents = cursor.fetchone()["total"]

        # Today's count
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) as total FROM violations WHERE substr(timestamp, 1, 10) = ?", (today_str,))
        today_violations = cursor.fetchone()["total"]

        # Resolved count
        cursor.execute("SELECT COUNT(*) as total FROM incidents WHERE status IN ('RESOLVED', 'CLOSED')")
        resolved_incidents = cursor.fetchone()["total"]

        # Severity breakdown
        cursor.execute("SELECT severity, COUNT(*) as count FROM violations GROUP BY severity")
        severity_dist = {r["severity"]: r["count"] for r in cursor.fetchall()}

        # Violation type breakdown
        cursor.execute("SELECT violation_type, COUNT(*) as count FROM violations GROUP BY violation_type")
        type_dist = {r["violation_type"]: r["count"] for r in cursor.fetchall()}

        # Violations by source
        cursor.execute("SELECT source, COUNT(*) as count FROM violations GROUP BY source")
        source_dist = {r["source"]: r["count"] for r in cursor.fetchall()}

        # Hourly breakdown
        cursor.execute("""
            SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as count
            FROM violations
            WHERE length(timestamp) >= 13
            GROUP BY hour
            ORDER BY hour ASC
        """)
        hourly_dist = {f"{r['hour']}:00": r["count"] for r in cursor.fetchall()}

        # Real timeline (Last 14 days)
        cursor.execute("""
            SELECT substr(timestamp, 1, 10) as day, COUNT(*) as count 
            FROM violations 
            GROUP BY day 
            ORDER BY day DESC 
            LIMIT 14
        """)
        timeline_rows = cursor.fetchall()
        timeline = {r["day"]: r["count"] for r in reversed(timeline_rows)}

        # Camera-wise violations
        cursor.execute("""
            SELECT COALESCE(c.name, v.source, 'Primary Camera') as camera_name, COUNT(*) as count
            FROM violations v
            LEFT JOIN cameras c ON v.camera_id = c.id
            GROUP BY camera_name
            ORDER BY count DESC
            LIMIT 8
        """)
        camera_dist = {r["camera_name"]: r["count"] for r in cursor.fetchall()}

        # Site-wise violations
        cursor.execute("""
            SELECT COALESCE(s.name, 'Default Site') as site_name, COUNT(*) as count
            FROM violations v
            LEFT JOIN sites s ON v.site_id = s.id
            GROUP BY site_name
            ORDER BY count DESC
            LIMIT 8
        """)
        site_dist = {r["site_name"]: r["count"] for r in cursor.fetchall()}

        # Missing-PPE counts per category. PPE violation descriptions end with
        # "...: Head, Visibility" (see SafetyRuleEngine.evaluate_rules).
        ppe_category_counts = {cat: 0 for cat in PPE_CATEGORY_IDS}
        title_to_cat = {cat.replace("_", " ").title(): cat for cat in PPE_CATEGORY_IDS}
        cursor.execute("SELECT description FROM violations WHERE violation_type LIKE '%!_NO!_%' ESCAPE '!'")
        for r in cursor.fetchall():
            desc = r["description"] or ""
            if ":" not in desc:
                continue
            for item in desc.rsplit(":", 1)[1].split(","):
                cat = title_to_cat.get(item.strip())
                if cat:
                    ppe_category_counts[cat] += 1

        cursor.execute("SELECT COUNT(*) as total FROM violations WHERE violation_type = 'RESTRICTED_ZONE_BREACH'")
        restricted_breach_count = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM violations WHERE violation_type NOT LIKE 'SAFE!_%' ESCAPE '!'")
        zone_violation_count = cursor.fetchone()["total"]

        conn.close()

        return {
            "total_violations": total_violations,
            "total_incidents": total_incidents,
            "today_violations": today_violations,
            "resolved_incidents": resolved_incidents,
            "open_incidents": total_incidents - resolved_incidents,
            "severity_distribution": severity_dist,
            "type_distribution": type_dist,
            "source_distribution": source_dist,
            "hourly_distribution": hourly_dist,
            "timeline": timeline,
            "camera_distribution": camera_dist,
            "site_distribution": site_dist,
            "ppe_category_counts": ppe_category_counts,
            "restricted_breach_count": restricted_breach_count,
            "zone_violation_count": zone_violation_count
        }


class AlertModel:
    @staticmethod
    def create(alert_code, violation_id=None, incident_id=None, violation_type="SAFETY_VIOLATION", severity="MEDIUM", camera_id=None, site_id=None, assigned_to=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT OR IGNORE INTO alerts (alert_code, violation_id, incident_id, violation_type, severity, status, camera_id, site_id, assigned_to, timestamp)
            VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
        """, (alert_code, violation_id, incident_id, violation_type, severity, camera_id, site_id, assigned_to, now_str))

        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_all(limit=100, offset=0, severity=None, status=None, search=None):
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT a.*, v.worker_label, v.description, v.snapshot_path, v.confidence, v.source,
                   s.name as site_name, c.name as camera_name
            FROM alerts a
            LEFT JOIN violations v ON a.violation_id = v.id OR a.incident_id = v.incident_id
            LEFT JOIN sites s ON a.site_id = s.id
            LEFT JOIN cameras c ON a.camera_id = c.id
            WHERE 1=1
        """
        params = []

        if severity and severity != "ALL":
            query += " AND a.severity = ?"
            params.append(severity)

        if status and status != "ALL":
            query += " AND a.status = ?"
            params.append(status)

        if search:
            query += " AND (a.alert_code LIKE ? OR a.violation_type LIKE ? OR v.worker_label LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        query += " ORDER BY a.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_by_id(alert_id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, v.worker_label, v.description, v.snapshot_path, v.confidence, v.source,
                   s.name as site_name, c.name as camera_name
            FROM alerts a
            LEFT JOIN violations v ON a.violation_id = v.id
            LEFT JOIN sites s ON a.site_id = s.id
            LEFT JOIN cameras c ON a.camera_id = c.id
            WHERE a.id = ? OR a.alert_code = ?
        """, (alert_id, alert_id))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_open_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM alerts WHERE status NOT IN ('RESOLVED', 'CLOSED')")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0

    @staticmethod
    def update_lifecycle(alert_id, status, assigned_to=None, resolution_note=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "ACKNOWLEDGED":
            cursor.execute("""
                UPDATE alerts SET status = 'ACKNOWLEDGED', acknowledged_at = ?, assigned_to = COALESCE(?, assigned_to)
                WHERE id = ? OR alert_code = ?
            """, (now_str, assigned_to, alert_id, alert_id))
        elif status == "UNDER_REVIEW":
            cursor.execute("""
                UPDATE alerts SET status = 'UNDER_REVIEW', assigned_to = COALESCE(?, assigned_to)
                WHERE id = ? OR alert_code = ?
            """, (assigned_to, alert_id, alert_id))
        elif status in ("RESOLVED", "CLOSED"):
            cursor.execute("""
                UPDATE alerts SET status = 'RESOLVED', resolved_at = ?, resolution_note = ?, assigned_to = COALESCE(?, assigned_to)
                WHERE id = ? OR alert_code = ?
            """, (now_str, resolution_note, assigned_to, alert_id, alert_id))

        conn.commit()
        conn.close()


class WorkerTrackModel:
    @staticmethod
    def record_observation(track_code, label, is_compliant=True, has_violation=False):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("SELECT * FROM workers_track WHERE track_code = ?", (track_code,))
        row = cursor.fetchone()

        if row:
            cursor.execute("""
                UPDATE workers_track
                SET last_seen = ?,
                    total_observations = total_observations + 1,
                    compliant_observations = compliant_observations + ?,
                    violation_count = violation_count + ?
                WHERE track_code = ?
            """, (now_str, 1 if is_compliant else 0, 1 if has_violation else 0, track_code))
        else:
            cursor.execute("""
                INSERT INTO workers_track (track_code, label, first_seen, last_seen, total_observations, compliant_observations, violation_count)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (track_code, label, now_str, now_str, 1 if is_compliant else 0, 1 if has_violation else 0))

        conn.commit()
        conn.close()

    @staticmethod
    def get_all():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workers_track ORDER BY last_seen DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()

        workers = []
        for r in rows:
            w = dict(r)
            tot = max(1, w["total_observations"])
            comp = w["compliant_observations"]
            w["compliance_rate"] = round((comp / tot) * 100, 1)
            workers.append(w)
        return workers

    @staticmethod
    def get_summary_stats():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_tracked, SUM(total_observations) as total_obs, SUM(compliant_observations) as comp_obs, SUM(violation_count) as total_viols FROM workers_track")
        row = cursor.fetchone()
        conn.close()
        if not row or not row["total_tracked"]:
            return {"total_tracked": 0, "compliance_rate": None, "total_violations": 0}
        total_obs = row["total_obs"] or 0
        comp_obs = row["comp_obs"] or 0
        comp_rate = round((comp_obs / total_obs) * 100, 1) if total_obs > 0 else None
        return {
            "total_tracked": row["total_tracked"] or 0,
            "compliance_rate": comp_rate,
            "total_violations": row["total_viols"] or 0
        }


class NotificationModel:
    @staticmethod
    def create(title, message, severity="INFO", incident_id=None, violation_id=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO notifications (title, message, severity, is_read, incident_id, violation_id, timestamp)
            VALUES (?, ?, ?, 0, ?, ?, ?)
        """, (title, message, severity, incident_id, violation_id, now_str))
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_recent(limit=20):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_unread_count():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM notifications WHERE is_read = 0")
        row = cursor.fetchone()
        conn.close()
        return row["total"] if row else 0

    @staticmethod
    def mark_all_read():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE notifications SET is_read = 1 WHERE is_read = 0")
        conn.commit()
        conn.close()


class AuditLogModel:
    @staticmethod
    def log(user_role, action, category, details=None, target_id=None, ip_address=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, user_role, action, category, details, target_id, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (now_str, user_role, action, category, str(details or ""), str(target_id or ""), str(ip_address or "")))
        conn.commit()
        conn.close()

    @staticmethod
    def get_all(limit=100, offset=0, category=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        if category and category != "ALL":
            cursor.execute("SELECT * FROM audit_logs WHERE category = ? ORDER BY id DESC LIMIT ? OFFSET ?", (category, limit, offset))
        else:
            cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]


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
