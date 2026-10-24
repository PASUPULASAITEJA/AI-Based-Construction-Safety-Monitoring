"""
Safety Module: Alert Manager
Handles incident numbering, snapshot cropping/saving, database persistence,
alert lifecycle creation, notifications, and worker tracking integration.
"""
import os
import cv2
import time
import datetime
import config
from database.db import get_db_connection
from database.models import (
    IncidentModel, AlertModel, NotificationModel, AuditLogModel, WorkerTrackModel, ViolationModel,
    RECORD_CODE_LOCK, next_record_number
)

class AlertManager:
    def __init__(self, snapshots_dir=config.SNAPSHOTS_DIR):
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.recent_alerts = []

    def record_incident(self, incident_data, frame=None, source="Webcam", site_id=None, camera_id=None):
        """
        Processes a confirmed incident:
        1. Formats violation and incident codes
        2. Captures/saves snapshot if frame provided
        3. Writes to SQLite database (violations, incidents & alerts tables)
        4. Dispatches in-app notification for HIGH & CRITICAL alerts
        5. Logs audit trail & updates worker tracking stats
        6. Updates in-memory recent alerts list
        """
        worker_id = incident_data.get("worker_id", 1)
        worker_label = incident_data.get("label", f"Worker #{worker_id}")
        v_type = incident_data.get("violation_type", "SAFETY_VIOLATION")
        desc = incident_data.get("description", "Safety rule breach detected")
        severity = incident_data.get("severity", "MEDIUM")
        confidence = float(incident_data.get("confidence", 0.85))

        # Codes are allocated and the three rows inserted under one process-wide lock, so concurrent
        # pipelines (live stream, browser camera, uploads) never collide on the shared sequence.
        with RECORD_CODE_LOCK:
            conn = get_db_connection()
            num = next_record_number(conn.cursor())
            conn.close()
            code = f"INC-{num:04d}"
            viol_code = f"VIO-{num:04d}"
            alert_code = f"ALT-{num:04d}"
            snapshot_rel_path, viol_id, db_id = self._persist(
                incident_data, frame, source, site_id, camera_id,
                code, viol_code, alert_code, worker_id, worker_label, v_type, desc, severity, confidence
            )

        # Create Notification if HIGH or CRITICAL
        if severity in ("HIGH", "CRITICAL"):
            NotificationModel.create(
                title=f"🚨 {severity} Alert: {v_type}",
                message=f"{worker_label} triggered {v_type} on {source}. Immediate safety inspection required.",
                severity=severity,
                incident_id=db_id,
                violation_id=viol_id
            )

        # Audit Log
        AuditLogModel.log(
            user_role="SYSTEM_AI",
            action="VIOLATION_TRIGGERED",
            category="DETECTION",
            details=f"Code: {code}, Type: {v_type}, Severity: {severity}, Source: {source}",
            target_id=code
        )

        # Worker Track Stats
        track_code = incident_data.get("track_code") or f"TRK-{worker_id:03d}"
        WorkerTrackModel.record_observation(
            track_code=track_code,
            label=worker_label,
            is_compliant=False,
            has_violation=True
        )

        alert_entry = {
            "id": db_id,
            "incident_code": code,
            "alert_code": alert_code,
            "worker_label": worker_label,
            "violation_type": v_type,
            "description": desc,
            "severity": severity,
            "confidence": round(confidence * 100, 1),
            "source": source,
            "snapshot_path": snapshot_rel_path,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        }

        self.recent_alerts.insert(0, alert_entry)
        if len(self.recent_alerts) > 50:
            self.recent_alerts.pop()

        print(f"[AlertManager] Logged {code} ({alert_code}) | {worker_label} | {v_type} | {severity} | Source: {source}")
        return alert_entry

    def _persist(self, incident_data, frame, source, site_id, camera_id,
                 code, viol_code, alert_code, worker_id, worker_label, v_type, desc, severity, confidence):
        """Saves the snapshot and writes the violation, incident and alert rows. Returns (snapshot, violation_id, incident_id)."""
        snapshot_rel_path = None
        if frame is not None:
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snap_name = f"{code}_{worker_label.replace(' ', '_').replace('#', '')}_{ts_str}.jpg"
            snap_full_path = os.path.join(self.snapshots_dir, snap_name)

            snap_img = frame.copy()
            bbox = incident_data.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Highlight worker with high-visibility red box and label banner
                cv2.rectangle(snap_img, (x1, y1), (x2, y2), (0, 0, 235), 3)
                cv2.rectangle(snap_img, (x1, max(0, y1 - 24)), (x2, y1), (0, 0, 235), -1)
                cv2.putText(snap_img, f"{worker_label} | {v_type}", (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(snap_full_path, snap_img)
            snapshot_rel_path = f"snapshots/{snap_name}"

        # 1. Persist to Violations table
        viol_id = ViolationModel.create(
            violation_code=viol_code,
            worker_id=worker_id,
            worker_label=worker_label,
            violation_type=v_type,
            description=desc,
            severity=severity,
            confidence=confidence,
            source=source,
            snapshot_path=snapshot_rel_path,
            duration_sec=incident_data.get("duration_sec", 0.0),
            site_id=site_id,
            zone_id=incident_data.get("zone_id"),
            camera_id=camera_id
        )

        # 2. Persist to Incidents table
        db_id = IncidentModel.create(
            incident_code=code,
            violation_id=viol_id,
            worker_id=worker_id,
            worker_label=worker_label,
            violation_type=v_type,
            description=desc,
            severity=severity,
            confidence=confidence,
            source=source,
            snapshot_path=snapshot_rel_path,
            duration=incident_data.get("duration_sec", 0.0),
            site_id=site_id,
            camera_id=camera_id,
            zone_id=incident_data.get("zone_id")
        )

        # 3. Persist to Alerts table
        AlertModel.create(
            alert_code=alert_code,
            incident_id=db_id,
            violation_id=viol_id,
            violation_type=v_type,
            severity=severity,
            camera_id=camera_id,
            site_id=site_id
        )
        return snapshot_rel_path, viol_id, db_id

    def get_recent_alerts(self, limit=10):
        return self.recent_alerts[:limit]

