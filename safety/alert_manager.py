"""
Safety Module: Alert Manager
Handles incident numbering, snapshot cropping/saving, and database persistence.
"""
import os
import cv2
import time
import datetime
import config
from database.models import IncidentModel

class AlertManager:
    def __init__(self, snapshots_dir=config.SNAPSHOTS_DIR):
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.recent_alerts = []
        self._sync_incident_counter()

    def _sync_incident_counter(self):
        count = IncidentModel.get_count()
        self.next_incident_num = count + 1

    def record_incident(self, incident_data, frame=None, source="Webcam"):
        """
        Processes a confirmed incident:
        1. Formats incident code (INC-XXXX)
        2. Captures/saves snapshot if frame provided
        3. Writes to SQLite database
        4. Updates in-memory recent alerts list
        """
        code = f"INC-{self.next_incident_num:04d}"
        self.next_incident_num += 1

        worker_id = incident_data.get("worker_id", 1)
        worker_label = incident_data.get("label", f"Worker #{worker_id}")
        v_type = incident_data.get("violation_type", "SAFETY_VIOLATION")
        desc = incident_data.get("description", "Safety rule breach detected")
        severity = incident_data.get("severity", "MEDIUM")
        confidence = float(incident_data.get("confidence", 0.85))

        snapshot_rel_path = None
        if frame is not None:
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snap_name = f"{code}_{worker_label.replace(' ', '_')}_{ts_str}.jpg"
            snap_full_path = os.path.join(self.snapshots_dir, snap_name)

            # Crop or save full annotated frame
            snap_img = frame.copy()
            bbox = incident_data.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                # Highlight the worker with a red box in snapshot
                cv2.rectangle(snap_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.imwrite(snap_full_path, snap_img)
            snapshot_rel_path = f"snapshots/{snap_name}"

        # Persist to database
        db_id = IncidentModel.create(
            incident_code=code,
            worker_id=worker_id,
            worker_label=worker_label,
            violation_type=v_type,
            description=desc,
            severity=severity,
            confidence=confidence,
            source=source,
            snapshot_path=snapshot_rel_path
        )

        alert_entry = {
            "id": db_id,
            "incident_code": code,
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

        print(f"[Alert] Registered {code} | {worker_label} | {v_type} | {severity} | Source: {source}")
        return alert_entry

    def get_recent_alerts(self, limit=10):
        return self.recent_alerts[:limit]
