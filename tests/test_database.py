"""
Tests for SQLite Database Persistence, Models, and Incident Lifecycles.
"""
import pytest
from database.db import get_db_connection
from database.models import (
    SiteModel, CameraModel, ZoneModel, ViolationModel, IncidentModel, AlertModel,
    WorkerTrackModel, NotificationModel, AuditLogModel, SettingsModel
)

def test_database_connection_and_table_existence():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r["name"] for r in cursor.fetchall()}
    conn.close()

    expected_tables = {
        "users", "sites", "cameras", "zones", "violations",
        "incidents", "alerts", "workers_track", "notifications",
        "audit_logs", "settings"
    }
    for t in expected_tables:
        assert t in tables, f"Expected table '{t}' missing from database schema"

def test_site_and_camera_models():
    # Site creation
    site_id = SiteModel.create("TEST-SITE-01", "Test Plant Alpha", "Sector 7", "Project Alpha", "Officer Smith")
    assert site_id is not None
    site = SiteModel.get_by_id(site_id)
    assert site["name"] == "Test Plant Alpha"

    # Camera creation
    cam_id = CameraModel.create("CAM-TEST-01", "North Gate Cam", site_id, None, "0", "webcam")
    assert cam_id is not None
    cam = CameraModel.get_by_id(cam_id)
    assert cam["name"] == "North Gate Cam"

    # Cleanup
    CameraModel.delete(cam_id)
    SiteModel.delete(site_id)

def test_incident_lifecycle_workflow():
    import uuid
    code = f"VIOL-TEST-{uuid.uuid4().hex[:6].upper()}"
    # Insert violation
    v_id = ViolationModel.create(
        violation_code=code,
        worker_id=99,
        worker_label="Worker #99",
        violation_type="NO_HELMET",
        description="Worker observed without safety helmet",
        severity="CRITICAL",
        confidence=0.92,
        duration_sec=3.5,
        source="Test Stream"
    )
    assert v_id is not None

    # Confirm violation and escalate to incident
    inc_code = ViolationModel.confirm_violation(v_id, reviewer="Safety Officer Lead", notes="Confirmed via test")
    assert inc_code is not None

    # Verify incident
    inc = IncidentModel.get_by_id(inc_code)
    assert inc is not None
    assert inc["status"] in ("OPEN", "UNDER_REVIEW")

    # Update to resolved with root cause and corrective action
    IncidentModel.update_resolution(
        incident_id=inc["id"],
        status="RESOLVED",
        assigned_to="Supervisor Dan",
        root_cause="Defective helmet harness",
        corrective_action="Issued new Class E hardhat",
        resolution_note="Resolved on site"
    )

    resolved_inc = IncidentModel.get_by_id(inc["id"])
    assert resolved_inc["status"] == "RESOLVED"
    assert resolved_inc["root_cause"] == "Defective helmet harness"
