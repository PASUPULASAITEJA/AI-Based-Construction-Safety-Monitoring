"""
SiteGuard AI — Intelligent Construction Safety Management Platform
Main Web Application, RESTful APIs, Role-Based Access, & Real-Time AI Inference Server
"""
import os
import io
import csv
import time
import uuid
import base64
import shutil
import secrets
import sqlite3
import threading
import datetime
from functools import wraps
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for, session, flash, abort
)

import config
from cv.pipeline import SafetyPipeline
from database.db import init_db, get_db_connection
from database.models import (
    UserModel, SiteModel, CameraModel, ZoneModel, ViolationModel, IncidentModel, AlertModel,
    WorkerTrackModel, NotificationModel, AuditLogModel, SettingsModel
)
from safety.ppe_config import get_ppe_capabilities
from safety.safety_rules import ZONE_RULES

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max upload
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

APP_START_TIME = time.time()

# Initialize Database Schema & Migrations
init_db()

# Master Safety Computer Vision Pipeline (hardware camera stream). Other pipelines share its loaded model.
pipeline = SafetyPipeline()
# Browser-camera frames get their own tracker/temporal state so worker IDs don't mix with the hardware stream
browser_pipeline = SafetyPipeline(detector=pipeline.detector)
LIVE_PIPELINES = (pipeline, browser_pipeline)

# ==============================================================================
# Authentication, CSRF & Role-Based Access Control (RBAC)
# ==============================================================================

ALL_ROLES = ("ADMIN", "SAFETY_OFFICER", "SUPERVISOR", "VIEWER")
OPERATOR_ROLES = ("ADMIN", "SAFETY_OFFICER", "SUPERVISOR")   # review, resolve, run analyses
ZONE_EDITOR_ROLES = ("ADMIN", "SAFETY_OFFICER")
ADMIN_ROLES = ("ADMIN",)

PUBLIC_ENDPOINTS = {"login_route", "static"}
JSON_PATH_PREFIXES = ("/api/", "/upload/", "/camera/", "/zones", "/video_feed")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")

def wants_json():
    return request.path.startswith(JSON_PATH_PREFIXES)

def get_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]

def error_response(message, status):
    if wants_json():
        return jsonify({"success": False, "message": message}), status
    abort(status, description=message)

@app.before_request
def enforce_auth_and_csrf():
    # The sign-in form is left exactly as it was (no CSRF field)
    if request.method not in SAFE_METHODS and request.endpoint != "login_route":
        sent = request.headers.get("X-CSRFToken") or request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not expected or not secrets.compare_digest(str(sent), expected):
            return error_response("Invalid or missing CSRF token. Reload the page and try again.", 400)

    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if "user_id" not in session:
        if wants_json():
            return jsonify({"success": False, "message": "Authentication required."}), 401
        return redirect(url_for("login_route", next=request.full_path.rstrip("?")))
    return None

def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("role") not in roles:
                return error_response("Your role is not allowed to perform this action.", 403)
            return view(*args, **kwargs)
        return wrapped
    return decorator

def current_username():
    return session.get("username", "unknown")

def audit(action, category, details, target_id=None):
    """Audit entry attributed to the signed-in user (role, username and client IP)."""
    AuditLogModel.log(
        user_role=session.get("role", "ANONYMOUS"),
        action=action,
        category=category,
        details=f"[{current_username()}] {details}",
        target_id=target_id,
        ip_address=request.remote_addr
    )

@app.context_processor
def inject_global_counts():
    try:
        open_alerts = AlertModel.get_open_count()
        unread_notifs = NotificationModel.get_unread_count()
    except Exception:
        open_alerts = 0
        unread_notifs = 0
    return {
        "open_alerts_count": open_alerts,
        "unread_notifs_count": unread_notifs,
        "current_user": session.get("username", ""),
        "current_role": session.get("role", "VIEWER"),
        "is_authenticated": "user_id" in session,
        "ppe_capabilities": get_ppe_capabilities(),
        "csrf_token": get_csrf_token()
    }

@app.route("/login", methods=["GET", "POST"])
def login_route():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = UserModel.authenticate(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            AuditLogModel.log(user["role"], "USER_LOGIN", "AUTHENTICATION", f"User {username} signed in.", str(user["id"]))
            return redirect(url_for("dashboard_view"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout_route():
    user = session.get("username", "Unknown")
    role = session.get("role", "VIEWER")
    AuditLogModel.log(role, "USER_LOGOUT", "AUTHENTICATION", f"User {user} signed out.")
    session.clear()
    return redirect(url_for("login_route"))

# ==============================================================================
# Camera & High-Performance Background Stream Engine (Zero Fake Fallback)
# ==============================================================================

def empty_frame_summary():
    return {
        "fps": 0.0,
        "worker_count": 0,
        "compliant_count": 0,
        "violation_count": 0,
        "critical_count": 0,
        "workers": [],
        "is_live": False
    }

def parse_stream_source(source):
    """'0' / 0 -> local device index 0; anything else (rtsp://, http://, file path) is passed to OpenCV as-is."""
    src = str(source if source is not None else "0").strip()
    return int(src) if src.isdigit() else src

class VideoCamera:
    """
    Hardware Camera and RTSP Stream interface.
    STRICT PRODUCTION RULE: Never substitutes dataset images or synthetic frames
    when a hardware feed is disconnected. Returns explicit 'Camera Offline' state.
    """
    def __init__(self):
        self.cap = None
        self.is_running = True
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.source = 0            # device index or stream URL
        self.camera_id = None      # registered camera row (None = default webcam)
        self.site_id = None

    def describe(self):
        if isinstance(self.source, int):
            return f"Local camera device {self.source}"
        return "Network stream"

    def _open_hardware_camera(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                return True
            self.cap = None
            if isinstance(self.source, int) and os.name == "nt":
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)

            if self.cap and self.cap.isOpened():
                if isinstance(self.source, int):
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._set_db_status("ONLINE")
                return True
        except Exception as e:
            print(f"[-] Hardware camera open error: {e}")
        return False

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[-] Camera release error: {e}")
            self.cap = None
            self._set_db_status("OFFLINE")

    def _set_db_status(self, status):
        if self.camera_id is not None:
            try:
                CameraModel.update_status(self.camera_id, status)
            except Exception as e:
                print(f"[-] Could not update camera status: {e}")

    def is_open(self):
        with self.lock:
            return self.cap is not None and self.cap.isOpened()

    def start(self, source=None, camera_id=None, site_id=None):
        with self.lock:
            if source is not None:
                new_source = parse_stream_source(source)
                if new_source != self.source or camera_id != self.camera_id:
                    self._release_cap()
                    self.source = new_source
                    self.camera_id = camera_id
                    self.site_id = site_id
            self.is_running = True
            return self._open_hardware_camera()

    def stop(self):
        with self.lock:
            self.is_running = False
            self._release_cap()
            return True

    def release_hardware(self):
        """Frees the device without pausing monitoring (used when the last viewer disconnects)."""
        with self.lock:
            self._release_cap()

    def get_raw_frame(self):
        """
        Returns (frame, is_live_stream).
        If camera is offline or stopped, returns a crisp industrial status card.
        """
        with self.lock:
            if not self.is_running:
                idle = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(idle, (20, 20), (620, 460), (35, 45, 55), 2)
                cv2.putText(idle, "CAMERA MONITORING PAUSED", (170, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 160, 170), 2, cv2.LINE_AA)
                cv2.putText(idle, "Click 'Start Camera' to resume live hardware capture", (140, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 110, 120), 1, cv2.LINE_AA)
                return idle, False

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame_time = time.time()
                    return frame, True

            # Attempt reconnection
            if self._open_hardware_camera():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame_time = time.time()
                    return frame, True

            # Offline State Placeholder (Real State - No Fake Data)
            offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(offline_frame, (10, 10), (630, 470), (25, 30, 40), -1)
            cv2.rectangle(offline_frame, (20, 20), (620, 460), (40, 50, 65), 2)

            # Red Offline Badge
            cv2.circle(offline_frame, (50, 50), 8, (0, 0, 235), -1)
            cv2.putText(offline_frame, "CAMERA OFFLINE", (70, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 235), 2, cv2.LINE_AA)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(offline_frame, f"System Time: {now_str}", (70, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)

            cv2.putText(offline_frame, "NO HARDWARE CAMERA / RTSP STREAM CONNECTED", (105, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 225), 1, cv2.LINE_AA)
            cv2.putText(offline_frame, "Verify USB video device or RTSP endpoint in Camera Network", (100, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 120, 135), 1, cv2.LINE_AA)
            cv2.putText(offline_frame, "Use Image/Video Analysis tabs to inspect recorded media", (125, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 158, 11), 1, cv2.LINE_AA)

            return offline_frame, False

camera = VideoCamera()

# Real-time state telemetry cache
latest_frame_summary = empty_frame_summary()

class StreamEngine:
    """
    Runs the CV pipeline on the camera feed only while at least one /video_feed client
    is connected, so no inference, webcam access, or incident logging happens unattended.
    """
    def __init__(self):
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.viewers = 0
        self.frames_processed = 0
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def add_viewer(self):
        with self.lock:
            self.viewers += 1

    def remove_viewer(self):
        with self.lock:
            self.viewers = max(0, self.viewers - 1)

    def has_viewers(self):
        with self.lock:
            return self.viewers > 0

    def _worker_loop(self):
        global latest_frame_summary
        idle = True
        while True:
            if not self.has_viewers():
                if not idle:
                    # Last viewer left: free the webcam, clear stale live telemetry, start a new tracking session
                    camera.release_hardware()
                    latest_frame_summary = empty_frame_summary()
                    pipeline.reset()
                    with self.lock:
                        self.latest_jpeg = None
                    idle = True
                time.sleep(0.2)
                continue
            idle = False

            try:
                frame, is_live = camera.get_raw_frame()
                if frame is not None:
                    if is_live:
                        annotated_frame, summary = pipeline.process_frame(
                            frame, source="Live Stream", site_id=camera.site_id, camera_id=camera.camera_id)
                        summary["is_live"] = True
                        summary.pop("new_incidents", None)
                        latest_frame_summary = summary
                        self.frames_processed += 1
                    else:
                        annotated_frame = frame
                        latest_frame_summary = empty_frame_summary()

                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        with self.lock:
                            self.latest_jpeg = buffer.tobytes()
            except Exception as e:
                print(f"[-] Live stream processing error: {e}")
                time.sleep(0.1)
            time.sleep(0.03)

    def get_latest_frame_bytes(self):
        with self.lock:
            return self.latest_jpeg

stream_engine = StreamEngine()

def generate_video_stream():
    """Multi-client MJPEG distribution; each connected client counts as a viewer until it disconnects."""
    stream_engine.add_viewer()
    try:
        while True:
            frame_bytes = stream_engine.get_latest_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04)
    finally:
        stream_engine.remove_viewer()

# ==============================================================================
# Helpers
# ==============================================================================

def default_settings():
    """Values the Settings page falls back to (and 'Reset Defaults' restores)."""
    return {
        "conf_thresh": str(config.CONFIDENCE_THRESHOLD),
        "iou_thresh": str(config.IOU_THRESHOLD),
        "min_violation_frames": str(config.MIN_VIOLATION_FRAMES),
        "min_zone_frames": str(config.MIN_ZONE_FRAMES),
    }

def optional_int(value):
    if value in (None, "", "null", "None"):
        return None
    return int(value)

def clean_text(value, max_len=200):
    """Stripped string (or None if blank), truncated to max_len."""
    if value is None:
        return None
    value = str(value).strip()
    return value[:max_len] if value else None

def allowed_extension(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def save_upload(file, allowed):
    """Validates the extension and saves under a unique, sanitized name. Returns (path, stem, ext) or raises ValueError."""
    original = secure_filename(file.filename or "")
    if not original or not allowed_extension(original, allowed):
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}.")
    ext = original.rsplit(".", 1)[1].lower()
    stem = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(config.UPLOADS_DIR, f"upload_{stem}.{ext}")
    file.save(path)
    return path, stem, ext

def csv_safe(value):
    """Neutralizes spreadsheet formula injection in exported cells."""
    text = "" if value is None else str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text

# ==============================================================================
# Primary View Routes
# ==============================================================================

@app.route("/")
def dashboard_view():
    recent = IncidentModel.get_recent(limit=6)
    today_count = IncidentModel.get_count(today_only=True)
    open_alerts = AlertModel.get_open_count()
    resolved_count = IncidentModel.get_count(status="RESOLVED")
    worker_stats = WorkerTrackModel.get_summary_stats()

    total_sites = SiteModel.get_count()
    total_cameras = CameraModel.get_count()
    active_cameras = CameraModel.get_active_count()
    total_zones = len(ZoneModel.get_all())

    summary_stats = {
        "workers_count": latest_frame_summary.get("worker_count", 0),
        "violations_count": latest_frame_summary.get("violation_count", 0),
        "critical_count": latest_frame_summary.get("critical_count", 0)
    }

    return render_template(
        "index.html",
        active_page="dashboard",
        recent_alerts=recent,
        stats=summary_stats,
        today_violations_count=today_count,
        open_alerts_count=open_alerts,
        resolved_incidents_count=resolved_count,
        compliance_rate=worker_stats.get("compliance_rate"),
        total_sites_count=total_sites,
        total_cameras_count=total_cameras,
        active_cameras_count=active_cameras,
        total_zones_count=total_zones
    )

@app.route("/monitor")
def monitor_view():
    sites = SiteModel.get_all()
    cameras = CameraModel.get_all()
    return render_template("monitor.html", active_page="monitor", sites=sites, cameras=cameras)

@app.route("/video-analysis")
def video_analysis_view():
    return render_template("video_analysis.html", active_page="video_analysis")

@app.route("/image-analysis")
def image_analysis_view():
    return render_template("image_analysis.html", active_page="image_analysis")

@app.route("/upload")
def upload_view():
    return render_template("upload.html", active_page="upload")

@app.route("/violations")
def violations_view():
    v_type = request.args.get("violation_type", "ALL")
    sev = request.args.get("severity", "ALL")
    status = request.args.get("status", "ALL")
    site_id = request.args.get("site_id", "ALL")
    search = request.args.get("search", "")

    violations = ViolationModel.get_all(
        limit=200,
        violation_type=v_type,
        severity=sev,
        status=status,
        site_id=site_id,
        search=search
    )
    review_stats = ViolationModel.get_stats()
    sites = SiteModel.get_all()

    return render_template(
        "violations.html",
        active_page="violations",
        violations=violations,
        review_stats=review_stats,
        sites=sites,
        violation_types=ViolationModel.get_distinct_types(),
        current_v_type=v_type,
        current_sev=sev,
        current_status=status,
        current_site=site_id,
        current_search=search
    )

@app.route("/api/violations/<int:v_id>/confirm", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def confirm_violation_api(v_id):
    data = request.get_json(silent=True) or {}
    reviewer = current_username()
    notes = clean_text(data.get("notes"), 2000) or ""

    inc_code = ViolationModel.confirm_violation(violation_id=v_id, reviewer=reviewer, notes=notes)
    if inc_code:
        audit("VIOLATION_CONFIRMED", "HUMAN_REVIEW",
              f"Violation #{v_id} confirmed by {reviewer}. Escalated to {inc_code}.", str(v_id))
        return jsonify({"success": True, "incident_code": inc_code})
    return jsonify({"success": False, "message": "Violation not found."}), 404

@app.route("/api/violations/<int:v_id>/false-positive", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def mark_false_positive_api(v_id):
    data = request.get_json(silent=True) or {}
    reviewer = current_username()
    reason = clean_text(data.get("reason"), 500) or "Model Misclassification"

    if not ViolationModel.mark_false_positive(violation_id=v_id, reviewer=reviewer, reason=reason):
        return jsonify({"success": False, "message": "Violation not found."}), 404
    audit("FALSE_POSITIVE_FLAGGED", "HUMAN_REVIEW",
          f"Violation #{v_id} marked as False Positive by {reviewer}. Reason: {reason}", str(v_id))
    return jsonify({"success": True})

@app.route("/alerts")
def alerts_view():
    status = request.args.get("status", "ALL")
    sev = request.args.get("severity", "ALL")
    search = request.args.get("search", "")

    alerts = AlertModel.get_all(
        limit=200,
        status=status,
        severity=sev,
        search=search
    )
    open_count = AlertModel.get_open_count()
    return render_template(
        "alerts.html",
        active_page="alerts",
        alerts=alerts,
        open_count=open_count,
        current_status=status,
        current_sev=sev,
        current_search=search
    )

ALERT_STATUSES = ("ACKNOWLEDGED", "UNDER_REVIEW", "RESOLVED", "CLOSED")

@app.route("/api/alerts/<int:alert_id>/action", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def update_alert_action(alert_id):
    data = request.get_json(silent=True) or {}
    new_status = str(data.get("status", "ACKNOWLEDGED")).upper()
    if new_status not in ALERT_STATUSES:
        return jsonify({"success": False, "message": f"Status must be one of {', '.join(ALERT_STATUSES)}."}), 400
    if not AlertModel.get_by_id(alert_id):
        return jsonify({"success": False, "message": "Alert not found."}), 404

    assigned_to = clean_text(data.get("assigned_to"), 120) or current_username()
    resolution_note = clean_text(data.get("resolution_note"), 2000)

    AlertModel.update_lifecycle(
        alert_id=alert_id,
        status=new_status,
        assigned_to=assigned_to,
        resolution_note=resolution_note
    )
    audit(f"ALERT_{new_status}", "ALERT_MANAGEMENT",
          f"Alert #{alert_id} moved to status {new_status}. Note: {resolution_note or 'None'}", f"ALT-{alert_id}")
    return jsonify({"success": True})

@app.route("/incidents")
def incidents_view():
    v_type = request.args.get("violation_type", "ALL")
    sev = request.args.get("severity", "ALL")
    status = request.args.get("status", "ALL")
    search = request.args.get("search", "")

    incidents = IncidentModel.get_all(
        limit=200,
        violation_type=v_type,
        severity=sev,
        status=status,
        search=search
    )
    return render_template(
        "incidents.html",
        active_page="incidents",
        incidents=incidents,
        violation_types=IncidentModel.get_distinct_types(),
        current_v_type=v_type,
        current_sev=sev,
        current_status=status,
        current_search=search
    )

INCIDENT_STATUSES = ("OPEN", "UNDER_REVIEW", "RESOLVED", "CLOSED")

@app.route("/api/incidents/<int:inc_id>/update", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def update_incident_details(inc_id):
    data = request.get_json(silent=True) or {}
    status = str(data.get("status", "RESOLVED")).upper()
    if status not in INCIDENT_STATUSES:
        return jsonify({"success": False, "message": f"Status must be one of {', '.join(INCIDENT_STATUSES)}."}), 400
    if not IncidentModel.get_by_id(inc_id):
        return jsonify({"success": False, "message": "Incident not found."}), 404

    assigned_to = clean_text(data.get("assigned_to"), 120)
    root_cause = clean_text(data.get("root_cause"), 2000)
    corrective_action = clean_text(data.get("corrective_action"), 2000)
    resolution_note = clean_text(data.get("resolution_note"), 2000)
    if status == "RESOLVED" and not root_cause:
        return jsonify({"success": False, "message": "A root cause is required to resolve an incident."}), 400

    IncidentModel.update_resolution(
        incident_id=inc_id,
        status=status,
        assigned_to=assigned_to,
        root_cause=root_cause,
        corrective_action=corrective_action,
        resolution_note=resolution_note,
        closed_by=current_username() if status in ("RESOLVED", "CLOSED") else None
    )
    audit(f"INCIDENT_{status}", "INCIDENT_AUDIT",
          f"Incident #{inc_id} set to {status}. Assigned: {assigned_to or '-'}. Root cause: {root_cause or '-'}", f"INC-{inc_id}")
    return jsonify({"success": True})

@app.route("/sites")
def sites_view():
    sites = SiteModel.get_all()
    zones = ZoneModel.get_all()
    return render_template("sites.html", active_page="sites", sites=sites, zones=zones)

@app.route("/api/sites", methods=["POST"])
@role_required(*ADMIN_ROLES)
def create_site():
    data = request.get_json(silent=True) or {}
    name = clean_text(data.get("name"), 120)
    if not name:
        return jsonify({"success": False, "message": "Site name is required."}), 400
    site_code = clean_text(data.get("site_code"), 40) or f"SITE-{uuid.uuid4().hex[:6].upper()}"

    try:
        site_id = SiteModel.create(
            site_code, name,
            clean_text(data.get("location"), 200) or "",
            clean_text(data.get("project_name"), 200) or "",
            clean_text(data.get("safety_officer"), 120) or ""
        )
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": f"Site code {site_code} already exists."}), 409
    audit("SITE_CREATED", "INFRASTRUCTURE", f"Created Site {name} ({site_code})", str(site_id))
    return jsonify({"success": True, "site_id": site_id})

@app.route("/api/sites/<int:site_id>", methods=["PUT"])
@role_required(*ADMIN_ROLES)
def update_site(site_id):
    if not SiteModel.get_by_id(site_id):
        return jsonify({"success": False, "message": "Site not found."}), 404
    data = request.get_json(silent=True) or {}
    name = clean_text(data.get("name"), 120)
    if not name:
        return jsonify({"success": False, "message": "Site name is required."}), 400
    SiteModel.update(
        site_id=site_id,
        name=name,
        location=clean_text(data.get("location"), 200) or "",
        project_name=clean_text(data.get("project_name"), 200) or "",
        status=(clean_text(data.get("status"), 20) or "ACTIVE").upper(),
        safety_officer=clean_text(data.get("safety_officer"), 120) or ""
    )
    audit("SITE_UPDATED", "INFRASTRUCTURE", f"Updated Site #{site_id} ({name})", str(site_id))
    return jsonify({"success": True})

@app.route("/api/sites/<int:site_id>", methods=["DELETE"])
@role_required(*ADMIN_ROLES)
def delete_site(site_id):
    deleted = SiteModel.delete(site_id)
    if deleted:
        audit("SITE_DELETED", "INFRASTRUCTURE", f"Deleted Site #{site_id}", str(site_id))
    return jsonify({"success": deleted}), (200 if deleted else 404)

@app.route("/cameras")
def cameras_view():
    cameras = CameraModel.get_all()
    sites = SiteModel.get_all()
    return render_template("cameras.html", active_page="cameras", cameras=cameras, sites=sites)

STREAM_TYPES = ("webcam", "rtsp", "http", "file")

@app.route("/api/cameras", methods=["POST"])
@role_required(*ADMIN_ROLES)
def create_camera():
    data = request.get_json(silent=True) or {}
    name = clean_text(data.get("name"), 120)
    if not name:
        return jsonify({"success": False, "message": "Camera name is required."}), 400
    stream_type = (clean_text(data.get("stream_type"), 20) or "webcam").lower()
    if stream_type not in STREAM_TYPES:
        return jsonify({"success": False, "message": f"Stream type must be one of {', '.join(STREAM_TYPES)}."}), 400
    source = clean_text(data.get("stream_source"), 500) or "0"
    try:
        site_id = optional_int(data.get("site_id"))
        zone_id = optional_int(data.get("zone_id"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Invalid site or zone."}), 400

    cam_code = clean_text(data.get("camera_code"), 40)
    if cam_code and CameraModel.code_exists(cam_code):
        return jsonify({"success": False, "message": f"Camera code {cam_code} already exists."}), 409
    if not cam_code:
        cam_code = f"CAM-{uuid.uuid4().hex[:6].upper()}"

    try:
        cam_id = CameraModel.create(cam_code, name, site_id, zone_id, source, stream_type)
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Camera code already exists or the site/zone does not exist."}), 409
    audit("CAMERA_CREATED", "INFRASTRUCTURE", f"Added Camera {name} ({cam_code})", str(cam_id))
    return jsonify({"success": True, "camera_id": cam_id, "camera_code": cam_code})

@app.route("/api/cameras/<int:cam_id>", methods=["DELETE"])
@role_required(*ADMIN_ROLES)
def delete_camera(cam_id):
    deleted = CameraModel.delete(cam_id)
    if deleted:
        audit("CAMERA_DELETED", "INFRASTRUCTURE", f"Deleted Camera #{cam_id}", str(cam_id))
    return jsonify({"success": deleted}), (200 if deleted else 404)

@app.route("/workers")
def workers_view():
    workers = WorkerTrackModel.get_all()
    stats = WorkerTrackModel.get_summary_stats()
    return render_template("workers.html", active_page="workers", workers=workers, worker_stats=stats)

@app.route("/analytics")
def analytics_view():
    summary = IncidentModel.get_analytics_summary()
    return render_template("analytics.html", active_page="analytics", summary=summary)

@app.route("/reports")
def reports_view():
    period = request.args.get("period", "ALL")
    site_id = request.args.get("site_id", "ALL")

    now = datetime.datetime.now()
    start_date = None
    if period == "TODAY":
        start_date = now.strftime("%Y-%m-%d")
    elif period == "7DAYS":
        start_date = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    elif period == "30DAYS":
        start_date = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    incidents = IncidentModel.get_all(limit=1000, site_id=site_id, start_date=start_date)
    sites = SiteModel.get_all()

    crit_count = sum(1 for i in incidents if i["severity"] == "CRITICAL")
    resolved_count = sum(1 for i in incidents if i["status"] == "RESOLVED")
    worker_stats = WorkerTrackModel.get_summary_stats()

    return render_template(
        "reports.html",
        active_page="reports",
        period=period,
        current_site=site_id,
        sites=sites,
        report_incidents=incidents,
        crit_count=crit_count,
        resolved_count=resolved_count,
        report_compliance=worker_stats.get("compliance_rate"),
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/model")
def model_view():
    detector = pipeline.detector
    model_info = {
        "model_path": detector.model_path,
        "device": config.DEVICE.upper(),
        "img_size": config.IMG_SIZE,
        "conf": detector.conf,
        "iou": detector.iou,
        "classes": detector.classes,
        "loaded": getattr(detector, "model", None) is not None,
        "class_count": len(detector.classes)
    }
    return render_template("model.html", active_page="model", model_info=model_info)

@app.route("/insights")
def insights_view():
    summary = IncidentModel.get_analytics_summary()
    total_events = summary["total_incidents"]
    type_dist = summary["type_distribution"]
    sev_dist = summary["severity_distribution"]
    hourly = summary["hourly_distribution"]

    peak_hour = None
    peak_count = 0
    if hourly:
        peak_hour = max(hourly, key=hourly.get)
        peak_count = hourly[peak_hour]

    return render_template(
        "insights.html",
        active_page="insights",
        total_events=total_events,
        type_dist=type_dist,
        sev_dist=sev_dist,
        resolved_count=summary["resolved_incidents"],
        peak_hour=peak_hour,
        peak_count=peak_count
    )

def _format_uptime(seconds):
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"

def collect_health():
    """Real checks behind the System Health page."""
    health = {"db_ok": False, "db_latency_ms": None}
    try:
        t0 = time.perf_counter()
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        health["db_ok"] = True
        health["db_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        print(f"[-] Health check DB error: {e}")

    health["model_loaded"] = getattr(pipeline.detector, "model", None) is not None
    health["model_class_count"] = len(pipeline.detector.classes)
    health["camera_online"] = camera.is_open()
    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    health["snapshots_writable"] = os.access(config.SNAPSHOTS_DIR, os.W_OK)
    health["disk_free_gb"] = round(shutil.disk_usage(config.BASE_DIR).free / (1024 ** 3), 1)
    health["uptime"] = _format_uptime(time.time() - APP_START_TIME)
    return health

@app.route("/system-health")
def system_health_view():
    total_incidents = IncidentModel.get_count()
    snapshots_count = len(os.listdir(config.SNAPSHOTS_DIR)) if os.path.exists(config.SNAPSHOTS_DIR) else 0
    unread = NotificationModel.get_unread_count()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "system_health.html",
        active_page="system_health",
        config_port=config.PORT,
        db_path=config.DB_PATH,
        total_incidents=total_incidents,
        device_str=config.DEVICE.upper(),
        model_path=pipeline.detector.model_path,
        camera_mode=camera.describe(),
        snapshots_count=snapshots_count,
        unread_count=unread,
        now_str=now_str,
        health=collect_health()
    )

@app.route("/ppe")
def ppe_view():
    return render_template("ppe.html", active_page="ppe")

@app.route("/notifications")
def notifications_view():
    notifs = NotificationModel.get_recent(limit=50)
    return render_template("notifications.html", active_page="notifications", notifications=notifs)

@app.route("/api/notifications/mark-read", methods=["POST"])
def mark_notifications_read():
    NotificationModel.mark_all_read()
    return jsonify({"success": True})

@app.route("/audit-logs")
@role_required(*ADMIN_ROLES)
def audit_logs_view():
    logs = AuditLogModel.get_all(limit=100)
    return render_template("audit_logs.html", active_page="audit_logs", audit_logs=logs)

@app.route("/settings")
def settings_view():
    defaults = default_settings()
    settings = {**defaults, **SettingsModel.get_all()}
    device_name = f"PyTorch on {config.DEVICE.upper()}"
    return render_template("settings.html", active_page="settings", settings=settings,
                           defaults=defaults, device_name=device_name)

# ==============================================================================
# Camera & Video Stream Controls
# ==============================================================================

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera/start", methods=["POST"])
def start_camera():
    data = request.get_json(silent=True) or {}
    source, camera_id, site_id = 0, None, None
    if data.get("camera_id") not in (None, ""):
        try:
            cam = CameraModel.get_by_id(int(data["camera_id"]))
        except (TypeError, ValueError):
            cam = None
        if not cam:
            return jsonify({"success": False, "message": "Camera not found."}), 404
        source, camera_id, site_id = cam["stream_source"], cam["id"], cam["site_id"]

    success = camera.start(source=source, camera_id=camera_id, site_id=site_id)
    message = "Camera started successfully." if success else f"Failed to open camera source ({camera.describe()})."
    return jsonify({"success": success, "message": message})

@app.route("/camera/stop", methods=["POST"])
def stop_camera():
    success = camera.stop()
    return jsonify({"success": success, "message": "Camera stopped."})

@app.route("/api/status")
def get_status():
    return jsonify(latest_frame_summary)

@app.route("/api/process_browser_frame", methods=["POST"])
def process_browser_frame():
    """Receives base64 webcam frame from browser, runs CV pipeline, and returns annotated frame + telemetry."""
    global latest_frame_summary
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "No image data"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Invalid image data"}), 400

    if frame is None:
        return jsonify({"success": False, "message": "Failed to decode frame"}), 400

    try:
        annotated_frame, summary = browser_pipeline.process_frame(frame, source="Browser Camera")
        summary["is_live"] = True
        summary.pop("new_incidents", None)
        latest_frame_summary = summary

        ret, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{out_b64}",
            "summary": summary
        })
    except Exception as e:
        print(f"[-] Browser frame processing error: {e}")
        return jsonify({"success": False, "message": "Frame processing failed."}), 500

# ==============================================================================
# Media Upload Endpoints (Real YOLOv8 Inference)
# ==============================================================================

@app.route("/upload/image", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    try:
        upload_path, stem, ext = save_upload(file, config.ALLOWED_IMAGE_EXTENSIONS)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    try:
        annotated_img, stats = pipeline.process_single_image(upload_path, source="Image Inspection")

        out_filename = f"annotated_{stem}.jpg"
        cv2.imwrite(os.path.join(config.OUTPUTS_DIR, out_filename), annotated_img)
        audit("IMAGE_ANALYZED", "DETECTION",
              f"Image {file.filename!r}: {stats['worker_count']} workers, {len(stats['incidents'])} violations")

        return jsonify({
            "success": True,
            "annotated_url": f"/outputs/{out_filename}",
            "stats": stats
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        print(f"[-] Image processing error: {e}")
        return jsonify({"success": False, "message": "Inference error while processing the image."}), 500

def open_video_writer(base_path, fps, size):
    """Tries browser-playable codecs first. Returns (writer, path) or (None, None)."""
    for fourcc, ext in (("avc1", "mp4"), ("VP80", "webm"), ("mp4v", "mp4")):
        path = f"{base_path}.{ext}"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        if os.path.exists(path):
            os.remove(path)
    return None, None

@app.route("/upload/video", methods=["POST"])
@role_required(*OPERATOR_ROLES)
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    try:
        upload_path, stem, ext = save_upload(file, config.ALLOWED_VIDEO_EXTENSIONS)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    cap = cv2.VideoCapture(upload_path)
    out_writer = None
    try:
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Could not open video file."}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if width <= 0 or height <= 0:
            return jsonify({"success": False, "message": "Video has no readable frames."}), 400

        out_writer, out_path = open_video_writer(os.path.join(config.OUTPUTS_DIR, f"annotated_{stem}"), fps, (width, height))
        if out_writer is None:
            return jsonify({"success": False, "message": "No video encoder available on the server."}), 500

        processed_frames = 0
        total_workers_seen = set()
        total_violations_recorded = 0
        sample_workers = []

        # Fresh tracking state per video, sharing the already-loaded model
        vid_pipeline = SafetyPipeline(detector=pipeline.detector)

        while processed_frames < config.MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, summary = vid_pipeline.process_frame(frame, source="Uploaded Video")
            out_writer.write(annotated_frame)
            processed_frames += 1

            for w in summary.get("workers", []):
                total_workers_seen.add(w["worker_id"])
            total_violations_recorded += len(summary.get("new_incidents", []))
            if summary.get("workers") and not sample_workers:
                sample_workers = summary["workers"]

        audit("VIDEO_ANALYZED", "DETECTION",
              f"Video {file.filename!r}: {processed_frames} frames, {total_violations_recorded} violations")
        return jsonify({
            "success": True,
            "output_video_url": f"/outputs/{os.path.basename(out_path)}",
            "stats": {
                "frames_processed": processed_frames,
                "frame_limit": config.MAX_VIDEO_FRAMES,
                "total_workers": len(total_workers_seen),
                "total_violations": total_violations_recorded,
                "sample_workers": sample_workers
            }
        })
    except Exception as e:
        print(f"[-] Video processing error: {e}")
        return jsonify({"success": False, "message": "Video processing error."}), 500
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()

# ==============================================================================
# Zone Management Endpoints
# ==============================================================================

@app.route("/api/zones", methods=["GET"])
def get_zones_api():
    zones = ZoneModel.get_all()
    return jsonify(zones)

def validate_zone_coordinates(coords):
    if not isinstance(coords, list) or len(coords) < 3 or len(coords) > 100:
        return None
    cleaned = []
    for pt in coords:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
            return None
        cleaned.append([x, y])
    return cleaned

@app.route("/zones", methods=["POST"])
@role_required(*ZONE_EDITOR_ROLES)
def create_zone():
    data = request.get_json(silent=True) or {}
    coords = validate_zone_coordinates(data.get("coordinates"))
    if coords is None:
        return jsonify({"success": False, "message": "Invalid zone coordinates (need at least 3 [x, y] points)."}), 400

    name = clean_text(data.get("name"), 100) or "Zone"
    zone_type = str(data.get("zone_type", "RESTRICTED")).upper()
    if zone_type not in ZONE_RULES:
        return jsonify({"success": False, "message": f"Unknown zone type {zone_type}."}), 400
    try:
        site_id = optional_int(data.get("site_id"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Invalid site."}), 400

    try:
        zone_id = ZoneModel.create(name=name, zone_type=zone_type, coordinates=coords, site_id=site_id)
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Site does not exist."}), 400
    for p in LIVE_PIPELINES:
        p.reload_zones()
    audit("ZONE_CREATED", "SPATIAL_RULES", f"Defined {zone_type} zone: {name}", str(zone_id))
    return jsonify({"success": True, "zone_id": zone_id})

@app.route("/zones/<int:zone_id>", methods=["DELETE"])
@role_required(*ZONE_EDITOR_ROLES)
def delete_zone(zone_id):
    deleted = ZoneModel.delete(zone_id)
    for p in LIVE_PIPELINES:
        p.reload_zones()
    if deleted:
        audit("ZONE_DELETED", "SPATIAL_RULES", f"Deleted zone #{zone_id}", str(zone_id))
    return jsonify({"success": deleted}), (200 if deleted else 404)

# ==============================================================================
# Settings Endpoints
# ==============================================================================

@app.route("/settings", methods=["POST"])
@role_required(*ADMIN_ROLES)
def update_settings_route():
    defaults = default_settings()
    raw = {
        "conf_thresh": request.form.get("conf_thresh", defaults["conf_thresh"]),
        "iou_thresh": request.form.get("iou_thresh", defaults["iou_thresh"]),
        "min_violation_frames": request.form.get("min_violation_frames", defaults["min_violation_frames"]),
        "min_zone_frames": request.form.get("min_zone_frames", defaults["min_zone_frames"]),
    }
    try:
        values = SafetyPipeline.validate_settings(
            conf=raw["conf_thresh"], iou=raw["iou_thresh"],
            min_v_frames=raw["min_violation_frames"], min_z_frames=raw["min_zone_frames"]
        )
    except ValueError as e:
        flash(f"Settings not saved: {e}", "error")
        return redirect(url_for("settings_view"))

    SettingsModel.set("conf_thresh", values["conf"])
    SettingsModel.set("iou_thresh", values["iou"])
    SettingsModel.set("min_violation_frames", values["min_v_frames"])
    SettingsModel.set("min_zone_frames", values["min_z_frames"])

    for p in LIVE_PIPELINES:
        p.update_settings(conf=values["conf"], iou=values["iou"],
                          min_v_frames=values["min_v_frames"], min_z_frames=values["min_z_frames"])

    audit("SETTINGS_UPDATED", "SYSTEM_CONFIG",
          f"Conf: {values['conf']}, IoU: {values['iou']}, MinVFrames: {values['min_v_frames']}, MinZFrames: {values['min_z_frames']}")
    flash("Settings saved.", "success")
    return redirect(url_for("settings_view"))

# ==============================================================================
# Data Export & Static File Delivery
# ==============================================================================

@app.route("/api/incidents/export")
def export_incidents_csv():
    incidents = IncidentModel.get_all(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Incident Code", "Worker Label", "Violation Type", "Severity", "Confidence", "Site", "Timestamp", "Source", "Status", "Assigned Officer", "Resolution Note"])

    for inc in incidents:
        writer.writerow([csv_safe(v) for v in (
            inc["id"],
            inc["incident_code"],
            inc["worker_label"],
            inc["violation_type"],
            inc["severity"],
            f"{inc['confidence']:.2f}",
            inc.get("site_name") or "Default Site",
            inc["timestamp"],
            inc["source"],
            inc.get("status") or "OPEN",
            inc.get("assigned_to") or "",
            inc.get("resolution_note") or ""
        )])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=siteguard_safety_incidents_{datetime.datetime.now().strftime('%Y%m%d')}.csv"}
    )

@app.route("/api/violations/export")
@app.route("/export_violations_csv")
def export_violations_csv():
    violations = ViolationModel.get_all(limit=5000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Violation Code", "Worker Label", "Violation Type", "Description", "Severity", "Confidence", "Site", "Timestamp", "Status", "Review Decision", "Reviewed By", "Reviewed At"])

    for v in violations:
        writer.writerow([csv_safe(x) for x in (
            v["id"],
            v["violation_code"],
            v["worker_label"],
            v["violation_type"],
            v.get("description") or "",
            v["severity"],
            f"{v['confidence']:.2f}",
            v.get("site_name") or "Construction Site",
            v["timestamp"],
            v.get("status") or "OPEN",
            v.get("review_decision") or "PENDING",
            v.get("reviewed_by") or "",
            v.get("reviewed_at") or ""
        )])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=siteguard_violations_{datetime.datetime.now().strftime('%Y%m%d')}.csv"}
    )

@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(config.SNAPSHOTS_DIR, filename)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUTS_DIR, filename)

@app.errorhandler(413)
def too_large(_e):
    return error_response("File is too large (limit 250 MB).", 413)

if __name__ == "__main__":
    print(f"[SiteGuard AI] Starting Construction Safety Monitoring App on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
