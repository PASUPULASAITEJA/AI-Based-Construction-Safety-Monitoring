"""
SiteGuard AI — Intelligent Construction Safety Management Platform
Main Web Application, RESTful APIs, Role-Based Access, & Real-Time AI Inference Server
"""
import os
import io
import time
import csv
import json
import threading
import datetime
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for, session

import config
from cv.pipeline import SafetyPipeline
from database.db import init_db
from database.models import (
    UserModel, SiteModel, CameraModel, ZoneModel, ViolationModel, IncidentModel, AlertModel,
    WorkerTrackModel, NotificationModel, AuditLogModel, SettingsModel
)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max upload

# Initialize Database Schema & Migrations
init_db()

# Initialize Master Safety Computer Vision Pipeline
pipeline = SafetyPipeline()

# Global Context Processor for Navigation, Telemetry & Authentication across All Views
from safety.ppe_config import get_ppe_capabilities

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
        "current_user": session.get("username", "Operator"),
        "current_role": session.get("role", "ADMIN"),
        "is_authenticated": "user_id" in session,
        "ppe_capabilities": get_ppe_capabilities()
    }

# ==============================================================================
# Camera & High-Performance Background Stream Engine (Zero Fake Fallback)
# ==============================================================================

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

    def _open_hardware_camera(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                return True
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return True
        except Exception as e:
            print(f"[-] Hardware camera open error: {e}")
        return False

    def start(self):
        with self.lock:
            self.is_running = True
            return self._open_hardware_camera()

    def stop(self):
        with self.lock:
            self.is_running = False
            self._release_cap()
            return True

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
latest_frame_summary = {
    "fps": 0.0,
    "worker_count": 0,
    "compliant_count": 0,
    "violation_count": 0,
    "critical_count": 0,
    "workers": [],
    "is_live": False
}

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
                    # Last viewer left: free the webcam and clear stale live telemetry
                    camera.release_hardware()
                    latest_frame_summary = empty_frame_summary()
                    idle = True
                time.sleep(0.2)
                continue
            idle = False

            try:
                frame, is_live = camera.get_raw_frame()
                if frame is not None:
                    if is_live:
                        annotated_frame, summary = pipeline.process_frame(frame, source="Live Stream")
                        crit_count = sum(1 for w in summary.get("workers", []) if "CRITICAL" in w.get("status", ""))
                        summary["critical_count"] = crit_count
                        summary["is_live"] = True
                        latest_frame_summary = summary
                    else:
                        annotated_frame = frame
                        latest_frame_summary = {
                            "fps": 0.0,
                            "worker_count": 0,
                            "compliant_count": 0,
                            "violation_count": 0,
                            "critical_count": 0,
                            "workers": [],
                            "is_live": False
                        }

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
    """High-throughput multi-client stream distribution."""
    while True:
        frame_bytes = stream_engine.get_latest_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

# ==============================================================================
# Authentication & Role-Based Access Control (RBAC)
# ==============================================================================

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
# Primary View Routes
# ==============================================================================

@app.route("/")
def dashboard_view():
    recent = IncidentModel.get_recent(limit=6)
    today_count = IncidentModel.get_count(today_only=True)
    open_alerts = AlertModel.get_open_count()
    resolved_count = len(IncidentModel.get_all(status="RESOLVED", limit=1000))
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
        current_v_type=v_type,
        current_sev=sev,
        current_status=status,
        current_site=site_id,
        current_search=search
    )

@app.route("/api/violations/<int:v_id>/confirm", methods=["POST"])
def confirm_violation_api(v_id):
    data = request.get_json() or {}
    reviewer = data.get("reviewer") or session.get("username", "Safety Officer")
    notes = data.get("notes", "")

    inc_code = ViolationModel.confirm_violation(violation_id=v_id, reviewer=reviewer, notes=notes)
    if inc_code:
        AuditLogModel.log(
            user_role=session.get("role", "SAFETY_OFFICER"),
            action="VIOLATION_CONFIRMED",
            category="HUMAN_REVIEW",
            details=f"Violation #{v_id} confirmed by {reviewer}. Escalated to {inc_code}.",
            target_id=str(v_id)
        )
        return jsonify({"success": True, "incident_code": inc_code})
    return jsonify({"success": False, "message": "Violation not found."}), 404

@app.route("/api/violations/<int:v_id>/false-positive", methods=["POST"])
def mark_false_positive_api(v_id):
    data = request.get_json() or {}
    reviewer = data.get("reviewer") or session.get("username", "Safety Officer")
    reason = data.get("reason", "Model Misclassification")

    ViolationModel.mark_false_positive(violation_id=v_id, reviewer=reviewer, reason=reason)
    AuditLogModel.log(
        user_role=session.get("role", "SAFETY_OFFICER"),
        action="FALSE_POSITIVE_FLAGGED",
        category="HUMAN_REVIEW",
        details=f"Violation #{v_id} marked as False Positive by {reviewer}. Reason: {reason}",
        target_id=str(v_id)
    )
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

@app.route("/api/alerts/<int:alert_id>/action", methods=["POST"])
def update_alert_action(alert_id):
    data = request.get_json() or {}
    new_status = data.get("status", "ACKNOWLEDGED")
    assigned_to = data.get("assigned_to", "Safety Officer")
    resolution_note = data.get("resolution_note")

    AlertModel.update_lifecycle(
        alert_id=alert_id,
        status=new_status,
        assigned_to=assigned_to,
        resolution_note=resolution_note
    )
    AuditLogModel.log(
        user_role=session.get("role", "SAFETY_OFFICER"),
        action=f"ALERT_{new_status}",
        category="ALERT_MANAGEMENT",
        details=f"Alert #{alert_id} moved to status {new_status}. Note: {resolution_note or 'None'}",
        target_id=f"ALT-{alert_id}"
    )
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
        current_v_type=v_type,
        current_sev=sev,
        current_status=status,
        current_search=search
    )

@app.route("/api/incidents/<int:inc_id>/update", methods=["POST"])
def update_incident_details(inc_id):
    data = request.get_json() or {}
    status = data.get("status", "RESOLVED")
    assigned_to = data.get("assigned_to")
    root_cause = data.get("root_cause")
    corrective_action = data.get("corrective_action")
    resolution_note = data.get("resolution_note")

    IncidentModel.update_resolution(
        incident_id=inc_id,
        status=status,
        assigned_to=assigned_to,
        root_cause=root_cause,
        corrective_action=corrective_action,
        resolution_note=resolution_note
    )
    AuditLogModel.log(
        user_role=session.get("role", "SAFETY_OFFICER"),
        action="INCIDENT_RESOLVED",
        category="INCIDENT_AUDIT",
        details=f"Incident #{inc_id} resolved by {assigned_to}. Root cause: {root_cause}",
        target_id=f"INC-{inc_id}"
    )
    return jsonify({"success": True})

@app.route("/sites")
def sites_view():
    sites = SiteModel.get_all()
    zones = ZoneModel.get_all()
    return render_template("sites.html", active_page="sites", sites=sites, zones=zones)

@app.route("/api/sites", methods=["POST"])
def create_site():
    data = request.get_json() or {}
    site_code = data.get("site_code", f"SITE-{int(time.time())}")
    name = data.get("name", "Site")
    location = data.get("location", "")
    project_name = data.get("project_name", "")
    safety_officer = data.get("safety_officer", "")

    site_id = SiteModel.create(site_code, name, location, project_name, safety_officer)
    AuditLogModel.log(session.get("role", "ADMIN"), "SITE_CREATED", "INFRASTRUCTURE", f"Created Site {name} ({site_code})", str(site_id))
    return jsonify({"success": True, "site_id": site_id})

@app.route("/api/sites/<int:site_id>", methods=["PUT"])
def update_site(site_id):
    data = request.get_json() or {}
    SiteModel.update(
        site_id=site_id,
        name=data.get("name", ""),
        location=data.get("location", ""),
        project_name=data.get("project_name", ""),
        status=data.get("status", "ACTIVE"),
        safety_officer=data.get("safety_officer", "")
    )
    return jsonify({"success": True})

@app.route("/api/sites/<int:site_id>", methods=["DELETE"])
def delete_site(site_id):
    deleted = SiteModel.delete(site_id)
    return jsonify({"success": deleted})

@app.route("/cameras")
def cameras_view():
    cameras = CameraModel.get_all()
    sites = SiteModel.get_all()
    return render_template("cameras.html", active_page="cameras", cameras=cameras, sites=sites)

@app.route("/api/cameras", methods=["POST"])
def create_camera():
    data = request.get_json() or {}
    cam_code = data.get("camera_code", f"CAM-{int(time.time())}")
    name = data.get("name", "Camera")
    site_id = data.get("site_id")
    zone_id = data.get("zone_id")
    source = data.get("stream_source", "0")
    stream_type = data.get("stream_type", "webcam")

    cam_id = CameraModel.create(cam_code, name, site_id, zone_id, source, stream_type)
    AuditLogModel.log(session.get("role", "ADMIN"), "CAMERA_CREATED", "INFRASTRUCTURE", f"Added Camera {name} ({cam_code})", str(cam_id))
    return jsonify({"success": True, "camera_id": cam_id})

@app.route("/api/cameras/<int:cam_id>", methods=["DELETE"])
def delete_camera(cam_id):
    deleted = CameraModel.delete(cam_id)
    return jsonify({"success": deleted})

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
    model_info = {
        "model_path": pipeline.detector.model_path,
        "device": config.DEVICE.upper(),
        "img_size": config.IMG_SIZE,
        "conf": pipeline.detector.conf,
        "iou": pipeline.detector.iou,
        "classes": pipeline.detector.classes
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
        camera_mode="Hardware Webcam",
        snapshots_count=snapshots_count,
        unread_count=unread,
        now_str=now_str
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
    success = camera.start()
    return jsonify({"success": success, "message": "Camera started successfully." if success else "Failed to open camera."})

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
    import base64
    global latest_frame_summary
    data = request.get_json() or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "No image data"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Failed to decode frame"}), 400

        annotated_frame, summary = pipeline.process_frame(frame, source="Browser Camera")
        crit_count = sum(1 for w in summary.get("workers", []) if "CRITICAL" in w.get("status", ""))
        summary["critical_count"] = crit_count
        latest_frame_summary = summary

        ret, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{out_b64}",
            "summary": summary
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==============================================================================
# Media Upload Endpoints (Real YOLOv8 Inference)
# ==============================================================================

@app.route("/upload/image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.basename(file.filename).replace(" ", "_")
    filename = f"upload_{timestamp}_{clean_name}"
    upload_path = os.path.join(config.UPLOADS_DIR, filename)
    file.save(upload_path)

    try:
        annotated_img, stats = pipeline.process_single_image(upload_path, source="Image Inspection")

        out_filename = f"annotated_{timestamp}_{clean_name}"
        out_path = os.path.join(config.OUTPUTS_DIR, out_filename)
        cv2.imwrite(out_path, annotated_img)

        return jsonify({
            "success": True,
            "annotated_url": f"/outputs/{out_filename}",
            "stats": stats
        })
    except Exception as e:
        print(f"[-] Image processing error: {e}")
        return jsonify({"success": False, "message": f"Inference error: {str(e)}"}), 500

@app.route("/upload/video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.basename(file.filename).replace(" ", "_")
    upload_filename = f"upload_{timestamp}_{clean_name}"
    upload_path = os.path.join(config.UPLOADS_DIR, upload_filename)
    file.save(upload_path)

    out_filename = f"annotated_{timestamp}.mp4"
    out_path = os.path.join(config.OUTPUTS_DIR, out_filename)

    try:
        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Could not open video file."}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        processed_frames = 0
        total_workers_seen = set()
        total_violations_recorded = 0
        sample_workers = []

        vid_pipeline = SafetyPipeline()

        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= 600:
                break

            annotated_frame, summary = vid_pipeline.process_frame(frame, source="Uploaded Video")
            out_writer.write(annotated_frame)
            processed_frames += 1

            for w in summary.get("workers", []):
                total_workers_seen.add(w["worker_id"])
            if summary.get("new_incidents"):
                total_violations_recorded += len(summary["new_incidents"])
            if summary.get("workers") and not sample_workers:
                sample_workers = summary["workers"]

        cap.release()
        out_writer.release()

        return jsonify({
            "success": True,
            "output_video_url": f"/outputs/{out_filename}",
            "stats": {
                "frames_processed": processed_frames,
                "total_workers": len(total_workers_seen),
                "total_violations": total_violations_recorded,
                "sample_workers": sample_workers
            }
        })
    except Exception as e:
        print(f"[-] Video processing error: {e}")
        return jsonify({"success": False, "message": f"Video processing error: {str(e)}"}), 500

# ==============================================================================
# Zone Management Endpoints
# ==============================================================================

@app.route("/api/zones", methods=["GET"])
def get_zones_api():
    zones = ZoneModel.get_all()
    return jsonify(zones)

@app.route("/zones", methods=["POST"])
def create_zone():
    data = request.get_json()
    if not data or "coordinates" not in data:
        return jsonify({"success": False, "message": "Invalid zone coordinates."}), 400

    name = data.get("name", "Zone")
    zone_type = str(data.get("zone_type", "RESTRICTED")).upper()
    coords = data.get("coordinates", [])
    site_id = data.get("site_id")

    zone_id = ZoneModel.create(name=name, zone_type=zone_type, coordinates=coords, site_id=site_id)
    pipeline.reload_zones()
    AuditLogModel.log(session.get("role", "SAFETY_OFFICER"), "ZONE_CREATED", "SPATIAL_RULES", f"Defined {zone_type} zone: {name}", str(zone_id))
    return jsonify({"success": True, "zone_id": zone_id})

@app.route("/zones/<int:zone_id>", methods=["DELETE"])
def delete_zone(zone_id):
    deleted = ZoneModel.delete(zone_id)
    pipeline.reload_zones()
    return jsonify({"success": deleted})

# ==============================================================================
# Settings Endpoints
# ==============================================================================

@app.route("/settings", methods=["POST"])
def update_settings_route():
    conf_thresh = request.form.get("conf_thresh", "0.02")
    iou_thresh = request.form.get("iou_thresh", "0.45")
    min_violation_frames = request.form.get("min_violation_frames", "10")
    min_zone_frames = request.form.get("min_zone_frames", "8")

    SettingsModel.set("conf_thresh", conf_thresh)
    SettingsModel.set("iou_thresh", iou_thresh)
    SettingsModel.set("min_violation_frames", min_violation_frames)
    SettingsModel.set("min_zone_frames", min_zone_frames)

    pipeline.update_settings(
        conf=conf_thresh,
        iou=iou_thresh,
        min_v_frames=min_violation_frames,
        min_z_frames=min_zone_frames
    )

    AuditLogModel.log(
        session.get("role", "ADMIN"), "SETTINGS_UPDATED", "SYSTEM_CONFIG",
        f"Conf: {conf_thresh}, IoU: {iou_thresh}, MinVFrames: {min_violation_frames}, MinZFrames: {min_zone_frames}"
    )

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
        writer.writerow([
            inc["id"],
            inc["incident_code"],
            inc["worker_label"],
            inc["violation_type"],
            inc["severity"],
            f"{inc['confidence']:.2f}",
            inc.get("site_name", "Default Site"),
            inc["timestamp"],
            inc["source"],
            inc.get("status", "CONFIRMED"),
            inc.get("assigned_to", ""),
            inc.get("resolution_note", "")
        ])

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
        writer.writerow([
            v["id"],
            v["violation_code"],
            v["worker_label"],
            v["violation_type"],
            v.get("description", ""),
            v["severity"],
            f"{v['confidence']:.2f}",
            v.get("site_name", "Construction Site"),
            v["timestamp"],
            v.get("status", "OPEN"),
            v.get("review_decision", "PENDING"),
            v.get("reviewed_by", ""),
            v.get("reviewed_at", "")
        ])

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

if __name__ == "__main__":
    print(f"[SiteGuard AI] Starting Construction Safety Monitoring App on http://127.0.0.1:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
