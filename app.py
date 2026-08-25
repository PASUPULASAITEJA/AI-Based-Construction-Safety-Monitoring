"""
AI-Based Construction Safety Monitoring
Main Flask Web Application & Video Streaming Server
"""
import os
import io
import time
import csv
import threading
import datetime
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for

import config
from cv.pipeline import SafetyPipeline
from database.db import init_db
from database.models import IncidentModel, ZoneModel, SettingsModel

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max upload

# Initialize Database Schema
init_db()

# Initialize Master Safety Pipeline
pipeline = SafetyPipeline()

# ==============================================================================
# Camera & High-Performance Background Stream Engine
# ==============================================================================

class VideoCamera:
    def __init__(self):
        self.cap = None
        self.is_running = True
        self.lock = threading.Lock()
        self.mode = "webcam"  # 'webcam' or 'simulator'
        self.test_images = []
        self.test_img_idx = 0
        self._load_fallback_samples()

    def _load_fallback_samples(self):
        test_dir = os.path.join(config.DATASET_DIR, "images", "test")
        if os.path.exists(test_dir):
            imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.test_images = sorted(imgs)

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            if mode == "simulator":
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            elif mode == "webcam":
                self._open_hardware_camera()

    def _open_hardware_camera(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                return True
            # DirectShow on Windows for instant capture
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
            if self.mode == "webcam":
                if not self._open_hardware_camera():
                    print("[Camera] Hardware webcam not accessible, falling back to simulator.")
                    self.mode = "simulator"
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            return True

    def get_raw_frame(self):
        with self.lock:
            if not self.is_running:
                idle = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(idle, "CAMERA FEED PAUSED", (170, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 130, 140), 2, cv2.LINE_AA)
                cv2.putText(idle, "Click 'Start Camera' to resume", (165, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 100), 1, cv2.LINE_AA)
                return idle

            if self.mode == "simulator" and self.test_images:
                img_path = self.test_images[self.test_img_idx % len(self.test_images)]
                frame = cv2.imread(img_path)
                time.sleep(0.06)
                self.test_img_idx += 1
                if frame is not None:
                    return frame

            # Hardware webcam read
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame

            # Retry open
            if self.mode == "webcam":
                if self._open_hardware_camera():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        return frame

            if self.test_images:
                img_path = self.test_images[self.test_img_idx % len(self.test_images)]
                frame = cv2.imread(img_path)
                if frame is not None:
                    return frame

            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback, "Searching Video Signal...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return fallback

camera = VideoCamera()

# Real-time state cache
latest_frame_summary = {
    "fps": 0.0,
    "worker_count": 0,
    "compliant_count": 0,
    "violation_count": 0,
    "critical_count": 0,
    "workers": []
}

class StreamEngine:
    def __init__(self):
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        global latest_frame_summary
        while True:
            try:
                frame = camera.get_raw_frame()
                if frame is not None:
                    annotated_frame, summary = pipeline.process_frame(frame, source="Live Stream")
                    crit_count = sum(1 for w in summary.get("workers", []) if "CRITICAL" in w.get("status", ""))
                    summary["critical_count"] = crit_count
                    latest_frame_summary = summary

                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        with self.lock:
                            self.latest_jpeg = buffer.tobytes()
            except Exception as e:
                time.sleep(0.1)
            time.sleep(0.03)

    def get_latest_frame_bytes(self):
        with self.lock:
            return self.latest_jpeg

stream_engine = StreamEngine()

def generate_video_stream():
    """Ultra-fast zero-cost streaming generator for multi-client distribution."""
    while True:
        frame_bytes = stream_engine.get_latest_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

# ==============================================================================
# Page Routes
# ==============================================================================

@app.route("/")
def dashboard_view():
    recent = IncidentModel.get_recent(limit=8)
    summary_stats = {
        "workers_count": latest_frame_summary.get("worker_count", 0),
        "violations_count": latest_frame_summary.get("violation_count", 0),
        "critical_count": latest_frame_summary.get("critical_count", 0),
        "compliance_rate": 100 if latest_frame_summary.get("worker_count", 0) == 0 else int((latest_frame_summary.get("compliant_count", 0) / max(1, latest_frame_summary.get("worker_count", 1))) * 100)
    }
    return render_template("index.html", active_page="dashboard", recent_alerts=recent, stats=summary_stats)

@app.route("/monitor")
def monitor_view():
    return render_template("monitor.html", active_page="monitor")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera/mode", methods=["POST"])
def set_camera_mode():
    data = request.get_json() or {}
    mode = data.get("mode", "webcam")
    camera.set_mode(mode)
    return jsonify({"success": True, "mode": camera.mode})

@app.route("/api/process_browser_frame", methods=["POST"])
def process_browser_frame():
    """Receives a base64 frame from browser webcam, processes it through CV pipeline, and returns annotated frame + telemetry."""
    import base64
    global latest_frame_summary
    data = request.get_json() or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "No image data"}), 400

    try:
        # Strip header
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

        # Encode back to base64
        ret, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{out_b64}",
            "summary": summary
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/upload")
def upload_view():
    return render_template("upload.html", active_page="upload")

@app.route("/incidents")
def incidents_view():
    incidents = IncidentModel.get_all(limit=200)
    return render_template("incidents.html", active_page="incidents", incidents=incidents)

@app.route("/analytics")
def analytics_view():
    summary = IncidentModel.get_analytics_summary()
    return render_template("analytics.html", active_page="analytics", summary=summary)

@app.route("/settings")
def settings_view():
    settings = SettingsModel.get_all()
    device_name = f"PyTorch on {config.DEVICE.upper()}"
    return render_template("settings.html", active_page="settings", settings=settings, device_name=device_name)

# ==============================================================================
# Camera & Video API Endpoints
# ==============================================================================

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

# ==============================================================================
# Media Upload Endpoints
# ==============================================================================

@app.route("/upload/image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    # Save temporary upload
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{file.filename}"
    upload_path = os.path.join(config.UPLOADS_DIR, filename)
    file.save(upload_path)

    try:
        # Process single image
        annotated_img, stats = pipeline.process_single_image(upload_path, source="Image Upload")

        # Save annotated output
        out_filename = f"annotated_{timestamp}_{file.filename}"
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
    upload_filename = f"upload_{timestamp}_{file.filename}"
    upload_path = os.path.join(config.UPLOADS_DIR, upload_filename)
    file.save(upload_path)

    out_filename = f"annotated_{timestamp}.mp4"
    out_path = os.path.join(config.OUTPUTS_DIR, out_filename)

    try:
        # Sequential frame-by-frame video processing
        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Could not open video file."}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Use mp4v or avc1 codec for browser playback compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        processed_frames = 0
        total_workers_seen = set()
        total_violations_recorded = 0
        sample_workers = []

        # Fresh pipeline instance for video tracking session
        vid_pipeline = SafetyPipeline()

        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= 600: # Limit processing to 600 frames for responsiveness
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
    zone_type = data.get("zone_type", "RESTRICTED")
    coords = data.get("coordinates", [])

    zone_id = ZoneModel.create(name=name, zone_type=zone_type, coordinates=coords)
    pipeline.reload_zones()
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
    conf_thresh = request.form.get("conf_thresh", "0.35")
    iou_thresh = request.form.get("iou_thresh", "0.45")
    min_violation_frames = request.form.get("min_violation_frames", "10")
    min_zone_frames = request.form.get("min_zone_frames", "8")

    SettingsModel.set("conf_thresh", conf_thresh)
    SettingsModel.set("iou_thresh", iou_thresh)
    SettingsModel.set("min_violation_frames", min_violation_frames)
    SettingsModel.set("min_zone_frames", min_zone_frames)

    pipeline.update_settings(
        conf=float(conf_thresh),
        iou=float(iou_thresh),
        min_v_frames=int(min_violation_frames),
        min_z_frames=int(min_zone_frames)
    )

    return redirect(url_for("settings_view"))

# ==============================================================================
# Data Export & Static File Delivery
# ==============================================================================

@app.route("/api/incidents/export")
def export_incidents_csv():
    incidents = IncidentModel.get_all(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Incident Code", "Worker Label", "Violation Type", "Severity", "Confidence", "Timestamp", "Source"])

    for inc in incidents:
        writer.writerow([
            inc["id"],
            inc["incident_code"],
            inc["worker_label"],
            inc["violation_type"],
            inc["severity"],
            f"{inc['confidence']:.2f}",
            inc["timestamp"],
            inc["source"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=construction_safety_incidents_{datetime.datetime.now().strftime('%Y%m%d')}.csv"}
    )

@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(config.SNAPSHOTS_DIR, filename)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUTS_DIR, filename)

if __name__ == "__main__":
    print(f"[Server] Starting Construction Safety Monitoring App on http://127.0.0.1:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
