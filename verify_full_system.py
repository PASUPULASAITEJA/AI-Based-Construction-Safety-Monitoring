"""
SiteGuard AI / SafeGear — Full System Verification Suite
Validates:
  1. System & Dependencies (Python, PyTorch, Ultralytics YOLOv8, OpenCV, Flask)
  2. AI Model Weights & 8-Category PPE Support
  3. SQLite Database Schema & Tables
  4. End-to-End CV Pipeline (Detector -> ByteTrack -> Spatial Association -> Zone Engine -> Annotator)
  5. Web Application Endpoints & REST APIs
"""

import sys
import os
import time
import importlib.metadata
import numpy as np
import config

# Color helpers for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed_tests = 0
failed_tests = 0

def log_header(title):
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")

def report_result(name, success, details=""):
    global passed_tests, failed_tests
    if success:
        passed_tests += 1
        msg = f"[{GREEN}PASS{RESET}] {name}"
        if details:
            msg += f" ({details})"
        print(msg)
    else:
        failed_tests += 1
        msg = f"[{RED}FAIL{RESET}] {name}"
        if details:
            msg += f" — Error: {details}"
        print(msg)

def test_dependencies():
    log_header("1. Environment & Package Dependencies")
    
    # Python
    py_ver = sys.version.split()[0]
    report_result("Python Runtime", sys.version_info >= (3, 8), f"v{py_ver}")

    # PyTorch
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        report_result("PyTorch Library", True, f"v{torch.__version__} [{device}]")
    except Exception as e:
        report_result("PyTorch Library", False, str(e))

    # Ultralytics
    try:
        import ultralytics
        report_result("Ultralytics YOLOv8", True, f"v{ultralytics.__version__}")
    except Exception as e:
        report_result("Ultralytics YOLOv8", False, str(e))

    # OpenCV
    try:
        import cv2
        report_result("OpenCV Computer Vision", True, f"v{cv2.__version__}")
    except Exception as e:
        report_result("OpenCV Computer Vision", False, str(e))

    # Flask
    try:
        flask_ver = importlib.metadata.version("flask")
        report_result("Flask Web Engine", True, f"v{flask_ver}")
    except Exception as e:
        report_result("Flask Web Engine", False, str(e))

def test_model_and_ppe():
    log_header("2. AI Model Weights & 8-Category PPE Matrix")
    
    from safety.ppe_config import PPE_CATEGORIES, get_ppe_capabilities

    model_path = config.MODEL_PATH
    file_exists = os.path.isfile(model_path)
    report_result("Model Weights File Exists", file_exists, model_path)

    if not file_exists:
        return

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        class_names = model.names
        report_result("YOLOv8 Weights Loadable", True, f"{len(class_names)} classes detected")

        # Verify class schema extraction
        has_person = any("person" in name.lower() for name in class_names.values())
        report_result("Worker Anchor Detection Class ('person')", has_person)

        # 8 PPE Matrix
        caps = get_ppe_capabilities()
        supported = [c["name"] for c in caps if c.get("is_supported")]
        unsupported = [c["name"] for c in caps if not c.get("is_supported")]
        
        report_result(
            "PPE Capability Evaluation",
            len(caps) == 8,
            f"{len(supported)}/8 supported ({', '.join(supported)})"
        )
        if unsupported:
            print(f"  {YELLOW}ℹ Standby categories (awaiting training expansion): {', '.join(unsupported)}{RESET}")
            
    except Exception as e:
        report_result("YOLOv8 Weights Loadable", False, str(e))

def test_database():
    log_header("3. Database Schema & Models Integrity")
    
    try:
        import database.db as db
        from database.models import (
            UserModel, SiteModel, CameraModel, ZoneModel,
            IncidentModel, AlertModel, WorkerTrackModel,
            NotificationModel, AuditLogModel, SettingsModel
        )

        db.init_db()
        report_result("SQLite Database Connection & Initialization", True, config.DB_PATH)

        # Check tables existence via direct SQLite pragma
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = [
            "users", "sites", "cameras", "zones", "workers_track",
            "violations", "alerts", "incidents", "audit_logs", "settings"
        ]
        
        missing = [t for t in expected_tables if t not in tables]
        report_result("Core Database Tables", len(missing) == 0, f"Found {len(tables)} tables; Missing: {missing or 'None'}")

        # Test model methods
        users_count = len(UserModel.get_all())
        sites_count = len(SiteModel.get_all())
        cams_count = len(CameraModel.get_all())
        zones_count = len(ZoneModel.get_all())
        report_result(
            "Database Models Operational",
            True,
            f"Users: {users_count}, Sites: {sites_count}, Cameras: {cams_count}, Zones: {zones_count}"
        )

    except Exception as e:
        report_result("Database Integrity", False, str(e))

def test_cv_pipeline():
    log_header("4. End-to-End Computer Vision Pipeline")

    try:
        from cv.pipeline import SafetyPipeline
        
        t0 = time.time()
        pipeline = SafetyPipeline()
        report_result("SafetyPipeline Instantiation", True, f"initialized in {(time.time()-t0)*1000:.1f}ms")

        # Feed a dummy 640x480 frame through full pipeline
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic rectangle so frame is non-trivial
        test_frame[100:300, 200:400] = [100, 150, 200]

        annotated_frame, summary = pipeline.process_frame(test_frame, source="VerificationTest")
        
        valid_summary = (
            "fps" in summary and
            "worker_count" in summary and
            "workers" in summary
        )
        report_result(
            "Full Frame Processing Cycle",
            valid_summary and annotated_frame.shape == test_frame.shape,
            f"Output shape: {annotated_frame.shape}, FPS: {summary['fps']:.1f}"
        )

    except Exception as e:
        report_result("Computer Vision Pipeline", False, str(e))

def test_flask_routes():
    log_header("5. Web Application Endpoints & REST APIs")

    try:
        from app import app
        client = app.test_client()

        # HTML View Routes (must return HTTP 200)
        view_routes = [
            ("/", "Dashboard"),
            ("/monitor", "Live Monitoring"),
            ("/violations", "Safety Violations"),
            ("/reports", "Reports & Audits"),
            ("/cameras", "Camera Management"),
            ("/alerts", "Active Alerts"),
            ("/incidents", "Incident Tracking"),
            ("/analytics", "Safety Analytics"),
            ("/ppe", "PPE Requirements"),
            ("/model", "AI Model Info"),
            ("/settings", "System Settings"),
        ]

        all_views_ok = True
        failed_views = []
        for path, name in view_routes:
            res = client.get(path)
            if res.status_code != 200:
                all_views_ok = False
                failed_views.append(f"{path} ({res.status_code})")
        
        report_result(
            "HTML Page Views (HTTP 200)",
            all_views_ok,
            f"{len(view_routes)} routes tested; Failed: {failed_views or 'None'}"
        )

        # Authentication bypass check (login/logout redirect without prompt)
        res_login = client.get("/login")
        res_logout = client.get("/logout")
        auth_bypass_ok = (res_login.status_code == 302 and res_logout.status_code == 302)
        report_result("Auth Bypass Redirects (/login, /logout)", auth_bypass_ok, "302 Redirect to Dashboard")

        # REST API Routes
        api_routes = [
            ("/api/status", 200),
            ("/api/zones", 200),
            ("/api/incidents/export", 200),
            ("/api/violations/export", 200),
        ]

        all_apis_ok = True
        failed_apis = []
        for path, expected_status in api_routes:
            res = client.get(path)
            if res.status_code != expected_status:
                all_apis_ok = False
                failed_apis.append(f"{path} ({res.status_code})")

        report_result(
            "REST APIs Response",
            all_apis_ok,
            f"{len(api_routes)} endpoints tested; Failed: {failed_apis or 'None'}"
        )

    except Exception as e:
        report_result("Flask Web Engine", False, str(e))

def main():
    print(f"\n{BOLD}=================================================================={RESET}")
    print(f"{BOLD}{CYAN}      SITEGUARD AI / SAFEGEAR — FULL SYSTEM VERIFICATION{RESET}")
    print(f"{BOLD}=================================================================={RESET}")

    start_time = time.time()

    test_dependencies()
    test_model_and_ppe()
    test_database()
    test_cv_pipeline()
    test_flask_routes()

    elapsed = time.time() - start_time

    log_header("Verification Summary")
    total = passed_tests + failed_tests
    print(f"Total Tests Run: {BOLD}{total}{RESET}")
    print(f"Passed:          {BOLD}{GREEN}{passed_tests}{RESET}")
    print(f"Failed:          {BOLD}{RED if failed_tests > 0 else GREEN}{failed_tests}{RESET}")
    print(f"Execution Time:  {elapsed:.2f} seconds\n")

    if failed_tests == 0:
        print(f"{BOLD}{GREEN}✓ ALL SYSTEM MODULES FULLY OPERATIONAL AND VERIFIED!{RESET}\n")
        sys.exit(0)
    else:
        print(f"{BOLD}{RED}✗ {failed_tests} MODULE(S) FAILED VERIFICATION. REVIEW LOGS ABOVE.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
