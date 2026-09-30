# SiteGuard AI — AI-Powered Construction Safety Monitoring

A professional, modern, production-grade Computer Vision and Web Application platform for automated construction worker safety surveillance, Personal Protective Equipment (PPE) compliance detection, spatial hazard zone incursion monitoring, and temporal violation verification.

---

## 1. Product Identity

- **Product Name:** SiteGuard AI
- **Subtitle:** AI-Powered Construction Safety Monitoring
- **Target Users:** Construction Site Managers, Safety Officers, Supervisors, Project Managers, Safety Administrators.
- **Design Language:** Industrial modern dark theme with Safety Orange (`#f97316`), Safety Yellow (`#eab308`), Hazard Red (`#ef4444`), and Emerald Safe (`#10b981`).
- **Core Philosophy:** **REAL DATA ONLY** — Every statistic, count, chart, and alert is derived strictly from real camera feeds, real YOLOv8 detections, real database queries, and real safety officer actions.

---

## 2. Core Architecture & End-to-End Data Flow

```
   CONSTRUCTION SITE
          │
          ▼
   Camera / Image / Video (Hardware Webcam, RTSP, WebRTC, MP4)
          │
          ▼
   YOLOv8 Computer Vision Inference (Classes from data.yaml)
          │
          ▼
   Spatial PPE Association (Head -> Helmet, Torso -> Vest)
          │
          ▼
   Spatial Zone & Temporal State Engine (Point-in-polygon, 10+ frames buffer)
          │
     ┌────┴────────────────────────┐
     ▼                             ▼
 COMPLIANT                    VIOLATION
(Track Stats Updated)              │
                                   ▼
                              Alert Generated (OPEN)
                                   │
                                   ▼
                              In-App Notification & Evidence Snapshot Cropped
                                   │
                                   ▼
                              Safety Officer Action (Acknowledge -> Review -> Resolve)
                                   │
                                   ▼
                              Formal Safety Audit & PDF / CSV Report Generation
```

---

## 3. Key Platform Features & Modules

### 📊 1. Operations Overview Dashboard (`/`)
- Top KPI Cards: Active Cameras, Active Workers Observed, Today's Violations, Open Safety Alerts, Resolved Incidents, Real Compliance Rate ($\frac{\text{Compliant Observations}}{\text{Total Observations}} \times 100$).
- Live video feed widget with channel status and active CV HUD.
- Actionable real-time safety alert feed with instant evidence viewer modal.
- Active safety compliance rules and operational infrastructure summary.
- Professional empty states when no data is present.

### 📹 2. Live Monitoring Workstation (`/monitor`)
- Real-time video inference supporting Hardware Webcam, WebRTC Browser Camera, and Construction Dataset Stream Simulator.
- Live telemetry HUD: Stream FPS, active worker count, active violations, and stream channel.
- Interactive Zone Polygon Drawing Tool: Define `RESTRICTED`, `HAZARD`, and `SAFE` polygons directly on the canvas.
- Real-time worker tracking roster showing individual PPE badges (`Helmet`, `Vest`, `Zone`).

### 🎬 3. Video Analysis Studio (`/video-analysis`)
- Drag-and-drop site video upload (`.mp4`, `.avi`, `.mov`).
- Sequential frame extraction with live inference progress bar.
- Generates downloadable annotated MP4 video for in-browser playback.
- Displays actual frame count, tracked workers count, violation count, and compliance breakdown.

### 🖼️ 4. Image Inspection Studio (`/image-analysis`)
- Upload high-resolution construction photos (`.jpg`, `.png`, `.webp`).
- Instant YOLOv8 inference with anatomical spatial association.
- High-resolution annotated image viewer with download capability.
- Exact raw detections breakdown table (Class, Confidence %, Bounding Box Coordinates `[x1, y1, x2, y2]`).

### ⚠️ 5. Safety Violations History (`/violations`)
- Comprehensive filterable database log of all confirmed safety violations.
- Filter by Violation Type, Severity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), Site, and Keyword Search.
- Direct CSV audit log export.

### 🚨 6. Alert Center (`/alerts`)
- Real-time alert feed with severity badges and lifecycle state management.
- State transitions: `OPEN` $\rightarrow$ `ACKNOWLEDGED` $\rightarrow$ `UNDER_REVIEW` $\rightarrow$ `RESOLVED`.
- Interactive resolution modal to assign safety officers, record corrective action, and log resolution notes.

### 📋 7. Incident Management (`/incidents`)
- Formal incident audit workspace with root cause analysis, corrective action tracking, and safety officer assignments.
- Timestamped cropped snapshot evidence viewer.

### 🏗️ 8. Construction Sites & Zones (`/sites`)
- Site management (Site Code, Name, Location, Project Name, Assigned Safety Officer).
- Multi-site hierarchy: Site $\rightarrow$ Monitored Zones $\rightarrow$ Cameras $\rightarrow$ Violations.
- Polygon coordinate definitions and zone type management.

### 📷 9. Camera Network (`/cameras`)
- Manage physical and virtual cameras across construction sites.
- Stream sources: Hardware Webcam (DirectShow/V4L2), Browser WebRTC, RTSP/HTTP IP stream. "View Feed" opens the camera's configured source on the monitor page.
- Live status heartbeat tracking (`ONLINE` / `OFFLINE`) with last-seen timestamps and resolution/FPS specs.

### 👷 10. Worker Tracking & Compliance (`/workers`)
- Anonymous Worker Track management (`Worker Track #1`, `Worker Track #2`, ...).
- Lifetime observation counts, compliant frames, violation events, and individual safety rating.

### 📈 11. Safety Analytics (`/analytics`)
- Visual charts generated strictly from real database records:
  - Violation Category Distribution (Doughnut)
  - Severity Breakdown (Pie)
  - Real Dates Frequency Timeline (Line)
  - Time-of-Day Hourly Distribution (Bar)
  - Stream / Camera Breakdown (Bar)
  - Site-wise Breakdown (Bar)

### 📑 12. Report Center (`/reports`)
- Generates formal compliance audit reports for selected timeframes (`TODAY`, `7DAYS`, `30DAYS`, `ALL`).
- Formatted for print and export to PDF.
- CSV data export.

### 🧠 13. AI Detection Model Architecture (`/model`)
- YOLOv8 weights inspection (`model/best.pt`), inference device (PyTorch CUDA/CPU), and confidence/IoU hyperparameters.
- Class schema read from the weights themselves. The bundled `model/best.pt` detects 11 classes: `helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`. Class names are normalized (e.g. `Glove`→`gloves`, `Ear-protection`→`ear_protection`), so other PPE datasets map onto the same 8 PPE categories. `no_*` detections override a weaker positive detection on the same worker.
- PPE categories the model cannot detect (hearing protection, harness, protective pants with the bundled weights) are never reported as missing.

### 💡 14. AI Safety Insights (`/insights`)
- Deterministic fact synthesis engine analyzing actual recorded violation events without hallucinated numbers.

### 💓 15. System Health Diagnostics (`/system-health`)
- Live checks: SQLite round-trip latency, model loaded and class count, camera device open, snapshot directory writable, free disk space, uptime.

### 🔔 16. In-App Notifications (`/notifications`)
- Real-time notification feed for `HIGH` and `CRITICAL` severity events with "Mark All as Read" action.

### 📜 17. Immutable Audit Trail (`/audit-logs`)
- Complete audit trail recording all safety officer and administrative actions.

### ⚙️ 18. System Settings & Safety Rules (`/settings`)
- Fine-tune YOLO confidence thresholds, IoU overlap thresholds, and temporal sliding-window verification frames.

---

## 4. Installation & Setup

### Prerequisites
- Python 3.9+
- Webcam or construction video/image samples

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch SiteGuard AI Web Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | random, stored in `database/.secret_key` | Flask session signing key |
| `SITEGUARD_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the network) |
| `SITEGUARD_PORT` | `5000` | HTTP port |

### 4. Roles
| Role | Can |
|---|---|
| `ADMIN` | Everything, including sites, cameras, settings and the audit log |
| `SAFETY_OFFICER` | Review violations, manage alerts/incidents, draw zones, run image/video analysis |
| `SUPERVISOR` | Review violations, manage alerts/incidents, run image/video analysis |
| `VIEWER` | Read-only |

### 5. How violations are raised
- **Live / video:** a required PPE item must be missing for `MIN_VIOLATION_FRAMES` consecutive frames (Settings page) before a violation is logged; a cleared violation is not re-raised for the same worker within `VIOLATION_COOLDOWN_FRAMES`. A restricted zone breach needs `MIN_ZONE_FRAMES` frames inside the zone.
- **Single image:** rules are evaluated directly on the detections.
- Each violation opens an incident and an alert automatically; confirming a violation in review confirms that incident, and marking it a false positive closes it.

---

## 5. Technology Stack

- **Backend:** Python 3, Flask 3.0+, SQLite3
- **Computer Vision:** Ultralytics YOLOv8, OpenCV (`cv2`), PyTorch, NumPy, Pillow
- **Frontend:** Modern Semantic HTML5, Vanilla CSS Design System (no heavy utility dependencies), Vanilla JavaScript, Chart.js
- **Typography:** Google Fonts Inter & JetBrains Mono
