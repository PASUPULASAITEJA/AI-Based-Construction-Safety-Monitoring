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
- Stream sources: Hardware Webcam (DirectShow/V4L2), Browser WebRTC, Dataset Simulator, RTSP IP stream.
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
- Dynamic class schema directly read from `construction/data.yaml` (`['Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Person', 'Vest']`).

### 💡 14. AI Safety Insights (`/insights`)
- Deterministic fact synthesis engine analyzing actual recorded violation events without hallucinated numbers.

### 💓 15. System Health Diagnostics (`/system-health`)
- Real-time diagnostic monitors for Backend Web Server, SQLite Database, YOLOv8 AI Model, Live Video Engine, Storage Subsystem, and Alert Dispatcher.

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

### 3. Run End-to-End System Verification Suite
```bash
python verify_full_system.py
```

---

## 5. Technology Stack

- **Backend:** Python 3, Flask 3.0+, SQLite3
- **Computer Vision:** Ultralytics YOLOv8, OpenCV (`cv2`), PyTorch, NumPy, Pillow
- **Frontend:** Modern Semantic HTML5, Vanilla CSS Design System (no heavy utility dependencies), Vanilla JavaScript, Chart.js
- **Typography:** Google Fonts Inter & JetBrains Mono
