"""
System Configuration Settings
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'safety.db'}")

# Storage paths
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads"))
SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR", DATA_DIR / "snapshots"))
MODELS_DIR = BASE_DIR / "models"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
CUSTOM_MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "best.pt"))
FALLBACK_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))

# Anatomical Association Parameters
HEAD_ROI_HEIGHT_RATIO = 0.28         # Top 28% of height
TORSO_ROI_TOP_RATIO = 0.20           # 20% from top
TORSO_ROI_BOTTOM_RATIO = 0.75        # 75% from top
MIN_IOI_THRESHOLD = 0.40             # Minimum Intersection-over-Item ratio (>=40%)

# Temporal Sliding Window & Hysteresis
SLIDING_WINDOW_SIZE = 30             # Deque maxlen = 30 frames
VIOLATION_CONFIRM_THRESHOLD = 22     # >= 22 / 30 frames (73.3% persistence)
LATCH_RECOVERY_FRAMES = 20           # 20 consecutive compliant frames to unlatch

# Live Video Stream Settings
STREAM_FPS = 15
JPEG_QUALITY = 75
