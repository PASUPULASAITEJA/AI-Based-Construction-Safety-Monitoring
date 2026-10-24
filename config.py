"""
AI-Based Construction Safety Monitoring
Central Configuration Module
"""
import os
import secrets
import torch

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "construction")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FALLBACK_MODEL = "yolov8n.pt"

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

DB_PATH = os.path.join(BASE_DIR, "database", "safety_monitor.db")

# Ensure required directories exist
for folder in [MODEL_DIR, UPLOADS_DIR, OUTPUTS_DIR, SNAPSHOTS_DIR, RUNS_DIR, os.path.join(BASE_DIR, "database")]:
    os.makedirs(folder, exist_ok=True)

# Hardware Device Detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CV & Inference Hyperparameters (Defaults)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# Temporal Verification Parameters
MIN_VIOLATION_FRAMES = 10   # Number of consecutive missing PPE frames before confirming violation
MIN_ZONE_FRAMES = 8         # Number of consecutive frames inside zone before violation
VIOLATION_COOLDOWN_FRAMES = 30 # Cooldown frames between duplicate alerts for the same worker

# Anatomical Sub-region Thresholds for Worker-PPE Association
HEAD_REGION_SPAN = (0.0, 0.45)   # Top 45% for head/helmet (generous for bending workers)
TORSO_REGION_SPAN = (0.15, 0.85) # Middle 15% to 85% for torso/vest
MIN_OVERLAP_RATIO = 0.15        # Minimum fraction of a PPE box inside a body region to link it to that worker

# Video processing
DEFAULT_FRAME_SKIP = 1           # Process every Nth frame (1 = process all)
MAX_VIDEO_PREVIEW_WIDTH = 1280

# Video upload processing limits
MAX_VIDEO_FRAMES = 600           # Upper bound of frames analysed per uploaded video
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Minimum seconds between workers_track DB writes for the same tracked worker
WORKER_OBSERVATION_INTERVAL_SEC = 1.0

# Web server
HOST = os.environ.get("SITEGUARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("SITEGUARD_PORT", "5000"))
DEBUG = False

def _load_secret_key():
    """SECRET_KEY env var wins; otherwise a random key is generated once and kept in database/.secret_key."""
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key
    key_path = os.path.join(BASE_DIR, "database", ".secret_key")
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    key = secrets.token_hex(32)
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key)
    return key

SECRET_KEY = _load_secret_key()
