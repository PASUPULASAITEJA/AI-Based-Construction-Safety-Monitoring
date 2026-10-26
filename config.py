"""
AI-Based Construction Safety Monitoring
Central Configuration Module
"""
import os
import torch

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data_17classes.yaml")
if not os.path.exists(DATA_YAML_PATH):
    DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FALLBACK_MODEL = "yolov8n.pt"

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

UPLOAD_FOLDER = UPLOADS_DIR
OUTPUT_FOLDER = OUTPUTS_DIR
SNAPSHOT_FOLDER = SNAPSHOTS_DIR

DB_PATH = os.path.join(BASE_DIR, "database", "safety_monitor.db")

# 17 Detection Classes (1 Worker Anchor + 8 Compliance Pairs)
CLASSES_17 = [
    "person",          # 0
    "helmet",          # 1
    "no-helmet",       # 2
    "goggles",         # 3
    "no-goggles",      # 4
    "earmuff",         # 5
    "no-earmuff",      # 6
    "vest",            # 7
    "no-vest",         # 8
    "safety-pants",    # 9
    "no-safety-pants", # 10
    "gloves",          # 11
    "no-gloves",       # 12
    "boots",           # 13
    "no-boots",        # 14
    "harness",         # 15
    "no-harness"       # 16
]

# 8 Standard PPE Categories
PPE_CATEGORIES = [
    "helmet",
    "goggles",
    "earmuff",
    "vest",
    "safety_pants",
    "gloves",
    "boots",
    "harness"
]

# Ensure required directories exist
for folder in [MODEL_DIR, UPLOADS_DIR, OUTPUTS_DIR, SNAPSHOTS_DIR, RUNS_DIR, os.path.join(BASE_DIR, "database")]:
    os.makedirs(folder, exist_ok=True)

# Hardware Device Detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CV & Inference Hyperparameters (Defaults)
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# Temporal Verification Parameters
MIN_VIOLATION_FRAMES = 10      # Number of consecutive missing PPE frames before confirming violation
MIN_ZONE_FRAMES = 8            # Number of consecutive frames inside zone before violation
VIOLATION_COOLDOWN_FRAMES = 30 # Cooldown frames between duplicate alerts for the same worker

# Anatomical Sub-region Thresholds for Worker-PPE Association
HEAD_REGION_SPAN = (0.00, 0.25)   # Top 25% for helmet, goggles, earmuff
TORSO_REGION_SPAN = (0.25, 0.70)  # 25% to 70% for vest, harness
LOWER_REGION_SPAN = (0.70, 1.00)  # 70% to 100% for safety-pants, boots
MIN_OVERLAP_RATIO = 0.05          # Minimum bounding box overlap ratio to link PPE item to worker

# Video processing
DEFAULT_FRAME_SKIP = 1           # Process every Nth frame (1 = process all)
MAX_VIDEO_PREVIEW_WIDTH = 1280

# Web server
HOST = "0.0.0.0"
PORT = 5000
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")
SECRET_KEY = "construction-safety-monitoring-secret-key-2026"
