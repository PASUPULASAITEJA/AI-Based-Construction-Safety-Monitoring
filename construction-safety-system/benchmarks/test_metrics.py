"""
Automated Metric & Pipeline Unit Tests
Validates:
1. Detector fallback mechanism & output schemas
2. ByteTrack ID continuity across consecutive frames
3. Anatomical PPE Head (top 28%) and Torso (20-75%) association with IoI >= 0.40
4. Spatial restricted zone pointPolygonTest at bottom-center foot coordinate P_feet
5. Temporal deque(30) confirmation threshold (>= 22 frames) and hysteresis latching (20 frames)
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from core.detector import ObjectDetector
from core.tracker import ByteTrackTracker
from core.association import (
    associate_ppe_to_workers,
    compute_intersection_over_item,
    is_centroid_inside_roi
)
from core.spatial import evaluate_spatial_containment, compute_foot_anchor
from core.temporal_engine import TemporalSafetyEngine, WorkerTemporalState

# -----------------------------------------------------------------------------
# 1. Detector Tests
# -----------------------------------------------------------------------------
def test_detector_fallback_and_inference():
    """Validates that detector initializes cleanly and returns structured detections."""
    detector = ObjectDetector()
    assert detector.model is not None

    # Test on blank 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    workers, ppe = detector.detect(frame)
    assert isinstance(workers, list)
    assert isinstance(ppe, list)

# -----------------------------------------------------------------------------
# 2. ByteTrack Multi-Object Tracking Tests
# -----------------------------------------------------------------------------
def test_bytetrack_id_continuity():
    """Validates that persistent track_id is maintained across consecutive frames."""
    tracker = ByteTrackTracker(max_age=15, min_hits=1)

    # Frame 1: Worker at (100, 100, 200, 300)
    dets_f1 = [{"bbox": [100, 100, 200, 300], "confidence": 0.85, "class_name": "worker"}]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 1
    t_id = tracks_f1[0]["track_id"]

    # Frame 2: Worker slightly shifted to (105, 102, 205, 302)
    dets_f2 = [{"bbox": [105, 102, 205, 302], "confidence": 0.82, "class_name": "worker"}]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0]["track_id"] == t_id, "Track ID must remain continuous across adjacent frames"

# -----------------------------------------------------------------------------
# 3. Anatomical PPE Association Tests
# -----------------------------------------------------------------------------
def test_anatomical_association_head_and_torso():
    """
    Worker BBox: [100, 100, 200, 400] -> W=100, H=300
    Head ROI:   [100, 100, 200, 100 + 0.28 * 300] = [100, 100, 200, 184]
    Torso ROI:  [100, 100 + 0.20 * 300, 200, 100 + 0.75 * 300] = [100, 160, 200, 325]
    """
    worker = {"track_id": 1, "bbox": [100, 100, 200, 400], "confidence": 0.90}

    # Case A: Helmet placed right on head -> [120, 110, 180, 160] (centroid: (150, 135) inside R_head)
    valid_helmet = {"bbox": [120, 110, 180, 160], "confidence": 0.85, "class_name": "helmet"}
    # Case B: Vest placed on torso -> [110, 180, 190, 290] (centroid: (150, 235) inside R_torso)
    valid_vest = {"bbox": [110, 180, 190, 290], "confidence": 0.88, "class_name": "vest"}

    enriched = associate_ppe_to_workers([worker], [valid_helmet, valid_vest])
    assert len(enriched) == 1
    w_res = enriched[0]

    assert w_res["has_helmet"] is True, "Helmet must be associated"
    assert w_res["has_vest"] is True, "Vest must be associated"
    assert w_res["helmet_conf"] == 0.85
    assert w_res["vest_conf"] == 0.88

def test_anatomical_association_rejection():
    """Tests that misplaced PPE (e.g. helmet on feet) is strictly rejected."""
    worker = {"track_id": 2, "bbox": [100, 100, 200, 400], "confidence": 0.90}
    
    # Helmet bounding box placed near feet (y=360..390)
    misplaced_helmet = {"bbox": [120, 360, 180, 390], "confidence": 0.80, "class_name": "helmet"}

    enriched = associate_ppe_to_workers([worker], [misplaced_helmet])
    assert enriched[0]["has_helmet"] is False, "Misplaced helmet outside R_head must NOT be associated"

# -----------------------------------------------------------------------------
# 4. Spatial Foot-Anchor Geometry Tests
# -----------------------------------------------------------------------------
def test_spatial_ground_anchor_containment():
    """
    Worker BBox: [100, 100, 200, 300]
    P_feet = (int((100+200)/2), 300) = (150, 300)
    """
    bbox = [100, 100, 200, 300]
    feet = compute_foot_anchor(bbox)
    assert feet == (150, 300), f"P_feet must be exactly bottom-center, got {feet}"

    # Zone 1: Contains (150, 300) -> [100, 250], [200, 250], [200, 350], [100, 350]
    zone_inside = {
        "id": 10,
        "name": "Excavation Pit",
        "severity": "DANGER",
        "coordinates": [[100, 250], [200, 250], [200, 350], [100, 350]]
    }
    # Zone 2: Outside (500, 500)
    zone_outside = {
        "id": 11,
        "name": "Storage Area",
        "severity": "WARNING",
        "coordinates": [[400, 400], [500, 400], [500, 500], [400, 500]]
    }

    breaches = evaluate_spatial_containment(bbox, [zone_inside, zone_outside])
    assert len(breaches) == 1
    assert breaches[0]["zone_id"] == 10
    assert breaches[0]["zone_name"] == "Excavation Pit"

# -----------------------------------------------------------------------------
# 5. Temporal Sliding Window & Hysteresis Latching Tests
# -----------------------------------------------------------------------------
def test_temporal_sliding_window_latching():
    """
    Validates:
    - 21 frames of non-compliance: NO violation triggered.
    - 22nd frame of non-compliance: Violation TRIGGERED & LATCHED. Snapshot saved.
    - 19 frames of compliance: Violation REMAINS LATCHED.
    - 20th frame of compliance: Violation UNLATCHED & RESOLVED.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        engine = TemporalSafetyEngine(
            window_size=30,
            trigger_threshold=22,
            recovery_threshold=20,
            snapshots_dir=temp_dir
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        non_compliant_worker = {
            "track_id": 99,
            "bbox": [100, 100, 200, 300],
            "has_helmet": False,
            "has_vest": True,
            "confidence": 0.92
        }

        # Step 1: Feed 21 non-compliant frames -> No incident triggered yet
        for f in range(1, 22):
            events = engine.process_worker(non_compliant_worker, [], dummy_frame)
            new_events = [e for e in events if e.get("is_new")]
            assert len(new_events) == 0, f"Frame {f}: Should not trigger incident under 22 frames"

        # Step 2: Feed 22nd non-compliant frame -> MUST trigger incident!
        events_22 = engine.process_worker(non_compliant_worker, [], dummy_frame)
        new_events_22 = [e for e in events_22 if e.get("is_new")]
        assert len(new_events_22) == 1, "22nd frame MUST trigger confirmed incident"
        assert new_events_22[0]["violation_type"] == "NO_HELMET"
        assert new_events_22[0]["snapshot_path"] is not None

        # Verify snapshot file created on disk
        snapshot_files = os.listdir(temp_dir)
        assert len(snapshot_files) >= 1, "Snapshot cropped evidence must be written to disk"

        # Step 3: Worker begins complying. Feed 19 compliant frames -> Must remain latched!
        compliant_worker = {
            "track_id": 99,
            "bbox": [100, 100, 200, 300],
            "has_helmet": True,
            "has_vest": True,
            "confidence": 0.95
        }

        for f in range(1, 20):
            events = engine.process_worker(compliant_worker, [], dummy_frame)
            active_events = [e for e in events if e.get("is_active")]
            assert len(active_events) == 1, f"Compliant frame {f}: Must remain latched until 20 frames"

        # Step 4: Feed 20th compliant frame -> Must unlatch and resolve!
        events_recovery = engine.process_worker(compliant_worker, [], dummy_frame)
        resolved_events = [e for e in events_recovery if e.get("is_resolved")]
        assert len(resolved_events) == 1, "20th compliant frame MUST unlatch and resolve the incident"
        assert resolved_events[0]["violation_type"] == "NO_HELMET"

    finally:
        shutil.rmtree(temp_dir)
