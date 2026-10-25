"""
Tests for Multi-Object Worker Tracker and Spatial IoU Matching.
"""
import pytest
from cv.tracker import WorkerTracker, compute_iou

def test_compute_iou():
    box_a = [100, 100, 200, 200]
    box_b = [100, 100, 200, 200]
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    box_c = [300, 300, 400, 400]
    assert compute_iou(box_a, box_c) == pytest.approx(0.0)

    box_d = [150, 100, 250, 200]
    iou = compute_iou(box_a, box_d)
    assert 0.3 < iou < 0.4

def test_tracker_creation_and_continuity():
    tracker = WorkerTracker(iou_threshold=0.30)
    
    # Frame 1: One worker detected
    frame1_detections = [
        {"bbox": [100, 100, 200, 300], "confidence": 0.90}
    ]
    workers1 = tracker.update(frame1_detections)
    assert len(workers1) == 1
    worker_id = workers1[0]["worker_id"]
    assert worker_id == 1

    # Frame 2: Worker moves slightly
    frame2_detections = [
        {"bbox": [105, 102, 205, 305], "confidence": 0.88}
    ]
    workers2 = tracker.update(frame2_detections)
    assert len(workers2) == 1
    # Track ID should be preserved across frames
    assert workers2[0]["worker_id"] == worker_id

def test_tracker_multiple_workers_and_id_assignment():
    tracker = WorkerTracker(iou_threshold=0.30)
    
    # Frame 1: Two distinct workers
    detections = [
        {"bbox": [50, 50, 150, 250], "confidence": 0.85},
        {"bbox": [300, 50, 400, 250], "confidence": 0.92}
    ]
    workers = tracker.update(detections)
    assert len(workers) == 2
    ids = {w["worker_id"] for w in workers}
    assert len(ids) == 2

def test_tracker_reset():
    tracker = WorkerTracker()
    tracker.update([{"bbox": [100, 100, 200, 300], "confidence": 0.90}])
    assert len(tracker.tracks) == 1
    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.next_id == 1
