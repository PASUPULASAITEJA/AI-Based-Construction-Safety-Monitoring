"""
Tests for Temporal Safety Rule Engine and False-Alarm Reduction.
"""
import pytest
from safety.violation_manager import ViolationManager, WorkerTemporalState

def test_temporal_state_tracking():
    state = WorkerTemporalState(worker_id=1)
    # Update frame with helmet present
    state.update({"HEAD": 0.90}, zone_type="SAFE")
    assert state.missing_frames["HEAD"] == 0

    # Next 3 frames: helmet missing
    for _ in range(3):
        state.update({"HEAD": 0.0}, zone_type="SAFE")
    assert state.missing_frames["HEAD"] == 3

def test_violation_manager_temporal_confirmation():
    # min_violation_frames = 5
    vm = ViolationManager(min_violation_frames=5, min_zone_frames=3, cooldown_frames=10)
    
    worker = {
        "worker_id": 1,
        "label": "Worker #1",
        "bbox": [100, 100, 200, 300],
        "confidence": 0.90,
        "zone_type": "SAFE",
        "HEAD_conf": 0.0,      # Helmet missing
        "VISIBILITY_conf": 0.85 # Vest present
    }

    # Frames 1 through 4: Missing frames < 5 -> NO incident logged
    for frame_idx in range(1, 5):
        verified, new_incidents = vm.process_frame([dict(worker)])
        assert len(new_incidents) == 0, f"False alarm raised at frame {frame_idx}"
        assert verified[0]["is_compliant"] is True

    # Frame 5: Missing frames reaches 5 -> Confirmed incident generated
    verified, new_incidents = vm.process_frame([dict(worker)])
    assert len(new_incidents) == 1
    assert "NO_HEAD" in new_incidents[0]["violation_type"]
    assert verified[0]["is_compliant"] is False

    # Frame 6: Continuous violation -> Should NOT duplicate incident
    verified, new_incidents = vm.process_frame([dict(worker)])
    assert len(new_incidents) == 0, "Duplicate incident generated on consecutive frame"

def test_restricted_zone_incursion_confirmation():
    vm = ViolationManager(min_violation_frames=5, min_zone_frames=3)
    
    worker = {
        "worker_id": 2,
        "label": "Worker #2",
        "bbox": [100, 100, 200, 300],
        "confidence": 0.90,
        "zone_type": "RESTRICTED",
        "zone_name": "Excavation Zone",
        "zone_id": 1,
        "HEAD_conf": 0.90,
        "VISIBILITY_conf": 0.90
    }

    # Frames 1 and 2: under min_zone_frames
    for _ in range(2):
        _, new_incidents = vm.process_frame([dict(worker)])
        assert len(new_incidents) == 0

    # Frame 3: reaches min_zone_frames (3) -> Incident generated
    _, new_incidents = vm.process_frame([dict(worker)])
    assert len(new_incidents) == 1
    assert "ZONE" in new_incidents[0]["violation_type"] or "RESTRICTED" in new_incidents[0]["violation_type"]
