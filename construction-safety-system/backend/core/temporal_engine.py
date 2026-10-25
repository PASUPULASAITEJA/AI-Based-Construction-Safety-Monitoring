"""
Temporal Confirmation Engine & Hysteresis Latching Module
Eliminates transient false alarms using 30-frame sliding deque ring-buffers.
Triggers incident when non-compliance persistence >= 22 / 30 frames.
Latches state and saves cropped snapshot evidence until 20 consecutive compliant frames are observed.
"""
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import os
import time
import uuid
import datetime
import cv2
import numpy as np
import config

class WorkerTemporalState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        # Ring buffers (maxlen=30)
        self.helmet_missing_buffer = deque(maxlen=config.SLIDING_WINDOW_SIZE)
        self.vest_missing_buffer = deque(maxlen=config.SLIDING_WINDOW_SIZE)
        self.zone_intrusion_buffer = deque(maxlen=config.SLIDING_WINDOW_SIZE)

        # Hysteresis Latch Flags
        self.latched_violations: Dict[str, bool] = {
            "NO_HELMET": False,
            "NO_VEST": False,
            "ZONE_INTRUSION": False
        }

        # Consecutive compliant frames counter for unlatching
        self.consecutive_compliant: Dict[str, int] = {
            "NO_HELMET": 0,
            "NO_VEST": 0,
            "ZONE_INTRUSION": 0
        }

        # Active incident codes associated with currently latched episodes
        self.active_incident_codes: Dict[str, Optional[str]] = {
            "NO_HELMET": None,
            "NO_VEST": None,
            "ZONE_INTRUSION": None
        }

        # Frame counter for this tracklet
        self.total_observed_frames = 0
        self.last_seen_time = time.time()


class TemporalSafetyEngine:
    def __init__(
        self,
        window_size: int = config.SLIDING_WINDOW_SIZE,
        trigger_threshold: int = config.VIOLATION_CONFIRM_THRESHOLD,
        recovery_threshold: int = config.LATCH_RECOVERY_FRAMES,
        snapshots_dir: str = str(config.SNAPSHOTS_DIR)
    ):
        self.window_size = window_size
        self.trigger_threshold = trigger_threshold
        self.recovery_threshold = recovery_threshold
        self.snapshots_dir = snapshots_dir
        self.worker_states: Dict[int, WorkerTemporalState] = {}
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def process_worker(
        self,
        worker: Dict[str, Any],
        breached_zones: List[Dict[str, Any]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Ingests instantaneous worker state and updates the sliding window.
        Returns a list of newly triggered violation events or ongoing latched violations.
        """
        track_id = worker["track_id"]
        if track_id not in self.worker_states:
            self.worker_states[track_id] = WorkerTemporalState(track_id)

        state = self.worker_states[track_id]
        state.total_observed_frames += 1
        state.last_seen_time = time.time()

        # Instantaneous non-compliance: 1 if missing/breached, 0 if compliant
        is_missing_helmet = 1 if not worker.get("has_helmet", False) else 0
        is_missing_vest = 1 if not worker.get("has_vest", False) else 0
        is_in_zone = 1 if len(breached_zones) > 0 else 0

        # Append to sliding buffers
        state.helmet_missing_buffer.append(is_missing_helmet)
        state.vest_missing_buffer.append(is_missing_vest)
        state.zone_intrusion_buffer.append(is_in_zone)

        events: List[Dict[str, Any]] = []

        # Check all 3 violation categories
        violations_to_check = [
            ("NO_HELMET", state.helmet_missing_buffer, is_missing_helmet, "Missing Mandatory Safety Hardhat", "HIGH", None),
            ("NO_VEST", state.vest_missing_buffer, is_missing_vest, "Missing High-Visibility Safety Vest", "HIGH", None),
            ("ZONE_INTRUSION", state.zone_intrusion_buffer, is_in_zone, 
             f"Trespassing inside restricted zone: {breached_zones[0]['zone_name'] if breached_zones else 'Hazard Area'}",
             "CRITICAL", breached_zones[0] if breached_zones else None)
        ]

        for v_type, buffer, instant_val, desc, severity, zone_info in violations_to_check:
            event = self._evaluate_hysteresis(
                state=state,
                v_type=v_type,
                buffer=buffer,
                instant_val=instant_val,
                worker=worker,
                frame=frame,
                description=desc,
                severity=severity,
                zone_info=zone_info
            )
            if event:
                events.append(event)

        return events

    def _evaluate_hysteresis(
        self,
        state: WorkerTemporalState,
        v_type: str,
        buffer: deque,
        instant_val: int,
        worker: Dict[str, Any],
        frame: np.ndarray,
        description: str,
        severity: str,
        zone_info: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Evaluates temporal sliding window confirmation and recovery latching."""
        is_latched = state.latched_violations[v_type]
        non_compliance_count = sum(buffer)

        # 1. State: NOT YET LATCHED -> Check if threshold met
        if not is_latched:
            if non_compliance_count >= self.trigger_threshold:
                # Trigger new confirmed violation incident!
                state.latched_violations[v_type] = True
                state.consecutive_compliant[v_type] = 0
                incident_code = f"INC-{uuid.uuid4().hex[:8].upper()}"
                state.active_incident_codes[v_type] = incident_code

                # Save cropped evidence snapshot
                snapshot_path = self._save_cropped_snapshot(frame, worker["bbox"], state.track_id, v_type)

                return {
                    "is_new": True,
                    "is_active": True,
                    "incident_code": incident_code,
                    "track_id": state.track_id,
                    "violation_type": v_type,
                    "severity": severity,
                    "description": description,
                    "snapshot_path": snapshot_path,
                    "confidence": float(worker.get("confidence", 0.85)),
                    "zone_id": zone_info.get("zone_id") if zone_info else None,
                    "zone_name": zone_info.get("zone_name") if zone_info else None
                }
            return None

        # 2. State: ALREADY LATCHED -> Check for 20 consecutive compliant frames to unlatch
        else:
            if instant_val == 0:
                state.consecutive_compliant[v_type] += 1
            else:
                state.consecutive_compliant[v_type] = 0

            if state.consecutive_compliant[v_type] >= self.recovery_threshold:
                # Unlatch: Worker has returned to compliance
                resolved_code = state.active_incident_codes[v_type]
                state.latched_violations[v_type] = False
                state.consecutive_compliant[v_type] = 0
                state.active_incident_codes[v_type] = None

                return {
                    "is_new": False,
                    "is_active": False,
                    "is_resolved": True,
                    "incident_code": resolved_code,
                    "track_id": state.track_id,
                    "violation_type": v_type
                }

            # Still latched - ongoing continuous incident
            return {
                "is_new": False,
                "is_active": True,
                "incident_code": state.active_incident_codes[v_type],
                "track_id": state.track_id,
                "violation_type": v_type,
                "severity": severity,
                "description": description
            }

    def _save_cropped_snapshot(self, frame: np.ndarray, bbox: List[int], track_id: int, v_type: str) -> str:
        """Crops worker bounding box with margin and saves snapshot evidence."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 15% contextual padding
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = frame

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_trk{track_id}_{v_type.lower()}.jpg"
        full_path = os.path.join(self.snapshots_dir, filename)

        cv2.imwrite(full_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"/data/snapshots/{filename}"

    def prune_stale_workers(self, timeout_seconds: float = 10.0):
        """Removes tracking states for workers no longer in scene."""
        now = time.time()
        stale_ids = [tid for tid, s in self.worker_states.items() if (now - s.last_seen_time) > timeout_seconds]
        for tid in stale_ids:
            del self.worker_states[tid]
