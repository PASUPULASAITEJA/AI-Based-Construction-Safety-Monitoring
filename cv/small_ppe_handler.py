"""
AI-Based Construction Safety Monitoring — Small & Hard-to-Detect PPE Handler
===========================================================================
Addresses micro-scale and easily occluded PPE (goggles, earmuffs, pants chaps):
  1. Temporal confirmation buffer: only flags "missing" if confidence > 0.60 AND absent for >= N frames
  2. Tri-state status resolution: True (Compliant), False (Missing), 'uncertain' (Low-confidence / ?)
  3. High-resolution body-zone crop inference pass for zoomed small-object detection

Author: Construction CV Safety Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque


SMALL_PPE_CATEGORIES = {"goggles", "earmuff", "safety_pants", "gloves"}


class TemporalPPETracker:
    """
    Maintains per-worker temporal state across video frames to eliminate flickering
    and false positives for small, challenging PPE categories.
    """
    def __init__(
        self,
        min_absence_frames: int = 10,
        high_conf_missing_threshold: float = 0.60,
        uncertainty_threshold: float = 0.15  # keep in line with PPEAssociator.min_detection_confidence
    ):
        self.min_absence_frames = min_absence_frames
        self.high_conf_missing_threshold = high_conf_missing_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # History store: worker_id -> { category: {"absent_frames": int, "recent_conf": deque} }
        self.worker_history: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get_or_create_worker_state(self, worker_id: str) -> Dict[str, Dict[str, Any]]:
        if worker_id not in self.worker_history:
            self.worker_history[worker_id] = {}
        return self.worker_history[worker_id]

    def update_and_resolve(
        self,
        worker_id: str,
        category: str,
        raw_present: bool,
        confidence: float,
        is_small_ppe: bool
    ) -> Union[bool, str]:
        """
        Resolves PPE status into:
          - True (compliant / detected)
          - False (confirmed missing)
          - 'uncertain' (insufficient confidence or transient absence)
        """
        w_state = self.get_or_create_worker_state(worker_id)
        if category not in w_state:
            w_state[category] = {
                "absent_streak": 0,
                "history": deque(maxlen=15),
                "last_status": "uncertain"
            }

        item_state = w_state[category]
        item_state["history"].append((raw_present, confidence))

        # Standard PPE (helmet, vest, harness, boots)
        if not is_small_ppe:
            if raw_present and confidence >= self.uncertainty_threshold:
                item_state["absent_streak"] = 0
                item_state["last_status"] = True
                return True
            else:
                item_state["absent_streak"] += 1
                # Confirm missing after 3 frames of absence
                if item_state["absent_streak"] >= 3:
                    item_state["last_status"] = False
                    return False
                return "uncertain"

        # Small & Hard-to-Detect PPE logic (goggles, earmuff, chaps pants)
        if raw_present:
            if confidence >= 0.40:
                item_state["absent_streak"] = 0
                item_state["last_status"] = True
                return True
            else:
                # Borderline positive confidence -> uncertain
                return "uncertain"
        else:
            item_state["absent_streak"] += 1

            # Condition 1: Explicit high-confidence negative detection (e.g. bare face/ears > 0.60)
            if confidence >= self.high_conf_missing_threshold:
                item_state["last_status"] = False
                return False

            # Condition 2: Consistently absent across N frames
            if item_state["absent_streak"] >= self.min_absence_frames:
                item_state["last_status"] = False
                return False

            # Otherwise, transient absence or low confidence -> uncertain (grey '?')
            item_state["last_status"] = "uncertain"
            return "uncertain"

    def prune_stale_workers(self, active_worker_ids: List[str]) -> None:
        """Removes tracked history for workers no longer visible."""
        current_set = set(active_worker_ids)
        stale = [wid for wid in self.worker_history if wid not in current_set]
        for wid in stale:
            del self.worker_history[wid]


class HighResZoneInference:
    """
    Second-pass zoomed inference pipeline:
    Crops body zones (e.g. head, hands) and runs inference at high resolution
    so micro-scale PPE (goggles, earmuffs) can be resolved accurately.
    """
    def __init__(self, detector_model: Any = None, crop_target_size: int = 640):
        self.model = detector_model
        self.target_size = crop_target_size

    def crop_body_zone(
        self,
        frame: np.ndarray,
        zone_box: List[float],
        padding_ratio: float = 0.15
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Safely crops zone with padding from frame.
        Returns (cropped_bgr, (crop_x1, crop_y1, crop_w, crop_h)).
        """
        if frame is None or frame.size == 0:
            return None, (0, 0, 0, 0)

        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = zone_box
        zw = zx2 - zx1
        zh = zy2 - zy1

        # Apply padding
        pad_x = zw * padding_ratio
        pad_y = zh * padding_ratio

        cx1 = max(0, int(zx1 - pad_x))
        cy1 = max(0, int(zy1 - pad_y))
        cx2 = min(w, int(zx2 + pad_x))
        cy2 = min(h, int(zy2 + pad_y))

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        if crop_w < 10 or crop_h < 10:
            return None, (cx1, cy1, crop_w, crop_h)

        crop = frame[cy1:cy2, cx1:cx2]
        return crop, (cx1, cy1, crop_w, crop_h)

    def run_zoomed_inference(
        self,
        frame: np.ndarray,
        zone_box: List[float],
        target_categories: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Executes high-resolution zoomed inference on a cropped zone.
        Re-maps detected bboxes back into full-frame coordinates.
        """
        if self.model is None:
            return []

        crop, (cx1, cy1, cw, ch) = self.crop_body_zone(frame, zone_box)
        if crop is None:
            return []

        # Run model on crop resized or natively upscaled to target_size (e.g. 640)
        try:
            results = self.model.predict(
                crop,
                imgsz=self.target_size,
                conf=0.20,
                verbose=False
            )
        except Exception:
            return []

        refined_detections = []
        scale_x = cw / float(crop.shape[1])
        scale_y = ch / float(crop.shape[0])

        for res in results:
            boxes = res.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cat_name = res.names.get(cls_id, str(cls_id))

                if cat_name in target_categories:
                    # Normalized or absolute coords in crop
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    # Re-project to full image
                    proj_x1 = cx1 + (bx1 * scale_x)
                    proj_y1 = cy1 + (by1 * scale_y)
                    proj_x2 = cx1 + (bx2 * scale_x)
                    proj_y2 = cy1 + (by2 * scale_y)

                    refined_detections.append({
                        "category": cat_name,
                        "confidence": round(conf, 3),
                        "bbox": [proj_x1, proj_y1, proj_x2, proj_y2],
                        "source": "high_res_crop"
                    })

        return refined_detections
