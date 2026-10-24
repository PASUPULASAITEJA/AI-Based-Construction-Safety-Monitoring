"""
Hierarchical Two-Stage Computer Vision Detector with Zoom-Crop Secondary Inference
Specialized for detecting micro-PPE (goggles, earmuffs, gloves) reliably.
"""

from typing import Dict, Any, List
import cv2
import numpy as np
from collections import defaultdict

class MicroPPETracker:
    def __init__(self, min_absent_frames: int = 15, high_conf_thresh: float = 0.60, low_conf_thresh: float = 0.25):
        self.min_absent_frames = min_absent_frames
        self.high_conf = high_conf_thresh
        self.low_conf = low_conf_thresh
        # worker_id -> item -> consecutive absent counter
        self.absence_counters = defaultdict(lambda: defaultdict(int))

    def evaluate_item_state(self, worker_id: str, item_name: str, detected_conf: float) -> Any:
        """
        Returns:
            True: Confirmed present
            False: Confirmed absent (passed consecutive frame temporal threshold)
            None: Uncertain (low confidence or transitional state)
        """
        if detected_conf >= self.high_conf:
            self.absence_counters[worker_id][item_name] = 0
            return True

        if detected_conf <= self.low_conf:
            self.absence_counters[worker_id][item_name] += 1
            if self.absence_counters[worker_id][item_name] >= self.min_absent_frames:
                return False
            # Still within grace period
            return None

        # Moderate confidence (0.25 <= conf < 0.60): treat as uncertain
        return None

class HierarchicalPPEDetector:
    def __init__(self, primary_yolo_model, micro_yolo_model=None):
        """
        primary_yolo_model: Detects person + large PPE (vest, harness, boots, helmet)
        micro_yolo_model: Optional specialized or zoomed model for goggles, earmuffs, gloves
        """
        self.primary_model = primary_yolo_model
        self.micro_model = micro_yolo_model or primary_yolo_model
        self.tracker = MicroPPETracker()

    def detect_frame_hierarchical(self, frame_bgr: np.ndarray, tracked_workers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fh, fw = frame_bgr.shape[:2]

        for worker in tracked_workers:
            x1, y1, x2, y2 = [int(v) for v in worker["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            pw = x2 - x1
            ph = y2 - y1

            if pw <= 10 or ph <= 10:
                continue

            # Stage 2 Zoom-Crop: Extract Head Zone for Goggles & Earmuffs
            hy2 = min(fh, y1 + int(ph * 0.30))
            head_crop = frame_bgr[y1:hy2, x1:x2]

            if head_crop.size > 0:
                # Resize crop to high resolution for small-feature detection
                head_crop_resized = cv2.resize(head_crop, (640, 640), interpolation=cv2.INTER_CUBIC)
                crop_results = self.micro_model.predict(head_crop_resized, conf=0.20, verbose=False)[0]

                goggles_conf = 0.0
                earmuff_conf = 0.0

                for box in crop_results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.micro_model.names.get(cls_id, "")
                    conf = float(box.conf[0])

                    if cls_name == "goggles":
                        goggles_conf = max(goggles_conf, conf)
                    elif cls_name == "earmuff":
                        earmuff_conf = max(earmuff_conf, conf)

                # Apply temporal hysteresis
                w_id = worker["worker_id"]
                worker["goggles"] = self.tracker.evaluate_item_state(w_id, "goggles", goggles_conf)
                worker["earmuff"] = self.tracker.evaluate_item_state(w_id, "earmuff", earmuff_conf)
                worker["ppe_confidences"]["goggles"] = goggles_conf
                worker["ppe_confidences"]["earmuff"] = earmuff_conf

        return tracked_workers
