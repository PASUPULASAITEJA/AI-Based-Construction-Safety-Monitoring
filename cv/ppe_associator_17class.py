"""
Multi-Zone Anatomical PPE-to-Worker Spatial Associator
Maps 17 YOLOv8 detection classes to worker tracking anchors.
"""

from typing import List, Dict, Any
import numpy as np

# Anatomical height partitions relative to person box [ymin, ymax]
BODY_ZONES = {
    "HEAD": (0.00, 0.28),
    "TORSO": (0.22, 0.72),
    "LOWER": (0.65, 1.00),
    "HANDS": (0.35, 0.85)  # Lateral boundaries checked in code
}

PPE_CATEGORY_CONFIG = {
    "helmet":        {"zone": "HEAD",  "pair": "helmet",       "is_positive": True},
    "no-helmet":     {"zone": "HEAD",  "pair": "helmet",       "is_positive": False},
    "goggles":       {"zone": "HEAD",  "pair": "goggles",      "is_positive": True},
    "no-goggles":    {"zone": "HEAD",  "pair": "goggles",      "is_positive": False},
    "earmuff":       {"zone": "HEAD",  "pair": "earmuff",      "is_positive": True},
    "no-earmuff":    {"zone": "HEAD",  "pair": "earmuff",      "is_positive": False},
    "vest":          {"zone": "TORSO", "pair": "vest",         "is_positive": True},
    "no-vest":       {"zone": "TORSO", "pair": "vest",         "is_positive": False},
    "harness":       {"zone": "TORSO", "pair": "harness",      "is_positive": True},
    "no-harness":    {"zone": "TORSO", "pair": "harness",      "is_positive": False},
    "safety-pants":  {"zone": "LOWER", "pair": "safety_pants", "is_positive": True},
    "no-safety-pants":{"zone": "LOWER", "pair": "safety_pants", "is_positive": False},
    "boots":         {"zone": "LOWER", "pair": "boots",        "is_positive": True},
    "no-boots":      {"zone": "LOWER", "pair": "boots",        "is_positive": False},
    "gloves":        {"zone": "HANDS", "pair": "gloves",       "is_positive": True},
    "no-gloves":     {"zone": "HANDS", "pair": "gloves",       "is_positive": False},
}

ALL_PPE_ITEMS = ["helmet", "goggles", "earmuff", "vest", "harness", "safety_pants", "gloves", "boots"]

class PPEAssociator:
    def __init__(self, min_containment_threshold: float = 0.20):
        self.min_containment = min_containment_threshold

    @staticmethod
    def _get_zone_bbox(person_bbox: List[float], zone_name: str) -> List[float]:
        px1, py1, px2, py2 = person_bbox
        ph = py2 - py1
        pw = px2 - px1
        
        y_min_ratio, y_max_ratio = BODY_ZONES[zone_name]
        zy1 = py1 + (ph * y_min_ratio)
        zy2 = py1 + (ph * y_max_ratio)
        
        if zone_name == "HANDS":
            # Expand slightly outwards horizontally where hands typically hang
            zx1 = px1 - (pw * 0.15)
            zx2 = px2 + (pw * 0.15)
        else:
            zx1 = px1
            zx2 = px2
            
        return [zx1, zy1, zx2, zy2]

    @staticmethod
    def _box_intersection_ratio(box_a: List[float], box_b: List[float]) -> float:
        """Calculates area of intersection relative to box_a (PPE box)."""
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection_area = iw * ih

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        if area_a <= 0:
            return 0.0
        return intersection_area / area_a

    def associate(self, tracked_workers: List[Dict[str, Any]], detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        tracked_workers: List of worker dicts from tracker: [{'worker_id': 1, 'bbox': [x1,y1,x2,y2]}, ...]
        detections: All frame detections: [{'category': 'vest', 'confidence': 0.88, 'bbox': [...]}, ...]
        """
        results = []

        # Filter out anchor person detections to process PPE items only
        ppe_detections = [d for d in detections if d["category"] != "person" and d["category"] in PPE_CATEGORY_CONFIG]

        for worker in tracked_workers:
            w_id = worker["worker_id"]
            p_bbox = worker["bbox"]

            # Initialize worker state dictionary
            worker_state = {
                "worker_id": f"W-{w_id:03d}",
                "raw_id": w_id,
                "bbox": p_bbox,
                "confidence": worker.get("confidence", 1.0),
                "helmet": None,
                "goggles": None,
                "earmuff": None,
                "vest": None,
                "harness": None,
                "safety_pants": None,
                "gloves": None,
                "boots": None,
                "ppe_confidences": {},
                "missing_ppe": [],
                "compliance_score": 0.0
            }

            # Map detections to this worker based on zone intersection
            for ppe in ppe_detections:
                cat = ppe["category"]
                conf = ppe["confidence"]
                p_box = ppe["bbox"]
                cfg = PPE_CATEGORY_CONFIG[cat]
                zone_name = cfg["zone"]
                pair_key = cfg["pair"]
                is_positive = cfg["is_positive"]

                zone_bbox = self._get_zone_bbox(p_bbox, zone_name)
                containment = self._box_intersection_ratio(p_box, zone_bbox)

                if containment >= self.min_containment:
                    current_conf = worker_state["ppe_confidences"].get(pair_key, 0.0)
                    # Higher confidence detection takes precedence
                    if conf > current_conf:
                        worker_state[pair_key] = is_positive
                        worker_state["ppe_confidences"][pair_key] = conf

            # Compile missing PPE items & compliance score
            present_count = 0
            evaluated_count = 0

            for item in ALL_PPE_ITEMS:
                val = worker_state[item]
                if val is True:
                    present_count += 1
                    evaluated_count += 1
                elif val is False:
                    worker_state["missing_ppe"].append(item)
                    evaluated_count += 1
                else:
                    # Item not explicitly detected present or absent (e.g. occluded / not in view)
                    worker_state["missing_ppe"].append(item)
                    evaluated_count += 1

            worker_state["compliance_score"] = round(present_count / float(len(ALL_PPE_ITEMS)), 2)
            results.append(worker_state)

        return results
