"""
AI-Based Construction Safety Monitoring — PPE-to-Worker Association
===================================================================
Associates 17-class YOLOv8 object detections with tracked worker anchors
using anatomical body-zone spatial decomposition and distance metrics.

8 PPE Categories:
  1. helmet        / no-helmet
  2. goggles       / no-goggles
  3. earmuff       / no-earmuff
  4. vest          / no-vest
  5. safety_pants  / no-safety-pants
  6. gloves        / no-gloves
  7. boots         / no-boots
  8. harness       / no-harness
  + person (worker anchor)

Author: Construction CV Safety Team
"""

import math
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np


# 8 Standard PPE Categories and their present/absent class name aliases
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

CATEGORY_PAIRS = {
    "helmet": {"pos": ["helmet"], "neg": ["no-helmet", "no_helmet", "head"]},
    "goggles": {"pos": ["goggles", "glasses"], "neg": ["no-goggles", "no_goggles", "no-glasses", "face"]},
    "earmuff": {"pos": ["earmuff", "earmuffs", "ear-mufs"], "neg": ["no-earmuff", "no_earmuff", "ear"]},
    "vest": {"pos": ["vest", "safety-vest"], "neg": ["no-vest", "no_vest", "no-safety-vest"]},
    "safety_pants": {"pos": ["safety-pants", "safety_pants", "safety-suit"], "neg": ["no-safety-pants", "no_safety_pants", "no-pants"]},
    "gloves": {"pos": ["gloves", "glove"], "neg": ["no-gloves", "no_gloves", "hands", "hand"]},
    "boots": {"pos": ["boots", "safety-boots", "shoes"], "neg": ["no-boots", "no_boots", "foot", "no-shoes"]},
    "harness": {"pos": ["harness", "safety-harness"], "neg": ["no-harness", "no_harness"]}
}

# Reverse mapping from detected label -> (category, is_positive)
LABEL_LOOKUP = {}
for cat, pairs in CATEGORY_PAIRS.items():
    for label in pairs["pos"]:
        LABEL_LOOKUP[label.lower()] = (cat, True)
    for label in pairs["neg"]:
        LABEL_LOOKUP[label.lower()] = (cat, False)

# 17-Class ID to canonical name mapping
ID_TO_LABEL = {
    0: "person",
    1: "helmet",
    2: "no-helmet",
    3: "goggles",
    4: "no-goggles",
    5: "earmuff",
    6: "no-earmuff",
    7: "vest",
    8: "no-vest",
    9: "safety-pants",
    10: "no-safety-pants",
    11: "gloves",
    12: "no-gloves",
    13: "boots",
    14: "no-boots",
    15: "harness",
    16: "no-harness"
}


def compute_box_overlap(inner_box: List[float], outer_box: List[float]) -> float:
    """Computes the fraction of inner_box that is inside outer_box [0.0, 1.0]."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    inner_area = max(1e-5, (ix2 - ix1) * (iy2 - iy1))
    return float(inter_area / inner_area)


def compute_box_center(box: List[float]) -> Tuple[float, float]:
    """Returns center point (cx, cy) of bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def is_point_inside(point: Tuple[float, float], box: List[float]) -> bool:
    """Checks if point (x, y) is inside bounding box [x1, y1, x2, y2]."""
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


class PPEAssociator:
    """
    Anatomical Body-Zone PPE-to-Worker Association Engine.
    
    Partitions worker bounding boxes into:
      - HEAD ZONE  (top 25% of person box): helmet, goggles, earmuff
      - TORSO ZONE (25%–70% of person box): vest, harness
      - LOWER ZONE (70%–100% of person box): safety-pants, boots
      - HANDS ZONE (lateral edges of person box): gloves
    """
    def __init__(
        self,
        min_overlap: float = 0.05,
        min_detection_confidence: float = 0.25,
        head_span: Tuple[float, float] = (0.00, 0.25),
        torso_span: Tuple[float, float] = (0.25, 0.70),
        lower_span: Tuple[float, float] = (0.70, 1.00)
    ):
        self.min_overlap = min_overlap
        self.min_conf = min_detection_confidence
        self.head_span = head_span
        self.torso_span = torso_span
        self.lower_span = lower_span

    # -------------------------------------------------------------------------
    # Body Zone Generators
    # -------------------------------------------------------------------------
    def get_head_zone(self, person_box: List[float]) -> List[float]:
        """HEAD ZONE: top 25% of person bounding box, with top/lateral margins."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        
        # Extend slightly upwards (-15% ph) for hardhat crown and sideways for ear protection
        hy1 = py1 - 0.15 * ph
        hy2 = py1 + self.head_span[1] * ph
        hx1 = px1 - 0.10 * pw
        hx2 = px2 + 0.10 * pw
        return [hx1, hy1, hx2, hy2]

    def get_torso_zone(self, person_box: List[float]) -> List[float]:
        """TORSO ZONE: 25%–70% of person box (vest, safety harness)."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        
        ty1 = py1 + self.torso_span[0] * ph
        ty2 = py1 + self.torso_span[1] * ph
        tx1 = px1 - 0.08 * pw
        tx2 = px2 + 0.08 * pw
        return [tx1, ty1, tx2, ty2]

    def get_lower_zone(self, person_box: List[float]) -> List[float]:
        """LOWER ZONE: 70%–100% of person box (safety pants, boots)."""
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        
        ly1 = py1 + self.lower_span[0] * ph
        ly2 = py2 + 0.08 * ph # Slight downward margin for shoes hitting ground
        lx1 = px1 - 0.12 * pw
        lx2 = px2 + 0.12 * pw
        return [lx1, ly1, lx2, ly2]

    def get_hands_zone(self, person_box: List[float]) -> List[List[float]]:
        """
        HANDS ZONE: Regions near person width edges (left & right flanks)
        spanning between waist and upper thigh (30% to 80% height).
        """
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        
        hy1 = py1 + 0.30 * ph
        hy2 = py1 + 0.85 * ph
        
        # Left hand lateral flank
        left_hand = [px1 - 0.25 * pw, hy1, px1 + 0.35 * pw, hy2]
        # Right hand lateral flank
        right_hand = [px2 - 0.35 * pw, hy1, px2 + 0.25 * pw, hy2]
        return [left_hand, right_hand]

    def get_target_zone_for_category(self, person_box: List[float], category: str) -> List[Any]:
        """Maps PPE category to corresponding body zone bbox(es)."""
        if category in ["helmet", "goggles", "earmuff"]:
            return [self.get_head_zone(person_box)]
        elif category in ["vest", "harness"]:
            return [self.get_torso_zone(person_box)]
        elif category in ["safety_pants", "boots"]:
            return [self.get_lower_zone(person_box)]
        elif category == "gloves":
            return self.get_hands_zone(person_box)
        else:
            return [person_box]

    # -------------------------------------------------------------------------
    # Spatial Proximity & Scoring
    # -------------------------------------------------------------------------
    def calculate_association_score(
        self,
        ppe_box: List[float],
        person_box: List[float],
        category: str
    ) -> float:
        """
        Computes spatial affinity score between a PPE item and a candidate worker.
        Higher score = better match. Returns 0.0 if item is outside anatomical bounds.
        """
        candidate_zones = self.get_target_zone_for_category(person_box, category)
        ppe_cx, ppe_cy = compute_box_center(ppe_box)

        best_score = 0.0
        for zone in candidate_zones:
            overlap = compute_box_overlap(ppe_box, zone)
            inside = is_point_inside((ppe_cx, ppe_cy), zone)

            if inside or overlap >= self.min_overlap:
                zone_cx, zone_cy = compute_box_center(zone)
                pw = max(1.0, person_box[2] - person_box[0])
                ph = max(1.0, person_box[3] - person_box[1])

                # Normalized Euclidean distance to zone center
                norm_dx = (ppe_cx - zone_cx) / pw
                norm_dy = (ppe_cy - zone_cy) / ph
                norm_dist = math.sqrt(norm_dx**2 + norm_dy**2)

                # Affinity: High overlap, low normalized distance
                score = (overlap * 0.6) + max(0.0, 1.0 - norm_dist) * 0.4
                if inside:
                    score += 0.25
                best_score = max(best_score, score)

        return best_score

    # -------------------------------------------------------------------------
    # Main Association Logic (with Temporal & Uncertainty Resolution)
    # -------------------------------------------------------------------------
    def associate(
        self,
        detections: Any = None,
        workers: Optional[Any] = None,
        frame: Optional[np.ndarray] = None,
        temporal_tracker: Optional[Any] = None,
        **kwargs
    ) -> Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Associates detected bounding boxes for all 17 classes with workers.
        Supports both signatures:
          - associate(detections, workers=None) -> returns Dict[worker_id, worker_dict]
          - associate(workers, detections, frame=None) -> returns List[enriched_worker_dict]
        """
        return_as_list = False

        # Detect caller signature: if detections is actually workers list and workers is detections
        if isinstance(detections, list) and isinstance(workers, list):
            if len(detections) > 0 and ("worker_id" in detections[0] or "label" in detections[0]) and "category" not in detections[0]:
                actual_workers = detections
                actual_detections = workers
                return_as_list = True
            else:
                actual_detections = detections
                actual_workers = workers
        elif isinstance(detections, list) and workers is None:
            actual_detections = detections
            actual_workers = None
        else:
            actual_detections = detections or []
            actual_workers = workers

        detections = actual_detections
        workers = actual_workers

        # Step 1: Normalize detections to standardized category and polarity
        parsed_detections = []
        parsed_workers = []

        for det in detections:
            bbox = [float(v) for v in det["bbox"]]
            conf = float(det.get("confidence", 1.0))
            raw_cat = det.get("category", det.get("label", det.get("class_id", "")))

            # Convert class_id integer to label name if necessary
            if isinstance(raw_cat, int) and raw_cat in ID_TO_LABEL:
                cat_str = ID_TO_LABEL[raw_cat].lower()
            else:
                cat_str = str(raw_cat).lower().strip()

            if cat_str in ["person", "worker"]:
                parsed_workers.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "worker_id": det.get("worker_id", None)
                })
            else:
                lookup = LABEL_LOOKUP.get(cat_str)
                if lookup:
                    cat_name, is_pos = lookup
                    parsed_detections.append({
                        "category": cat_name,
                        "is_positive": is_pos,
                        "bbox": bbox,
                        "confidence": conf,
                        "raw_label": cat_str
                    })

        # Step 2: Assemble worker list
        if workers is not None and len(workers) > 0:
            active_workers = []
            for i, w in enumerate(workers):
                wid = w.get("worker_id")
                if wid is None:
                    wid = f"W-{i+1:03d}"
                elif isinstance(wid, int):
                    wid = f"W-{wid:03d}"
                active_workers.append({
                    "worker_id": str(wid),
                    "bbox": [float(v) for v in w["bbox"]],
                    "confidence": float(w.get("confidence", 1.0)),
                    "original_worker": w
                })
        else:
            active_workers = []
            for i, pw in enumerate(parsed_workers):
                wid = pw["worker_id"]
                if wid is None:
                    wid = f"W-{i+1:03d}"
                elif isinstance(wid, int):
                    wid = f"W-{wid:03d}"
                active_workers.append({
                    "worker_id": str(wid),
                    "bbox": pw["bbox"],
                    "confidence": pw["confidence"],
                    "original_worker": None
                })

        # Initialize results structure for all active workers
        results: Dict[str, Dict[str, Any]] = {}
        worker_ppe_state: Dict[str, Dict[str, Dict[str, float]]] = {}

        for w in active_workers:
            wid = w["worker_id"]
            results[wid] = {
                "worker_id": wid,
                "label": f"Worker #{wid}" if not wid.startswith("Worker") else wid,
                "bbox": w["bbox"],
                "confidence": round(w["confidence"], 3),
                "helmet": False,
                "goggles": False,
                "earmuff": False,
                "vest": False,
                "harness": False,
                "safety_pants": False,
                "gloves": False,
                "boots": False,
                "compliance_score": 0.0,
                "missing_ppe": [],
                "associated_items": {}
            }
            worker_ppe_state[wid] = {
                cat: {"pos_conf": 0.0, "neg_conf": 0.0, "best_box": None}
                for cat in PPE_CATEGORIES
            }

        if not active_workers or not parsed_detections:
            for wid, item in results.items():
                item["missing_ppe"] = list(PPE_CATEGORIES)
                item["compliance_score"] = 0.0
            if return_as_list:
                return list(results.values())
            return results

        # Step 3: Associate each PPE detection to the best matching worker
        for ppe in parsed_detections:
            cat = ppe["category"]
            ppe_box = ppe["bbox"]
            conf = ppe["confidence"]
            is_pos = ppe["is_positive"]

            best_wid = None
            best_score = 0.0

            for w in active_workers:
                wid = w["worker_id"]
                score = self.calculate_association_score(ppe_box, w["bbox"], cat)
                if score > best_score:
                    best_score = score
                    best_wid = wid

            if best_wid and best_score >= 0.15:
                state = worker_ppe_state[best_wid][cat]
                if is_pos:
                    if conf > state["pos_conf"]:
                        state["pos_conf"] = conf
                        state["best_box"] = ppe_box
                else:
                    if conf > state["neg_conf"]:
                        state["neg_conf"] = conf

        # Step 4: Resolve compliance status (with Task E temporal / uncertainty logic)
        for wid, state_dict in worker_ppe_state.items():
            compliant_count = 0
            missing_list = []

            for cat in PPE_CATEGORIES:
                pos_c = state_dict[cat]["pos_conf"]
                neg_c = state_dict[cat]["neg_conf"]

                raw_present = (pos_c >= self.min_conf) and (pos_c >= neg_c)
                eff_conf = pos_c if raw_present else max(neg_c, 0.0)

                if temporal_tracker:
                    from cv.small_ppe_handler import SMALL_PPE_CATEGORIES
                    is_small = cat in SMALL_PPE_CATEGORIES
                    status = temporal_tracker.update_and_resolve(
                        worker_id=wid,
                        category=cat,
                        raw_present=raw_present,
                        confidence=eff_conf,
                        is_small_ppe=is_small
                    )
                else:
                    status = raw_present

                results[wid][cat] = status
                if status is True:
                    compliant_count += 1
                elif status is False:
                    missing_list.append(cat)
                # Uncertain does not penalize score completely as confirmed violation
                elif status == "uncertain":
                    compliant_count += 0.5

            score = round(compliant_count / float(len(PPE_CATEGORIES)), 2)
            results[wid]["compliance_score"] = score
            results[wid]["missing_ppe"] = missing_list

        if return_as_list:
            enriched = []
            for w in active_workers:
                wid = w["worker_id"]
                orig = w.get("original_worker") or {}
                combined = {**orig, **results[wid]}
                enriched.append(combined)
            return enriched

        return results


# -----------------------------------------------------------------------------
# Quick Verification / Self-Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    associator = PPEAssociator()

    # Synthetic worker: [x1, y1, x2, y2]
    # Height = 400px (100 to 500), Width = 150px (200 to 350)
    worker_anchor = {"worker_id": "W-001", "bbox": [200, 100, 350, 500], "confidence": 0.95}

    synthetic_detections = [
        # Person
        {"category": "person", "bbox": [200, 100, 350, 500], "confidence": 0.95},
        # Head zone (y: 100 to 200) -> Helmet present
        {"category": "helmet", "bbox": [220, 95, 330, 160], "confidence": 0.91},
        # Head zone -> Goggles missing (detected no-goggles/face)
        {"category": "no-goggles", "bbox": [240, 140, 310, 175], "confidence": 0.82},
        # Torso zone (y: 200 to 380) -> Vest present, harness missing
        {"category": "vest", "bbox": [210, 210, 340, 360], "confidence": 0.88},
        {"category": "no-harness", "bbox": [220, 220, 330, 350], "confidence": 0.75},
        # Lower zone (y: 380 to 500) -> Safety pants present, boots present
        {"category": "safety-pants", "bbox": [215, 360, 335, 460], "confidence": 0.84},
        {"category": "boots", "bbox": [220, 460, 330, 505], "confidence": 0.89},
        # Lateral hand zone -> Gloves present
        {"category": "gloves", "bbox": [180, 280, 225, 330], "confidence": 0.79}
    ]

    res = associator.associate(synthetic_detections, workers=[worker_anchor])
    import json
    print(json.dumps(res, indent=2))
