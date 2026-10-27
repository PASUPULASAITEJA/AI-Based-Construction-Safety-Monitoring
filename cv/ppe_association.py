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
import cv2


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


def _crop(frame: np.ndarray, box: List[float]) -> Optional[np.ndarray]:
    ih, iw = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(iw, int(box[2])), min(ih, int(box[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def _non_skin_mask(region: np.ndarray) -> np.ndarray:
    # Skin in YCrCb; tanned skin otherwise passes as "orange" under warm light
    skin = cv2.inRange(cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127))
    return cv2.bitwise_not(skin)


def hivis_ratio(region: Optional[np.ndarray]) -> float:
    """Fraction of pixels that are fluorescent yellow/green or orange (and not skin)."""
    if region is None or region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Fluorescent yellow-green measured at H 30-45; plain yellow shirts sit at H 21-23
    yellow_green = cv2.inRange(hsv, (27, 120, 150), (55, 255, 255))
    orange = cv2.inRange(hsv, (3, 150, 130), (20, 255, 255))
    mask = cv2.bitwise_and(cv2.bitwise_or(yellow_green, orange), _non_skin_mask(region))
    return cv2.countNonZero(mask) / float(region.shape[0] * region.shape[1])


def hardhat_ratio(region: Optional[np.ndarray]) -> float:
    """Fraction of pixels with a typical hard-hat colour (yellow/orange/white/red/blue/green, not skin/hair)."""
    if region is None or region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (15, 70, 120), (35, 255, 255)),    # yellow (incl. faded/dusty shells)
        cv2.inRange(hsv, (5, 120, 120), (15, 255, 255)),    # orange
        cv2.inRange(hsv, (0, 0, 185), (180, 40, 255)),      # white
        cv2.inRange(hsv, (0, 120, 90), (5, 255, 255)),      # red
        cv2.inRange(hsv, (170, 120, 90), (180, 255, 255)),  # red (wrap)
        cv2.inRange(hsv, (95, 90, 70), (130, 255, 255)),    # blue
        cv2.inRange(hsv, (40, 90, 70), (85, 255, 255)),     # green
    ]
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    mask = cv2.bitwise_and(mask, _non_skin_mask(region))
    return cv2.countNonZero(mask) / float(region.shape[0] * region.shape[1])


def weak_box_colour_score(frame: Optional[np.ndarray], category: str, box: List[float]) -> float:
    """
    Colour verification of a low-confidence helmet/vest box. Returns 0.0 if its pixels
    don't look like the real thing, otherwise a confidence for the verified item (same
    scale as verify_anatomical_ppe_presence: 0.65 + ratio * 1.5, capped).
    """
    if frame is None:
        return 0.0
    if category == "vest":
        ratio = hivis_ratio(_crop(frame, box))
        return round(min(0.98, 0.65 + ratio * 1.5), 3) if ratio >= 0.15 else 0.0
    if category == "helmet":
        x1, y1, x2, y2 = box
        top = [x1, y1, x2, y1 + 0.6 * (y2 - y1)]  # shell, not the face below it
        ratio = hardhat_ratio(_crop(frame, top))
        return round(min(0.95, 0.60 + ratio * 0.5), 3) if ratio >= 0.30 else 0.0
    return 0.0


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
        min_detection_confidence: float = 0.15,  # same as the detector's helmet/vest thresholds
        head_span: Tuple[float, float] = (0.00, 0.25),
        torso_span: Tuple[float, float] = (0.25, 0.70),
        lower_span: Tuple[float, float] = (0.70, 1.00),
        required_ppe: Optional[List[str]] = None,
        cv_fallback_categories: Tuple[str, ...] = ("vest",)
    ):
        if required_ppe is None:
            import config
            required_ppe = getattr(config, "REQUIRED_PPE", ["helmet", "vest"])
        self.required_ppe = list(required_ppe)
        # Only the hi-vis vest colour check is reliable enough to override the model; the
        # edge/colour heuristics for helmet (any white wall), harness, boots, goggles etc.
        # marked people as wearing PPE they did not have.
        self.cv_fallback_categories = set(cv_fallback_categories)
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

    get_head_box = get_head_zone
    get_torso_box = get_torso_zone
    get_lower_box = get_lower_zone

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

    def verify_anatomical_ppe_presence(self, frame: np.ndarray, person_box: List[float], category: str) -> Tuple[bool, float]:
        """
        Anatomical Computer Vision Verification for all 8 PPE categories.
        Analyzes colorimetry, contrast, texture, and contours in specific body zones.
        Ensures high reliability across all 8 PPE categories even when neural weights lack a class.
        """
        if frame is None or frame.size == 0:
            return False, 0.0

        ih, iw = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(iw, px2), min(ih, py2)
        pw, ph = px2 - px1, py2 - py1

        if pw < 10 or ph < 20:
            return False, 0.0

        p_crop = frame[py1:py2, px1:px2]
        if p_crop.size == 0:
            return False, 0.0

        try:
            # 1. High-Visibility Safety Vest (Torso zone: 20% to 70% height)
            if category == "vest":
                ty1, ty2 = int(ph * 0.20), int(ph * 0.70)
                torso = p_crop[ty1:ty2, :]
                if torso.size == 0:
                    return False, 0.0
                ratio = hivis_ratio(torso)
                if ratio >= 0.10:
                    return True, round(min(0.98, 0.65 + ratio * 1.5), 3)
                return False, 0.0

            # 2. Hardhat / Safety Helmet (Head zone: top 25% height)
            elif category == "helmet":
                hy1, hy2 = 0, int(ph * 0.25)
                head = p_crop[hy1:hy2, :]
                if head.size == 0:
                    return False, 0.0
                hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
                mask_y = cv2.inRange(hsv, (18, 60, 100), (45, 255, 255))
                mask_o = cv2.inRange(hsv, (5, 90, 110), (18, 255, 255))
                mask_w = cv2.inRange(hsv, (0, 0, 175), (180, 45, 255))
                mask_b = cv2.inRange(hsv, (95, 70, 70), (130, 255, 255))
                mask_r = cv2.inRange(hsv, (0, 100, 100), (6, 255, 255))
                tot = cv2.countNonZero(mask_y) + cv2.countNonZero(mask_o) + cv2.countNonZero(mask_w) + cv2.countNonZero(mask_b) + cv2.countNonZero(mask_r)
                ratio = tot / float(head.shape[0] * head.shape[1])
                if ratio >= 0.12:
                    return True, round(min(0.95, 0.60 + ratio * 1.2), 3)
                return False, 0.0

            # 3. Protective Pants / Trousers (Leg zone: 60% to 92% height)
            elif category == "safety_pants":
                ly1, ly2 = int(ph * 0.60), int(ph * 0.92)
                legs = p_crop[ly1:ly2, :]
                if legs.size == 0:
                    return False, 0.0
                ycrcb = cv2.cvtColor(legs, cv2.COLOR_BGR2YCrCb)
                skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
                skin_ratio = cv2.countNonZero(skin_mask) / float(legs.shape[0] * legs.shape[1])
                gray = cv2.cvtColor(legs, cv2.COLOR_BGR2GRAY)
                std_dev = float(np.std(gray))
                if skin_ratio < 0.20 and std_dev > 10:
                    return True, 0.86
                return False, 0.0

            # 4. Safety Boots (Foot zone: 88% to 100% height)
            elif category == "boots":
                fy1, fy2 = int(ph * 0.88), ph
                feet = p_crop[fy1:fy2, :]
                if feet.size == 0:
                    return False, 0.0
                gray = cv2.cvtColor(feet, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 40, 120)
                edge_ratio = cv2.countNonZero(edges) / float(feet.shape[0] * feet.shape[1])
                if edge_ratio >= 0.05:
                    return True, 0.82
                return False, 0.0

            # 5. Work Gloves (Hands zone: flanks 45% to 80% height)
            elif category == "gloves":
                hy1, hy2 = int(ph * 0.45), int(ph * 0.80)
                lx1, lx2 = 0, int(pw * 0.35)
                rx1, rx2 = int(pw * 0.65), pw
                l_hand = p_crop[hy1:hy2, lx1:lx2]
                r_hand = p_crop[hy1:hy2, rx1:rx2]
                detected = False
                for h_img in [l_hand, r_hand]:
                    if h_img.size > 0:
                        hsv = cv2.cvtColor(h_img, cv2.COLOR_BGR2HSV)
                        glove_mask = cv2.inRange(hsv, (5, 60, 60), (45, 255, 255))
                        if cv2.countNonZero(glove_mask) / float(h_img.shape[0] * h_img.shape[1]) >= 0.10:
                            detected = True
                            break
                if detected:
                    return True, 0.80
                return False, 0.0

            # 6. Safety Harness (Torso strapping: 25% to 65% height)
            elif category == "harness":
                ty1, ty2 = int(ph * 0.25), int(ph * 0.65)
                torso = p_crop[ty1:ty2, :]
                if torso.size == 0:
                    return False, 0.0
                gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 60, 150)
                edge_density = cv2.countNonZero(edges) / float(torso.shape[0] * torso.shape[1])
                if edge_density >= 0.08:
                    return True, 0.78
                return False, 0.0

            # 7. Eye Protection / Goggles (Ocular span: 8% to 18% height)
            elif category == "goggles":
                ey1, ey2 = int(ph * 0.08), int(ph * 0.18)
                eyes = p_crop[ey1:ey2, :]
                if eyes.size == 0:
                    return False, 0.0
                gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 140)
                if cv2.countNonZero(edges) / float(eyes.shape[0] * eyes.shape[1]) >= 0.08:
                    return True, 0.75
                return False, 0.0

            # 8. Ear Protection / Earmuffs (Head lateral flanks: 8% to 22% height)
            elif category == "earmuff":
                my1, my2 = int(ph * 0.08), int(ph * 0.22)
                lx1, lx2 = 0, int(pw * 0.25)
                rx1, rx2 = int(pw * 0.75), pw
                l_ear = p_crop[my1:my2, lx1:lx2]
                r_ear = p_crop[my1:my2, rx1:rx2]
                for e_img in [l_ear, r_ear]:
                    if e_img.size > 0:
                        hsv = cv2.cvtColor(e_img, cv2.COLOR_BGR2HSV)
                        cup_mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
                        if cv2.countNonZero(cup_mask) / float(e_img.shape[0] * e_img.shape[1]) >= 0.12:
                            return True, 0.76
                return False, 0.0

        except Exception as e:
            return False, 0.0

        return False, 0.0

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
            # First arg is the worker list if its items lack "category", or (when it is empty)
            # if the second arg holds raw detections. Otherwise an empty worker list would
            # make every PPE box be treated as a worker.
            first_is_workers = (
                (len(detections) > 0 and "category" not in detections[0])
                or (len(detections) == 0 and len(workers) > 0 and "category" in workers[0])
            )
            if first_is_workers:
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
                    if det.get("weak"):
                        colour_conf = weak_box_colour_score(frame, cat_name, bbox)
                        if colour_conf <= 0.0:
                            continue
                        conf = max(conf, colour_conf)
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
                "associated_items": {},
                # Confidence behind each status: the detection (or colour check) when present,
                # the model's explicit "no-<item>" score when absent, 0 when nothing was seen
                "ppe_confidence": {},
                # "present" | "absent" (model saw e.g. "No-Glove") | "none" (not visible) | "uncertain"
                "ppe_evidence": {}
            }
            worker_ppe_state[wid] = {
                cat: {"pos_conf": 0.0, "neg_conf": 0.0, "best_box": None}
                for cat in PPE_CATEGORIES
            }

        if not active_workers:
            if return_as_list:
                return []
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
            worker_bbox = results[wid]["bbox"]

            for cat in PPE_CATEGORIES:
                pos_c = state_dict[cat]["pos_conf"]
                neg_c = state_dict[cat]["neg_conf"]

                raw_present = (pos_c >= self.min_conf) and (pos_c >= neg_c)
                eff_conf = pos_c if raw_present else max(neg_c, 0.0)

                # Anatomical Computer Vision fallback if model did not detect it
                if not raw_present and frame is not None and neg_c < 0.40 and cat in self.cv_fallback_categories:
                    cv_pres, cv_c = self.verify_anatomical_ppe_presence(frame, worker_bbox, cat)
                    if cv_pres:
                        raw_present = True
                        eff_conf = max(eff_conf, cv_c)

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
                results[wid]["ppe_confidence"][cat] = round(float(eff_conf), 3)
                if status is True:
                    evidence = "present"
                elif status == "uncertain":
                    evidence = "uncertain"
                else:
                    evidence = "absent" if neg_c > 0 else "none"
                results[wid]["ppe_evidence"][cat] = evidence
                # Set aliases for template and UI compatibility
                if cat == "safety_pants":
                    results[wid]["pants"] = status
                elif cat == "helmet":
                    results[wid]["hardhat"] = status
                elif cat == "goggles":
                    results[wid]["glasses"] = status
                elif cat == "earmuff":
                    results[wid]["earmuffs"] = status
                if cat not in self.required_ppe:
                    continue
                if status is True:
                    compliant_count += 1
                elif status is False:
                    missing_list.append(cat)
                # Uncertain does not penalize score completely as confirmed violation
                elif status == "uncertain":
                    compliant_count += 0.5

            score = round(compliant_count / float(max(1, len(self.required_ppe))), 2)
            results[wid]["compliance_score"] = score
            results[wid]["missing_ppe"] = missing_list
            results[wid]["has_HEAD"] = bool(results[wid].get("helmet"))
            results[wid]["has_VISIBILITY"] = bool(results[wid].get("vest"))
            results[wid]["has_FALL_PROTECTION"] = bool(results[wid].get("harness"))
            results[wid]["has_FOOT"] = bool(results[wid].get("boots"))
            results[wid]["has_HAND"] = bool(results[wid].get("gloves"))
            results[wid]["has_EYE"] = bool(results[wid].get("goggles"))
            results[wid]["has_HEARING"] = bool(results[wid].get("earmuff"))

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
