import cv2
import numpy as np
import config
from safety.ppe_config import PPE_CATEGORIES, NEGATIVE_CLASSES

# Which part of the worker box each PPE category is searched in
HEAD_CATEGORIES = {"HEAD", "EYE", "HEARING"}
TORSO_CATEGORIES = {"VISIBILITY", "FALL_PROTECTION"}

# Canonical model class -> PPE category
CLASS_TO_CATEGORY = {cls: cat["category"] for cat in PPE_CATEGORIES for cls in cat["model_classes"]}

# Vest detections at or above this confidence skip the colour check
VEST_TRUSTED_CONFIDENCE = 0.5
# Minimum fraction of fluorescent (hi-vis) pixels for a low-confidence vest to be accepted
MIN_HIGH_VIS_PIXEL_RATIO = 0.12

def validate_high_vis_vest(image_or_frame, bbox, confidence):
    """
    Confirms a low-confidence vest detection by checking the crop for fluorescent
    yellow / orange / lime pixels (typical high-visibility fabric).
    """
    if confidence >= VEST_TRUSTED_CONFIDENCE:
        return True
    if not isinstance(image_or_frame, np.ndarray):
        return False

    h, w = image_or_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    hsv = cv2.cvtColor(image_or_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    # OpenCV hue range is 0-179: orange ~5-22, yellow/lime ~22-50; hi-vis is highly saturated and bright
    mask = cv2.inRange(hsv, (5, 110, 130), (50, 255, 255))
    ratio = float(np.count_nonzero(mask)) / mask.size
    return ratio >= MIN_HIGH_VIS_PIXEL_RATIO

def compute_box_overlap(inner_box, outer_box):
    """
    Computes fraction of inner_box that is inside outer_box.
    Returns value between 0.0 and 1.0.
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area

def is_point_inside(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

class PPEAssociator:
    def __init__(
        self,
        head_span=config.HEAD_REGION_SPAN,
        torso_span=config.TORSO_REGION_SPAN,
        min_overlap=config.MIN_OVERLAP_RATIO
    ):
        self.head_span = head_span
        self.torso_span = torso_span
        self.min_overlap = min_overlap
        
        # Collect all possible model classes across all categories
        self.all_ppe_classes = list(CLASS_TO_CATEGORY)

    def get_head_box(self, person_box):
        px1, py1, px2, py2 = person_box
        ph = py2 - py1
        pw = px2 - px1
        # Head is upper portion of person bbox
        hy1 = max(0, py1 - int(ph * 0.20))
        hy2 = py1 + int(ph * self.head_span[1])
        hx1 = max(0, px1 - int(pw * 0.15))
        hx2 = px2 + int(pw * 0.15)
        return [hx1, hy1, hx2, hy2]

    def get_torso_box(self, person_box):
        px1, py1, px2, py2 = person_box
        ph = py2 - py1
        ty1 = py1 + int(ph * self.torso_span[0])
        ty2 = py1 + int(ph * self.torso_span[1])
        return [px1, ty1, px2, ty2]

    def _region_box(self, category_id, w_box):
        if category_id in HEAD_CATEGORIES:
            return self.get_head_box(w_box)
        if category_id in TORSO_CATEGORIES:
            return self.get_torso_box(w_box)
        return w_box  # gloves, boots, pants: anywhere on the body

    def _match_worker(self, item_box, category_id, workers):
        """Returns the index of the worker whose matching body region holds item_box (closest wins), or None."""
        item_center = ((item_box[0] + item_box[2]) / 2, (item_box[1] + item_box[3]) / 2)
        best_i = None
        best_dist = float("inf")
        for i, worker in enumerate(workers):
            w_box = worker["bbox"]
            region_box = self._region_box(category_id, w_box)
            overlap = compute_box_overlap(item_box, region_box)
            inside = is_point_inside(item_center, region_box) or overlap >= self.min_overlap
            if inside and (w_box[0] - 20 <= item_center[0] <= w_box[2] + 20):
                dist = (item_center[0] - (w_box[0] + w_box[2]) / 2) ** 2 + (item_center[1] - (w_box[1] + w_box[3]) / 2) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
        return best_i

    def associate(self, workers, detections, frame=None):
        """
        Associates PPE detections with each worker using 1-to-1 spatial matching.
        """
        # Dictionary to store filtered detections by class name
        class_detections = {cls: [] for cls in self.all_ppe_classes}
        
        for d in detections:
            cat = d["category"]
            if cat in self.all_ppe_classes:
                class_detections[cat].append(d)

        num_workers = len(workers)
        
        # Store associated item for each worker (worker_idx -> model_class -> detection dict)
        worker_ppe = {i: {cls: None for cls in self.all_ppe_classes} for i in range(num_workers)}

        for cls in self.all_ppe_classes:
            items = class_detections[cls]
            for item in items:
                item_box = item["bbox"]
                best_i = self._match_worker(item_box, CLASS_TO_CATEGORY[cls], workers)
                if best_i is not None:
                    existing = worker_ppe[best_i][cls]
                    if existing is None or item["confidence"] > existing["confidence"]:
                        worker_ppe[best_i][cls] = item

        # Negative detections (no_helmet, no_gloves, ...) per worker: category -> best confidence
        worker_negatives = {i: {} for i in range(num_workers)}
        for d in detections:
            neg_cat = NEGATIVE_CLASSES.get(d["category"])
            if neg_cat is None:
                continue
            best_i = self._match_worker(d["bbox"], neg_cat, workers)
            if best_i is not None:
                prev = worker_negatives[best_i].get(neg_cat, 0.0)
                worker_negatives[best_i][neg_cat] = max(prev, d["confidence"])

        # Build enriched worker dictionary based on Categories
        enriched_workers = []
        for i, worker in enumerate(workers):
            w_box = worker["bbox"]
            
            worker_data = {
                "worker_id": worker["worker_id"],
                "label": worker["label"],
                "bbox": w_box,
                "confidence": worker["confidence"],
                "head_box": self.get_head_box(w_box),
                "torso_box": self.get_torso_box(w_box),
                "associated_items": {}
            }
            
            # Map into PPE categories (HEAD, VISIBILITY, etc)
            for ppe_cat in PPE_CATEGORIES:
                cat_id = ppe_cat["category"]
                
                # Check if ANY of the model classes for this category was found
                best_item = None
                for m_cls in ppe_cat["model_classes"]:
                    if worker_ppe[i].get(m_cls):
                        if best_item is None or worker_ppe[i][m_cls]["confidence"] > best_item["confidence"]:
                            best_item = worker_ppe[i][m_cls]
                            
                # A more confident "no_<item>" detection on the same worker overrides the positive one
                neg_conf = worker_negatives[i].get(cat_id, 0.0)
                if best_item is not None and neg_conf > best_item["confidence"]:
                    best_item = None

                has_item = best_item is not None

                worker_data[f"has_{cat_id}"] = has_item
                worker_data[f"{cat_id}_conf"] = round(best_item["confidence"], 3) if has_item else 0.0
                worker_data["associated_items"][f"{cat_id}_box"] = best_item["bbox"] if has_item else None
                
            # For backwards compatibility with older templates (helmet/vest)
            worker_data["helmet"] = worker_data.get("has_HEAD", False)
            worker_data["vest"] = worker_data.get("has_VISIBILITY", False)

            enriched_workers.append(worker_data)

        return enriched_workers
