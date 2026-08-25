import cv2
import numpy as np
import config

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

def validate_high_vis_vest(frame, v_box, conf):
    """Verifies that a detected vest contains true safety fluorescent hues (lime-yellow or safety orange)."""
    if conf >= 0.32:
        return True
    if frame is None:
        return conf >= 0.10

    H, W = frame.shape[:2]
    vx1, vy1, vx2, vy2 = v_box
    crop = frame[max(0, vy1):min(H, vy2), max(0, vx1):min(W, vx2)]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 1. Fluorescent High-Vis Yellow/Lime (Yellow-dominant, not background tree green):
    mask_lime_yellow = cv2.inRange(hsv, np.array([27, 75, 90]), np.array([48, 255, 255]))
    # 2. High-Vis Safety Orange / Fluorescent Red:
    mask_orange = cv2.inRange(hsv, np.array([4, 95, 95]), np.array([18, 255, 255]))
    mask_red = cv2.inRange(hsv, np.array([168, 95, 95]), np.array([180, 255, 255]))

    mask = mask_lime_yellow | mask_orange | mask_red
    ratio = np.count_nonzero(mask) / float(crop.shape[0] * crop.shape[1])
    return ratio >= 0.12

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

    def get_head_box(self, person_box):
        px1, py1, px2, py2 = person_box
        ph = py2 - py1
        pw = px2 - px1
        # Head is top portion of person bbox, allowing upwards extension for shoulders/torsos
        hy1 = max(0, py1 - int(ph * 0.45))
        hy2 = py1 + int(ph * self.head_span[1])
        # Give horizontal padding for helmet margins
        hx1 = max(0, px1 - int(pw * 0.15))
        hx2 = px2 + int(pw * 0.15)
        return [hx1, hy1, hx2, hy2]

    def get_torso_box(self, person_box):
        px1, py1, px2, py2 = person_box
        ph = py2 - py1
        ty1 = py1 + int(ph * self.torso_span[0])
        ty2 = py1 + int(ph * self.torso_span[1])
        return [px1, ty1, px2, ty2]

    def associate(self, workers, detections, frame=None):
        """
        Associates PPE detections (helmets, vests, no_helmets) with each worker.
        
        workers: list of dicts with 'worker_id', 'label', 'bbox', 'confidence'
        detections: list of dicts from ConstructionDetector
        
        Returns list of enriched worker dicts:
        [
            {
                "worker_id": int,
                "label": "Worker #X",
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "helmet": bool,
                "helmet_conf": float,
                "vest": bool,
                "vest_conf": float,
                "no_helmet_detected": bool,
                "associated_items": {
                    "helmet_box": [..] or None,
                    "vest_box": [..] or None
                }
            }, ...
        ]
        """
        helmets = [d for d in detections if d["category"] == "helmet"]
        vests = [d for d in detections if d["category"] == "vest"]
        no_helmets = [d for d in detections if d["category"] == "no_helmet"]

        enriched_workers = []

        for worker in workers:
            w_box = worker["bbox"]
            head_box = self.get_head_box(w_box)
            torso_box = self.get_torso_box(w_box)

            # 1. Helmet Association
            has_helmet = False
            best_helmet_conf = 0.0
            best_helmet_box = None

            for h in helmets:
                h_box = h["bbox"]
                h_center = ((h_box[0] + h_box[2]) / 2, (h_box[1] + h_box[3]) / 2)
                overlap = compute_box_overlap(h_box, head_box)

                if is_point_inside(h_center, head_box) or overlap >= self.min_overlap:
                    if h["confidence"] > best_helmet_conf:
                        has_helmet = True
                        best_helmet_conf = h["confidence"]
                        best_helmet_box = h_box

            # Check for explicit no_helmet signal
            has_no_helmet_signal = False
            for nh in no_helmets:
                nh_box = nh["bbox"]
                nh_center = ((nh_box[0] + nh_box[2]) / 2, (nh_box[1] + nh_box[3]) / 2)
                if is_point_inside(nh_center, head_box) or compute_box_overlap(nh_box, head_box) >= self.min_overlap:
                    has_no_helmet_signal = True
                    break

            if has_no_helmet_signal and not has_helmet:
                has_helmet = False

            # 2. Vest Association
            has_vest = False
            best_vest_conf = 0.0
            best_vest_box = None

            for v in vests:
                v_box = v["bbox"]
                v_center = ((v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2)
                overlap = compute_box_overlap(v_box, torso_box)

                if is_point_inside(v_center, torso_box) or overlap >= self.min_overlap:
                    if validate_high_vis_vest(frame, v_box, v["confidence"]):
                        if v["confidence"] > best_vest_conf:
                            has_vest = True
                            best_vest_conf = v["confidence"]
                            best_vest_box = v_box

            enriched_workers.append({
                "worker_id": worker["worker_id"],
                "label": worker["label"],
                "bbox": w_box,
                "confidence": worker["confidence"],
                "helmet": has_helmet,
                "helmet_conf": round(best_helmet_conf, 3),
                "vest": has_vest,
                "vest_conf": round(best_vest_conf, 3),
                "no_helmet_detected": has_no_helmet_signal,
                "head_box": head_box,
                "torso_box": torso_box,
                "associated_items": {
                    "helmet_box": best_helmet_box,
                    "vest_box": best_vest_box
                }
            })

        return enriched_workers
