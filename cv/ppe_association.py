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
    if frame is None:
        return conf >= 0.04

    H, W = frame.shape[:2]
    vx1, vy1, vx2, vy2 = v_box
    vw, vh = vx2 - vx1, vy2 - vy1
    if vw > 1.6 * vh:  # Excessively wide landscape pile/tarp
        return False

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

def validate_hardhat_color(frame, h_box, conf):
    """Verifies that a hardhat detection contains valid hardhat colors (yellow, orange, white)."""
    if conf >= 0.10:
        return True
    if frame is None:
        return conf >= 0.025

    H, W = frame.shape[:2]
    hx1, hy1, hx2, hy2 = h_box
    hw, hh = hx2 - hx1, hy2 - hy1
    if hw > 1.6 * hh or hh > 1.6 * hw or hw > 0.40 * W:
        return False

    crop = frame[max(0, hy1):min(H, hy2), max(0, hx1):min(W, hx2)]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
    mask_orange = cv2.inRange(hsv, np.array([4, 100, 100]), np.array([15, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 45, 255]))

    mask = mask_yellow | mask_orange | mask_white
    ratio = np.count_nonzero(mask) / float(crop.shape[0] * crop.shape[1])
    return ratio >= 0.18

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
        # Head is upper portion of person bbox
        hy1 = max(0, py1 - int(ph * 0.20))
        hy2 = py1 + int(ph * 0.45)
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
        Associates PPE detections (helmets, vests, no_helmets) with each worker using 1-to-1 spatial matching.
        """
        helmets = [d for d in detections if d["category"] == "helmet" and validate_hardhat_color(frame, d["bbox"], d["confidence"])]
        vests = [d for d in detections if d["category"] == "vest" and validate_high_vis_vest(frame, d["bbox"], d["confidence"])]
        no_helmets = [d for d in detections if d["category"] == "no_helmet"]

        num_workers = len(workers)
        worker_helmets = {i: None for i in range(num_workers)}
        worker_vests = {i: None for i in range(num_workers)}
        worker_no_helmets = {i: False for i in range(num_workers)}

        # 1. Helmet 1-to-1 assignment
        for h in helmets:
            h_box = h["bbox"]
            h_center = ((h_box[0] + h_box[2]) / 2, (h_box[1] + h_box[3]) / 2)
            best_i = None
            best_dist = float("inf")

            for i, worker in enumerate(workers):
                w_box = worker["bbox"]
                head_box = self.get_head_box(w_box)
                overlap = compute_box_overlap(h_box, head_box)

                if (is_point_inside(h_center, head_box) or overlap >= 0.20) and (w_box[0] - 20 <= h_center[0] <= w_box[2] + 20):
                    dist = (h_center[0] - (w_box[0] + w_box[2]) / 2) ** 2 + (h_center[1] - w_box[1]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i

            if best_i is not None:
                if worker_helmets[best_i] is None or h["confidence"] > worker_helmets[best_i]["confidence"]:
                    worker_helmets[best_i] = h

        # 2. Vest 1-to-1 assignment
        for v in vests:
            v_box = v["bbox"]
            v_center = ((v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2)
            best_i = None
            best_dist = float("inf")

            for i, worker in enumerate(workers):
                w_box = worker["bbox"]
                torso_box = self.get_torso_box(w_box)
                overlap = compute_box_overlap(v_box, torso_box)

                if (is_point_inside(v_center, torso_box) or overlap >= 0.20) and (w_box[0] - 20 <= v_center[0] <= w_box[2] + 20):
                    dist = (v_center[0] - (w_box[0] + w_box[2]) / 2) ** 2 + (v_center[1] - (w_box[1] + w_box[3]) / 2) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i

            if best_i is not None:
                if worker_vests[best_i] is None or v["confidence"] > worker_vests[best_i]["confidence"]:
                    worker_vests[best_i] = v

        # 3. No-helmet signal assignment
        for nh in no_helmets:
            nh_box = nh["bbox"]
            nh_center = ((nh_box[0] + nh_box[2]) / 2, (nh_box[1] + nh_box[3]) / 2)
            for i, worker in enumerate(workers):
                w_box = worker["bbox"]
                head_box = self.get_head_box(w_box)
                if (is_point_inside(nh_center, head_box) or compute_box_overlap(nh_box, head_box) >= 0.20) and (w_box[0] - 20 <= nh_center[0] <= w_box[2] + 20):
                    worker_no_helmets[i] = True

        enriched_workers = []
        for i, worker in enumerate(workers):
            w_box = worker["bbox"]
            head_box = self.get_head_box(w_box)
            torso_box = self.get_torso_box(w_box)

            matched_h = worker_helmets[i]
            matched_v = worker_vests[i]

            has_helmet = matched_h is not None
            best_helmet_conf = matched_h["confidence"] if matched_h else 0.0
            best_helmet_box = matched_h["bbox"] if matched_h else None

            if worker_no_helmets[i] and not has_helmet:
                has_helmet = False

            has_vest = matched_v is not None
            best_vest_conf = matched_v["confidence"] if matched_v else 0.0
            best_vest_box = matched_v["bbox"] if matched_v else None

            enriched_workers.append({
                "worker_id": worker["worker_id"],
                "label": worker["label"],
                "bbox": w_box,
                "confidence": worker["confidence"],
                "helmet": has_helmet,
                "helmet_conf": round(best_helmet_conf, 3),
                "vest": has_vest,
                "vest_conf": round(best_vest_conf, 3),
                "no_helmet_detected": worker_no_helmets[i],
                "head_box": head_box,
                "torso_box": torso_box,
                "associated_items": {
                    "helmet_box": best_helmet_box,
                    "vest_box": best_vest_box
                }
            })

        return enriched_workers
