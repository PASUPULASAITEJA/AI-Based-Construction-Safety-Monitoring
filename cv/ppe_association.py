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
        
        # All 8 supported PPE classes in the SOTA system
        self.supported_classes = [
            "helmet", "vest", "boots", "gloves", 
            "goggles", "ear_muffs", "harness", "chaps"
        ]

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
        Associates all 8 PPE detections with each worker using 1-to-1 spatial matching.
        """
        # Dictionary to store filtered detections by category
        category_detections = {cat: [] for cat in self.supported_classes}
        
        for d in detections:
            cat = d["category"]
            if cat in self.supported_classes:
                category_detections[cat].append(d)

        num_workers = len(workers)
        
        # Store associated item for each worker (worker_idx -> category -> detection dict)
        worker_ppe = {i: {cat: None for cat in self.supported_classes} for i in range(num_workers)}

        # Perform assignment for all classes (simplified heuristic: closest center within bounding box)
        for cat in self.supported_classes:
            items = category_detections[cat]
            for item in items:
                item_box = item["bbox"]
                item_center = ((item_box[0] + item_box[2]) / 2, (item_box[1] + item_box[3]) / 2)
                best_i = None
                best_dist = float("inf")

                for i, worker in enumerate(workers):
                    w_box = worker["bbox"]
                    # Depending on the item, we check overlap with specific body regions (head vs torso vs full body)
                    if cat in ["helmet", "goggles", "ear_muffs"]:
                        region_box = self.get_head_box(w_box)
                    elif cat in ["vest", "harness"]:
                        region_box = self.get_torso_box(w_box)
                    else: # gloves, boots, chaps
                        region_box = w_box # Full body for now

                    overlap = compute_box_overlap(item_box, region_box)

                    if (is_point_inside(item_center, region_box) or overlap >= 0.15) and (w_box[0] - 20 <= item_center[0] <= w_box[2] + 20):
                        dist = (item_center[0] - (w_box[0] + w_box[2]) / 2) ** 2 + (item_center[1] - (w_box[1] + w_box[3]) / 2) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_i = i

                if best_i is not None:
                    existing = worker_ppe[best_i][cat]
                    if existing is None or item["confidence"] > existing["confidence"]:
                        worker_ppe[best_i][cat] = item

        # Build enriched worker dictionary
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
            
            # Map the 8 PPE classes into the worker_data dict
            for cat in self.supported_classes:
                matched_item = worker_ppe[i][cat]
                has_item = matched_item is not None
                
                # Flag expected by ViolationManager EWMA
                worker_data[f"has_{cat}"] = has_item
                worker_data[f"{cat}_conf"] = round(matched_item["confidence"], 3) if has_item else 0.0
                worker_data["associated_items"][f"{cat}_box"] = matched_item["bbox"] if has_item else None
                
                # For backwards compatibility with older templates (helmet/vest)
                if cat in ["helmet", "vest"]:
                    worker_data[cat] = has_item

            enriched_workers.append(worker_data)

        return enriched_workers
