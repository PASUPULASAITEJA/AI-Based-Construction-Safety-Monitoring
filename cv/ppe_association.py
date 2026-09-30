import cv2
import numpy as np
import config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.ppe_config import PPE_CATEGORIES, get_supported_model_classes

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
        self.all_ppe_classes = []
        for cat in PPE_CATEGORIES:
            self.all_ppe_classes.extend(cat["model_classes"])
            
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
                item_center = ((item_box[0] + item_box[2]) / 2, (item_box[1] + item_box[3]) / 2)
                best_i = None
                best_dist = float("inf")

                for i, worker in enumerate(workers):
                    w_box = worker["bbox"]
                    
                    # Heuristics for body regions
                    if cls in ["helmet", "goggles", "ear_muffs", "ear_protection", "hardhat"]:
                        region_box = self.get_head_box(w_box)
                    elif cls in ["vest", "harness", "safety_vest"]:
                        region_box = self.get_torso_box(w_box)
                    else: # gloves, boots, pants
                        region_box = w_box

                    overlap = compute_box_overlap(item_box, region_box)

                    if (is_point_inside(item_center, region_box) or overlap >= 0.15) and (w_box[0] - 20 <= item_center[0] <= w_box[2] + 20):
                        dist = (item_center[0] - (w_box[0] + w_box[2]) / 2) ** 2 + (item_center[1] - (w_box[1] + w_box[3]) / 2) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_i = i

                if best_i is not None:
                    existing = worker_ppe[best_i][cls]
                    if existing is None or item["confidence"] > existing["confidence"]:
                        worker_ppe[best_i][cls] = item

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
                            
                has_item = best_item is not None
                
                worker_data[f"has_{cat_id}"] = has_item
                worker_data[f"{cat_id}_conf"] = round(best_item["confidence"], 3) if has_item else 0.0
                worker_data["associated_items"][f"{cat_id}_box"] = best_item["bbox"] if has_item else None
                
            # For backwards compatibility with older templates (helmet/vest)
            worker_data["helmet"] = worker_data.get("has_HEAD", False)
            worker_data["vest"] = worker_data.get("has_VISIBILITY", False)

            enriched_workers.append(worker_data)

        return enriched_workers
