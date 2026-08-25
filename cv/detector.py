"""
CV Module: YOLO Object Detector
Handles YOLOv8 model loading, dynamic class resolution from data.yaml, and inference.
"""
import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import config

class ConstructionDetector:
    def __init__(self, model_path=None, data_yaml=config.DATA_YAML_PATH, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD):
        self.conf = conf
        self.iou = iou
        self.device = config.DEVICE
        self.data_yaml = data_yaml

        # Load class mappings dynamically from data.yaml
        # Determine model path
        if model_path is None or not os.path.exists(model_path):
            if os.path.exists(config.MODEL_PATH):
                self.model_path = config.MODEL_PATH
            else:
                self.model_path = config.FALLBACK_MODEL
        else:
            self.model_path = model_path

        print(f"[Detector] Loading model from: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)

        self.classes = {}
        self.canonical_map = {}
        self._load_classes()

    def _load_classes(self):
        # 1. Load Ground Truth class schema directly from Construction-PPE data.yaml
        if os.path.exists(self.data_yaml):
            try:
                with open(self.data_yaml, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    raw_names = cfg.get("names", {})
                    if isinstance(raw_names, list):
                        self.classes = {i: name for i, name in enumerate(raw_names)}
                    elif isinstance(raw_names, dict):
                        self.classes = {int(k): v for k, v in raw_names.items()}
            except Exception as e:
                print(f"[-] Warning: Failed to parse data.yaml: {e}")

        # Fallback if data.yaml missing
        if not self.classes:
            if hasattr(self.model, "names") and self.model.names:
                self.classes = {int(k): v for k, v in self.model.names.items()}
            else:
                self.classes = {
                    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
                    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
                    9: "no_gloves", 10: "no_boots"
                }

        # Build canonical map (normalized lowercase name -> list of class ids)
        self.canonical_map = {}
        for cid, cname in self.classes.items():
            norm = str(cname).strip().lower()
            if norm not in self.canonical_map:
                self.canonical_map[norm] = []
            self.canonical_map[norm].append(cid)

        # In case base COCO weights are running before fine-tuning, also map COCO 0 (person)
        if 0 not in self.canonical_map.get("person", []) and len(getattr(self.model, "names", {})) == 80:
            if "person" not in self.canonical_map:
                self.canonical_map["person"] = []
            self.canonical_map["person"].append(0)

        # Identify key class IDs
        self.person_cids = self.canonical_map.get("person", [6])
        self.helmet_cids = self.canonical_map.get("helmet", [0])
        self.vest_cids = self.canonical_map.get("vest", [2])
        self.no_helmet_cids = self.canonical_map.get("no_helmet", [7])

    def detect(self, image_or_frame, conf=None, iou=None):
        """
        Run inference on a single image (numpy array or filepath).
        Returns a list of structured detection dictionaries with class-specific filtering.
        """
        c_thresh = conf if conf is not None else self.conf
        i_thresh = iou if iou is not None else self.iou

        results = self.model.predict(
            source=image_or_frame,
            conf=c_thresh,
            iou=i_thresh,
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False
        )

        if not results or len(results) == 0:
            return []

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return []

        orig_h, orig_w = r.orig_shape
        raw_candidates = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.classes.get(class_id, self.model.names.get(class_id, f"class_{class_id}"))

            norm_name = str(class_name).strip().lower()
            category = "other"
            if class_id in self.person_cids or norm_name == "person":
                category = "person"
            elif class_id in self.helmet_cids or norm_name == "helmet":
                category = "helmet"
            elif class_id in self.vest_cids or norm_name == "vest":
                category = "vest"
            elif class_id in self.no_helmet_cids or norm_name == "no_helmet":
                category = "no_helmet"

            raw_candidates.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name,
                "category": category
            })

        # Class-specific confidence filtering
        min_thresholds = {
            "person": 0.08,      # Sensitive for crouching, seated, and overhead workers
            "helmet": 0.025,     # High recall for aerial/overhead hardhats (chromatic validator confirms)
            "vest": 0.04,        # Sensitive for safety vests (chromatic validator confirms high-vis)
            "no_helmet": 0.25,
            "other": 0.15
        }

        # 1. Filter PPE detections
        filtered_ppe = [
            d for d in raw_candidates 
            if d["category"] != "person" and d["confidence"] >= min_thresholds.get(d["category"], 0.10)
        ]

        # 2. Filter & Deduplicate Person boxes (NMS + Containment suppression)
        person_candidates = [
            d for d in raw_candidates 
            if d["category"] == "person" and d["confidence"] >= min_thresholds["person"]
        ]
        
        vests_list = [d for d in filtered_ppe if d["category"] == "vest"]

        clean_persons = []
        for cand in sorted(person_candidates, key=lambda x: x["confidence"], reverse=True):
            b1 = cand["bbox"]
            w1 = b1[2] - b1[0]
            h1 = b1[3] - b1[1]
            cx1 = (b1[0] + b1[2]) / 2
            a1 = max(1, w1 * h1)

            # Skip merged group box if it contains multiple horizontally separated vests
            contained_vests = [
                v for v in vests_list 
                if b1[0] <= (v["bbox"][0] + v["bbox"][2]) / 2 <= b1[2] and b1[1] <= (v["bbox"][1] + v["bbox"][3]) / 2 <= b1[3]
            ]
            if len(contained_vests) >= 2:
                v_cxs = [(v["bbox"][0] + v["bbox"][2]) / 2 for v in contained_vests]
                if max(v_cxs) - min(v_cxs) > 0.25 * w1:
                    continue  # Skip multi-person group merge box!

            keep = True
            for kept in clean_persons:
                b2 = kept["bbox"]
                w2 = b2[2] - b2[0]
                h2 = b2[3] - b2[1]
                cx2 = (b2[0] + b2[2]) / 2
                a2 = max(1, w2 * h2)

                dx = abs(cx1 - cx2)
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih

                union = a1 + a2 - inter
                iou_val = inter / union if union > 0 else 0
                containment = inter / min(a1, a2) if min(a1, a2) > 0 else 0

                # Suppress if heavily nested (>65%) or high IoU (>0.40)
                if containment > 0.65 or iou_val > 0.40:
                    keep = False
                    break

                # Suppress if moderate overlap in the same vertical column (dx < 25% width)
                if (containment > 0.45 or iou_val > 0.25) and dx < 0.25 * min(w1, w2):
                    keep = False
                    break

            if keep:
                clean_persons.append(cand)

        # 3. Recover any partially occluded / background / lying worker wearing a confirmed vest
        for v in filtered_ppe:
            if v["category"] == "vest" and v["confidence"] >= 0.04:
                from cv.ppe_association import validate_high_vis_vest
                if not validate_high_vis_vest(image_or_frame, v["bbox"], v["confidence"]):
                    continue

                vx1, vy1, vx2, vy2 = v["bbox"]
                vw, vh = vx2 - vx1, vy2 - vy1
                # Skip small corner debris artifacts
                if vh < 90 or vw < 70 or vw > 1.6 * vh:
                    continue

                vc_x = (vx1 + vx2) / 2
                vc_y = (vy1 + vy2) / 2
                is_covered = any(p["bbox"][0] <= vc_x <= p["bbox"][2] and p["bbox"][1] <= vc_y <= p["bbox"][3] for p in clean_persons)
                if not is_covered:
                    px1 = max(0, vx1 - int(vw * 0.15))
                    py1 = max(0, vy1 - int(vh * 0.30))
                    px2 = min(orig_w, vx2 + int(vw * 0.80))
                    py2 = min(orig_h, vy2 + int(vh * 0.90))
                    clean_persons.append({
                        "bbox": [px1, py1, px2, py2],
                        "confidence": v["confidence"],
                        "class_id": 6,
                        "class_name": "Person",
                        "category": "person"
                    })

        return clean_persons + filtered_ppe
