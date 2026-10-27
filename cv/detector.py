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

# Helmet/vest boxes scoring between WEAK_PPE_MIN_CONF and STRONG_PPE_CONF are flagged "weak"
# and only counted if their pixels pass a colour check (PPEAssociator). This recovers real
# hi-vis vests the model scores at ~0.1 and rejects 0.2-0.3 "vests" on yellow T-shirts and
# "helmets" on bare heads.
WEAK_PPE_CATEGORIES = {"helmet", "vest"}
WEAK_PPE_MIN_CONF = 0.05
STRONG_PPE_CONF = 0.35

# Auxiliary model class (lower-case) -> pipeline category. Helmets stay with the main model +
# colour verification. Vests from the aux model are added because the main model + hi-vis
# colour check miss non-fluorescent safety/utility vests (e.g. dark work vests).
AUX_CLASS_MAP = {
    "ear-protection": "earmuff", "no-ear-protection": "no-earmuff",
    "glass": "goggles", "no-glass": "no-goggles",
    "glove": "gloves", "no-glove": "no-gloves",
    "boots": "boots", "no-boots": "no-boots",
    "vest": "vest", "no-vest": "no-vest",
}
# Stricter than AUX_PPE_POS_CONF where a false positive would hide a violation
AUX_POS_CONF_OVERRIDES = {"vest": 0.45}

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

        # General-purpose COCO person detector. The PPE model only learned "Person" from
        # construction-site photos and misses people in other settings (webcam close-ups,
        # suits, crowds); its boxes are merged with the PPE model's before de-duplication.
        self.person_model = None
        if getattr(config, "USE_PERSON_MODEL", False) and os.path.exists(config.PERSON_MODEL_PATH) \
                and os.path.abspath(self.model_path) != os.path.abspath(config.PERSON_MODEL_PATH):
            try:
                self.person_model = YOLO(config.PERSON_MODEL_PATH)
                print(f"[Detector] Person model loaded from: {config.PERSON_MODEL_PATH}")
            except Exception as e:
                print(f"[-] Warning: person model unavailable ({e}); using PPE model persons only")

        self.aux_model = None
        self._aux_cache = []
        self._aux_calls = 0
        if getattr(config, "USE_AUX_PPE_MODEL", False) and os.path.exists(config.AUX_PPE_MODEL_PATH):
            try:
                self.aux_model = YOLO(config.AUX_PPE_MODEL_PATH)
                print(f"[Detector] Aux PPE model loaded from: {config.AUX_PPE_MODEL_PATH}")
            except Exception as e:
                print(f"[-] Warning: aux PPE model unavailable ({e}); ear/eye/hand/foot PPE limited")

    @property
    def supported_ppe(self):
        """PPE categories at least one loaded model can detect."""
        cats = {c for c in self.canonical_map if c in
                ("helmet", "vest", "goggles", "gloves", "boots", "earmuff", "harness", "safety-pants")}
        if self.aux_model is not None:
            cats |= {"goggles", "gloves", "boots", "earmuff"}
        return sorted(c.replace("safety-pants", "safety_pants") for c in cats)

    def _detect_aux_ppe(self, image_or_frame, every_n=1):
        """Ear/eye/hand/foot PPE from the auxiliary model. With every_n > 1 (live video) the
        model runs on every Nth call and the last result is reused in between."""
        if self.aux_model is None:
            return []
        self._aux_calls += 1
        if every_n > 1 and (self._aux_calls - 1) % every_n != 0:
            return self._aux_cache
        results = self.aux_model.predict(
            source=image_or_frame,
            conf=min(config.AUX_PPE_POS_CONF, config.AUX_PPE_NEG_CONF),
            iou=self.iou,
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False
        )
        found = []
        r = results[0] if results else None
        for box in (r.boxes if r is not None and r.boxes is not None else []):
            category = AUX_CLASS_MAP.get(str(r.names[int(box.cls[0])]).strip().lower())
            if category is None:
                continue
            confidence = float(box.conf[0].cpu().numpy())
            min_conf = config.AUX_PPE_NEG_CONF if category.startswith("no-") else \
                AUX_POS_CONF_OVERRIDES.get(category, config.AUX_PPE_POS_CONF)
            if confidence < min_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            found.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 4),
                "class_id": int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
                "category": category,
                "source": "aux"
            })

        # When the model calls the same region both "Vest" and "No-Vest" (e.g. 0.57 vs 0.45 on
        # an open winter coat) it is unsure; drop the positive so the item is not marked present.
        negatives = [d for d in found if d["category"].startswith("no-")]
        found = [d for d in found if d["category"].startswith("no-") or not any(
            n["category"] == "no-" + d["category"]
            and self._overlap(d["bbox"], n["bbox"])[0] > 0.5
            and d["confidence"] - n["confidence"] < 0.20
            for n in negatives)]
        self._aux_cache = found
        return found

    def _detect_coco_persons(self, image_or_frame):
        """Person boxes from the COCO model (class 0), in the same dict format as detect()."""
        if self.person_model is None:
            return []
        results = self.person_model.predict(
            source=image_or_frame,
            conf=config.PERSON_MODEL_CONF,
            iou=self.iou,
            classes=[0],
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False
        )
        persons = []
        for box in results[0].boxes if results and results[0].boxes is not None else []:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(box.conf[0].cpu().numpy()), 4),
                "class_id": 0,
                "class_name": "person",
                "category": "person",
                "source": "coco"
            })
        return persons

    @staticmethod
    def _overlap(b1, b2):
        """Returns (iou, fraction of b1 inside b2, fraction of b2 inside b1)."""
        a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        iw = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        ih = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter = iw * ih
        return inter / float(a1 + a2 - inter), inter / float(a1), inter / float(a2)

    def _same_or_group(self, b1, b2):
        """True if b1 is the same person as b2, or a box spanning b2 and its neighbours."""
        iou, in_b2, b2_in = self._overlap(b1, b2)
        return iou > 0.30 or in_b2 > 0.60 or b2_in > 0.60

    def _drop_group_boxes(self, persons):
        """Removes boxes that enclose 2+ other people, and near-duplicates (IoU > 0.7)."""
        kept = []
        for p in sorted(persons, key=lambda x: x["confidence"], reverse=True):
            enclosed = sum(1 for o in persons if o is not p and self._overlap(o["bbox"], p["bbox"])[1] >= 0.80
                           and (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])
                           < (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))
            if enclosed >= 2:
                continue
            if any(self._overlap(p["bbox"], k["bbox"])[0] > 0.70 for k in kept):
                continue
            kept.append(p)
        return kept

    def _load_classes(self):
        # 1. Always prioritize the model's own internal names dictionary
        # because the network's output logits directly correspond to model.names!
        if hasattr(self.model, "names") and self.model.names:
            self.classes = {int(k): v for k, v in self.model.names.items()}
        elif os.path.exists(self.data_yaml):
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

        if not self.classes:
            self.classes = {i: name for i, name in enumerate(config.CLASSES_17)}

        # Build canonical map (normalized lowercase name -> list of class ids) and category_map
        self.canonical_map = {}
        self.category_map = {}
        for cid, cname in self.classes.items():
            norm = self._normalize_category(cname)
            self.category_map[cid] = norm
            if norm not in self.canonical_map:
                self.canonical_map[norm] = []
            self.canonical_map[norm].append(cid)

    @property
    def person_cids(self):
        return self.canonical_map.get("person", [0])

    @staticmethod
    def _normalize_category(raw_name: str) -> str:
        s = str(raw_name).strip().lower().replace("_", "-")
        if s in ["person", "worker"]:
            return "person"
        elif s in ["helmet"]:
            return "helmet"
        elif s in ["no-helmet", "no-head"]:
            return "no-helmet"
        elif s in ["vest", "safety-vest"]:
            return "vest"
        elif s in ["no-vest", "no-safety-vest"]:
            return "no-vest"
        elif s in ["goggles", "glasses"]:
            return "goggles"
        elif s in ["no-goggle", "no-goggles", "no-glasses"]:
            return "no-goggles"
        elif s in ["gloves", "glove"]:
            return "gloves"
        elif s in ["no-gloves", "no-glove"]:
            return "no-gloves"
        elif s in ["boots", "safety-boots", "shoes"]:
            return "boots"
        elif s in ["no-boots", "no-shoes"]:
            return "no-boots"
        elif s in ["safety-pants", "safety_pants", "safety-suit"]:
            return "safety-pants"
        elif s in ["no-safety-pants", "no_safety_pants"]:
            return "no-safety-pants"
        elif s in ["earmuff", "earmuffs", "ear-mufs"]:
            return "earmuff"
        elif s in ["no-earmuff", "no_earmuff"]:
            return "no-earmuff"
        elif s in ["harness", "safety-harness"]:
            return "harness"
        elif s in ["no-harness", "no_harness"]:
            return "no-harness"
        return s

    def detect(self, image_or_frame, conf=None, iou=None, aux_every_n=1):
        """
        Run inference on a single image (numpy array or filepath).
        Returns a list of structured detection dictionaries across all 17 classes.
        """
        c_thresh = conf if conf is not None else self.conf
        i_thresh = iou if iou is not None else self.iou

        # Predict below the working threshold so weak helmet/vest boxes can be colour-verified
        # downstream (the model often scores real hi-vis vests at 0.10-0.15).
        results = self.model.predict(
            source=image_or_frame,
            conf=min(c_thresh, WEAK_PPE_MIN_CONF),
            iou=i_thresh,
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False
        )

        boxes = results[0].boxes if results and results[0].boxes is not None else []

        # Class-specific confidence filtering thresholds
        min_thresholds = {
            "person": 0.20,
            "helmet": 0.15,
            "no-helmet": 0.25,
            "goggles": 0.15,
            "no-goggles": 0.30,
            "earmuff": 0.15,
            "no-earmuff": 0.30,
            "vest": 0.15,
            "no-vest": 0.25,
            "safety-pants": 0.15,
            "no-safety-pants": 0.30,
            "gloves": 0.15,
            "no-gloves": 0.25,
            "boots": 0.15,
            "no-boots": 0.30,
            "harness": 0.15,
            "no-harness": 0.30,
            "other": 0.20
        }

        raw_candidates = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.classes.get(class_id, f"class_{class_id}")
            category = self._normalize_category(class_name)

            # Check threshold
            req_conf = max(c_thresh, min_thresholds.get(category, min_thresholds["other"]))
            if category in WEAK_PPE_CATEGORIES and WEAK_PPE_MIN_CONF <= confidence < max(req_conf, STRONG_PPE_CONF):
                # Only counted by PPEAssociator if the box passes a colour check
                raw_candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                    "category": category,
                    "weak": True
                })
            elif confidence >= req_conf:
                raw_candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                    "category": category
                })

        # 1. Filter PPE detections
        filtered_ppe = [
            d for d in raw_candidates
            if d["category"] not in ["person", "worker"]
        ]
        filtered_ppe += self._detect_aux_ppe(image_or_frame, every_n=aux_every_n)

        # 2. Filter & Deduplicate Person boxes (NMS + Containment suppression)
        person_candidates = [
            d for d in raw_candidates
            if d["category"] == "person" and d["confidence"] >= min_thresholds["person"]
        ]
        coco_persons = self._detect_coco_persons(image_or_frame)
        if coco_persons:
            # The COCO model is the authority on people (already NMS'd by YOLO). The PPE
            # model often returns one high-confidence box around two neighbouring workers;
            # ranking that first suppressed both real boxes and counted 2 workers as 1.
            clean_persons = self._drop_group_boxes(coco_persons)
            for cand in sorted(person_candidates, key=lambda x: x["confidence"], reverse=True):
                if not any(self._same_or_group(cand["bbox"], k["bbox"]) for k in clean_persons):
                    clean_persons.append(cand)
            return clean_persons + filtered_ppe

        person_candidates = sorted(person_candidates, key=lambda x: x["confidence"], reverse=True)

        clean_persons = []
        for cand in person_candidates:
            b1 = cand["bbox"]
            w1 = b1[2] - b1[0]
            h1 = b1[3] - b1[1]
            cx1 = (b1[0] + b1[2]) / 2
            a1 = max(1, w1 * h1)
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

                # Suppress if almost fully nested (>85%) or high IoU (>0.50). Looser values
                # merged neighbouring workers standing shoulder to shoulder.
                if containment > 0.85 or iou_val > 0.50:
                    keep = False
                    break

                # Suppress if moderate overlap in the same vertical column (dx < 25% width)
                if (containment > 0.50 or iou_val > 0.30) and dx < 0.25 * min(w1, w2):
                    keep = False
                    break

            if keep:
                clean_persons.append(cand)

        # (Inventing "workers" around stray vest boxes was removed: with the COCO person
        # model they are no longer needed, and they produced phantom workers.)
        return clean_persons + filtered_ppe
