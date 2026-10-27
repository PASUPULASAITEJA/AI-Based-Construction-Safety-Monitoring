"""
CV Module: Master Safety Monitoring Pipeline
Coordinates Detection, Tracking, PPE Association, Zone Reasoning, Temporal Verification, and Annotation.
"""
import time
import cv2
import config
from cv.detector import ConstructionDetector
from cv.tracker import WorkerTracker
from cv.ppe_association import PPEAssociator
from cv.zone_detector import ZoneDetector
from cv.annotator import FrameAnnotator
from safety.violation_manager import ViolationManager
from safety.alert_manager import AlertManager
from database.models import ZoneModel
from cv.small_ppe_handler import TemporalPPETracker, HighResZoneInference

class SafetyPipeline:
    def __init__(self, model_path=None, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD):
        self.detector = ConstructionDetector(model_path=model_path, conf=conf, iou=iou)
        self.tracker = WorkerTracker(min_hits=3)
        self.ppe_associator = PPEAssociator()
        self.temporal_tracker = TemporalPPETracker()
        self.high_res_zone_inference = HighResZoneInference(detector_model=getattr(self.detector, 'model', None))
        self.zone_detector = ZoneDetector()
        self.annotator = FrameAnnotator()
        self.violation_manager = ViolationManager()
        self.alert_manager = AlertManager()

        self.last_frame_time = time.time()
        self.fps = 0.0
        self.source_name = "Camera"

        # Load saved zones from DB
        self.reload_zones()

    def reload_zones(self):
        try:
            db_zones = ZoneModel.get_all()
            self.zone_detector.set_zones(db_zones)
        except Exception as e:
            print(f"[-] Could not load zones from DB: {e}")

    def update_settings(self, conf=None, iou=None, min_v_frames=None, min_z_frames=None):
        if conf is not None:
            self.detector.conf = float(conf)
        if iou is not None:
            self.detector.iou = float(iou)
        if min_v_frames is not None or min_z_frames is not None:
            self.violation_manager.set_thresholds(min_v_frames, min_z_frames)

    def _filter_live_persons(self, persons, frame_shape):
        """
        Stricter person filtering for live camera frames, where people sit close to the
        lens. Measured on a webcam close-up of one person, the count read 3-4 because of
        (a) a sliver of an arm/shoulder at the frame edge and (b) small low-confidence
        boxes beside the face.
        """
        h, w = frame_shape[:2]
        if any(p.get("source") == "coco" for p in persons):
            # The PPE model's "Person" class is tuned for construction photos and misfires on close-ups
            persons = [p for p in persons if p.get("source") == "coco"]
        persons = [p for p in persons if p["confidence"] >= 0.35]
        if not persons:
            return []

        max_h = max(p["bbox"][3] - p["bbox"][1] for p in persons)
        kept = []
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            bw, bh = x2 - x1, y2 - y1
            conf = p["confidence"]
            at_side_edge = x1 <= 0.01 * w or x2 >= 0.99 * w
            if at_side_edge and bw < 0.20 * w and conf < 0.65:
                continue  # partial arm / shoulder cut off by the frame
            if bh < 0.35 * max_h and conf < 0.60:
                continue  # small fragment next to a much larger (closer) person
            kept.append(p)
        return kept

    def process_frame(self, frame, source="Webcam"):
        """
        Executes full safety pipeline on a single video/camera frame.
        
        Returns:
        (
            annotated_frame: np.ndarray,
            frame_summary: {
                "fps": float,
                "worker_count": int,
                "compliant_count": int,
                "violation_count": int,
                "workers": list of worker summaries,
                "new_incidents": list of newly logged incidents
            }
        )
        """
        now = time.time()
        dt = now - self.last_frame_time
        self.fps = (1.0 / dt) if dt > 0 else 30.0
        self.last_frame_time = now

        # 1. Object Detection
        detections = self.detector.detect(frame, aux_every_n=getattr(config, "AUX_LIVE_EVERY_N_FRAMES", 1))

        # 2. Worker Filtering & Tracking
        person_detections = self._filter_live_persons(
            [d for d in detections if d["category"] == "person"], frame.shape)
        tracked_workers = self.tracker.update(person_detections)

        # Optional second-pass zoomed crop inference for small PPE (only on GPU to maintain real-time FPS)
        if config.DEVICE == "cuda" and getattr(self.detector, 'model', None) and len(tracked_workers) > 0:
            for w in tracked_workers:
                head_zone = self.ppe_associator.get_head_zone(w["bbox"])
                zoomed_dets = self.high_res_zone_inference.run_zoomed_inference(
                    frame=frame,
                    zone_box=head_zone,
                    target_categories=["goggles", "no-goggles", "earmuff", "no-earmuff"]
                )
                if zoomed_dets:
                    detections.extend(zoomed_dets)

        # 3. Spatial PPE Association with Temporal Verification
        # With no confirmed tracks, associate() would fall back to the raw (unfiltered)
        # person boxes in `detections` and count them as workers, so skip it.
        enriched_workers = []
        if tracked_workers:
            enriched_workers = self.ppe_associator.associate(
                detections=detections,
                workers=tracked_workers,
                frame=frame,
                temporal_tracker=self.temporal_tracker
            )
        if isinstance(enriched_workers, dict):
            enriched_workers = list(enriched_workers.values())

        # 4. Zone Analysis
        enriched_workers = self.zone_detector.evaluate_workers(enriched_workers)

        # 5. Temporal Verification & Safety Rule Engine
        verified_workers, new_incidents = self.violation_manager.process_frame(enriched_workers)

        # 6. Record New Incidents to Database & Snapshots
        for inc in new_incidents:
            try:
                self.alert_manager.record_incident(inc, frame=frame, source=source)
            except Exception as e:
                print(f"[-] Could not record incident: {e}")

        # 7. Visual Overlay Annotation
        annotated_frame = self.annotator.annotate(
            frame=frame,
            workers=verified_workers,
            zones=self.zone_detector.zones,
            fps=self.fps
        )

        # Build Frame Summary with all 8 PPE categories
        total_workers = len(verified_workers)
        violation_workers = sum(1 for w in verified_workers if w.get("violations") or w.get("compliance_score", 1.0) < 0.8)
        compliant_workers = total_workers - violation_workers

        worker_summaries = []
        for w in verified_workers:
            worker_summaries.append({
                "worker_id": w["worker_id"],
                "label": w.get("label", f"Worker #{w['worker_id']}"),
                "helmet": w.get("helmet", False),
                "goggles": w.get("goggles", False),
                "earmuff": w.get("earmuff", False),
                "vest": w.get("vest", False),
                "harness": w.get("harness", False),
                "safety_pants": w.get("safety_pants", False),
                "gloves": w.get("gloves", False),
                "boots": w.get("boots", False),
                "compliance_score": w.get("compliance_score", 0.0),
                "missing_ppe": w.get("missing_ppe", []),
                "zone_type": w.get("zone_type", "SAFE"),
                "zone_name": w.get("zone_name", "SAFE"),
                "status": w.get("status", "COMPLIANT"),
                "violations": [v.get("type", v.get("missing_item", "Violation")) for v in w.get("violations", [])]
            })

        summary = {
            "fps": round(self.fps, 1),
            "worker_count": total_workers,
            "compliant_count": compliant_workers,
            "violation_count": violation_workers,
            "workers": worker_summaries,
            "new_incidents": new_incidents
        }

        return annotated_frame, summary

    def process_single_image(self, image_input, source="Image Upload", style="full"):
        """
        Processes a single still image (without temporal tracking requirement).
        Immediate spatial association and rule checking applied.
        
        Args:
            image_input: image numpy array or filepath string
            source: Source metadata label
            style: 'full' (rich HUD + checklist panels + zones) or 'plain' (clean per-class boxes)
        """
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        else:
            frame = image_input

        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Extract Person boxes with clean NMS deduplication
        person_detections = [d for d in detections if d["category"] == "person"]
        sorted_pd = sorted(person_detections, key=lambda x: x["confidence"], reverse=True)
        clean_persons = []
        for pd in sorted_pd:
            keep = True
            b1 = pd["bbox"]
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            for kept in clean_persons:
                b2 = kept["bbox"]
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0
                containment = inter / min(a1, a2) if min(a1, a2) > 0 else 0
                # The detector already de-duplicated people; only drop exact repeats here, so a
                # worker standing partly behind another (box nested in theirs) is kept.
                if iou > 0.70:
                    keep = False
                    break
            if keep:
                clean_persons.append(pd)

        # Sort left-to-right for consistent labeling
        clean_persons = sorted(clean_persons, key=lambda x: x["bbox"][0])
        instant_workers = []
        for i, pd in enumerate(clean_persons):
            instant_workers.append({
                "worker_id": i + 1,
                "label": f"Worker #{i+1}",
                "bbox": pd["bbox"],
                "confidence": pd["confidence"]
            })

        # 3. PPE Association
        enriched = self.ppe_associator.associate(instant_workers, detections, frame=frame)
        if isinstance(enriched, dict):
            enriched = list(enriched.values())

        # 4. Zone Analysis
        enriched = self.zone_detector.evaluate_workers(enriched)

        # 5. Direct Rule Evaluation for single image
        rule_engine = self.violation_manager.rule_engine
        annotated_workers = []
        detected_incidents = []

        for w in enriched:
            eval_state = {
                "worker_id": w["worker_id"],
                "helmet_missing_frames": 999 if not w["helmet"] else 0,
                "vest_missing_frames": 999 if not w["vest"] else 0,
                "zone_type": w.get("zone_type", "SAFE"),
                "zone_frames": 999 if w.get("zone_type") != "SAFE" else 0,
                "min_violation_frames": 1,
                "min_zone_frames": 1,
                "is_helmet_present": w["helmet"],
                "is_vest_present": w["vest"]
            }
            v_list = rule_engine.evaluate_rules(eval_state)
            w_copy = dict(w)
            w_copy["violations"] = v_list
            if any(v["severity"] == "CRITICAL" for v in v_list):
                w_copy["status"] = "CRITICAL VIOLATION"
            elif v_list:
                w_copy["status"] = "CONFIRMED VIOLATION"
            else:
                w_copy["status"] = "COMPLIANT"

            annotated_workers.append(w_copy)

            for v in v_list:
                # Logging an incident must never turn a finished inspection into an error
                try:
                    inc_entry = self.alert_manager.record_incident({
                        "worker_id": w["worker_id"],
                        "label": w["label"],
                        "violation_type": v["type"],
                        "description": v["description"],
                        "severity": v["severity"],
                        "confidence": w["confidence"],
                        "bbox": w["bbox"]
                    }, frame=frame, source=source)
                except Exception as e:
                    print(f"[-] Could not record incident: {e}")
                    inc_entry = {"violation_type": v["type"], "severity": v["severity"]}
                detected_incidents.append(inc_entry)

        # Annotate
        annotated_frame = self.annotator.annotate(
            frame=frame,
            workers=annotated_workers,
            zones=self.zone_detector.zones,
            fps=0,
            style=style,
            raw_detections=detections
        )

        return annotated_frame, {
            "worker_count": len(annotated_workers),
            "compliant_count": sum(1 for w in annotated_workers if not w["violations"]),
            "violation_count": sum(1 for w in annotated_workers if w["violations"]),
            "workers": annotated_workers,
            "incidents": detected_incidents,
            "raw_detections": len(detections),
            "supported_ppe": self.detector.supported_ppe
        }
