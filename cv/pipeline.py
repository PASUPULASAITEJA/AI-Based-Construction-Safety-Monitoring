"""
CV Module: Master Safety Monitoring Pipeline
Coordinates Detection, Tracking, PPE Association, Zone Reasoning, Temporal Verification, and Annotation.
"""
import time
import datetime
import threading
import cv2
import config
from cv.detector import ConstructionDetector
from cv.tracker import WorkerTracker
from cv.ppe_association import PPEAssociator
from cv.zone_detector import ZoneDetector
from cv.annotator import FrameAnnotator
from safety.violation_manager import ViolationManager
from safety.alert_manager import AlertManager
from safety.ppe_config import PPE_CATEGORIES
from database.models import ZoneModel, WorkerTrackModel, SettingsModel

PPE_CATEGORY_IDS = [cat["category"] for cat in PPE_CATEGORIES]

# Accepted ranges for values coming from the Settings page
SETTING_LIMITS = {
    "conf": (0.01, 0.99),
    "iou": (0.05, 0.95),
    "min_v_frames": (1, 300),
    "min_z_frames": (1, 300),
}

def _session_tag():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]

def _check_range(name, value, cast):
    try:
        value = cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    lo, hi = SETTING_LIMITS[name]
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}.")
    return value

class SafetyPipeline:
    def __init__(self, model_path=None, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD, detector=None):
        # Pass an existing detector to share one loaded YOLO model between pipelines
        self.detector = detector or ConstructionDetector(model_path=model_path, conf=conf, iou=iou)
        self.tracker = WorkerTracker()
        self.ppe_associator = PPEAssociator()
        self.zone_detector = ZoneDetector()
        self.annotator = FrameAnnotator()
        self.violation_manager = ViolationManager()
        self.alert_manager = AlertManager()

        # One pipeline instance may be used from several request threads
        self.lock = threading.RLock()

        self.last_frame_time = time.time()
        self.fps = 0.0
        self.source_name = "Camera"

        # Tracker IDs restart at 1 for every pipeline session; the tag keeps workers_track rows distinct
        self.session_tag = _session_tag()
        self._last_observation = {}  # worker_id -> time of last workers_track write

        # Load saved zones and Settings-page thresholds from DB
        self.reload_zones()
        self.load_saved_settings()

    def reset(self):
        """Starts a fresh tracking session (new worker IDs, no carried-over temporal state)."""
        with self.lock:
            self.tracker.reset()
            self.violation_manager.reset()
            self.session_tag = _session_tag()
            self._last_observation = {}

    def reload_zones(self):
        try:
            db_zones = ZoneModel.get_all()
            with self.lock:
                self.zone_detector.set_zones(db_zones)
        except Exception as e:
            print(f"[-] Could not load zones from DB: {e}")

    def load_saved_settings(self):
        """Applies thresholds saved from the Settings page so they survive restarts."""
        try:
            saved = SettingsModel.get_all()
            self.update_settings(
                conf=saved.get("conf_thresh"),
                iou=saved.get("iou_thresh"),
                min_v_frames=saved.get("min_violation_frames"),
                min_z_frames=saved.get("min_zone_frames")
            )
        except Exception as e:
            print(f"[-] Could not load saved settings from DB: {e}")

    @staticmethod
    def validate_settings(conf=None, iou=None, min_v_frames=None, min_z_frames=None):
        """Returns the values parsed and range-checked; raises ValueError with a readable message."""
        return {
            "conf": _check_range("conf", conf, float) if conf is not None else None,
            "iou": _check_range("iou", iou, float) if iou is not None else None,
            "min_v_frames": _check_range("min_v_frames", min_v_frames, int) if min_v_frames is not None else None,
            "min_z_frames": _check_range("min_z_frames", min_z_frames, int) if min_z_frames is not None else None,
        }

    def update_settings(self, conf=None, iou=None, min_v_frames=None, min_z_frames=None):
        values = self.validate_settings(conf, iou, min_v_frames, min_z_frames)
        with self.lock:
            if values["conf"] is not None:
                self.detector.conf = values["conf"]
            if values["iou"] is not None:
                self.detector.iou = values["iou"]
            self.violation_manager.set_thresholds(values["min_v_frames"], values["min_z_frames"])

    def _track_code(self, worker_id):
        return f"TRK-{self.session_tag}-{worker_id:03d}"

    def _record_compliant_observation(self, worker, now):
        """Throttled workers_track write so a 30 FPS stream does not hit SQLite for every worker every frame."""
        wid = worker["worker_id"]
        if now - self._last_observation.get(wid, 0.0) < config.WORKER_OBSERVATION_INTERVAL_SEC:
            return
        self._last_observation[wid] = now
        WorkerTrackModel.record_observation(track_code=self._track_code(wid), label=worker["label"],
                                            is_compliant=True, has_violation=False)

    def process_frame(self, frame, source="Webcam", site_id=None, camera_id=None):
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
                "critical_count": int,
                "workers": list of worker summaries (violations as description strings),
                "new_incidents": list of newly logged incidents
            }
        )
        """
        with self.lock:
            now = time.time()
            dt = now - self.last_frame_time
            self.fps = (1.0 / dt) if dt > 0 else 30.0
            self.last_frame_time = now

            # 1. Object Detection
            detections = self.detector.detect(frame)

            # 2. Worker Filtering & Tracking
            person_detections = [d for d in detections if d["category"] == "person"]
            tracked_workers = self.tracker.update(person_detections)

            # 3. Spatial PPE Association
            enriched_workers = self.ppe_associator.associate(tracked_workers, detections, frame=frame)

            # 4. Zone Analysis
            enriched_workers = self.zone_detector.evaluate_workers(enriched_workers, frame.shape)

            # 5. Temporal Verification & Safety Rule Engine
            verified_workers, new_incidents = self.violation_manager.process_frame(enriched_workers)

            # 6. Record New Incidents to Database & Snapshots
            logged_incidents = []
            for inc in new_incidents:
                inc["track_code"] = self._track_code(inc["worker_id"])
                logged_incidents.append(self.alert_manager.record_incident(
                    inc, frame=frame, source=source, site_id=site_id, camera_id=camera_id))

            # Record compliant observations for workers without violations
            for w in verified_workers:
                if not w.get("violations"):
                    self._record_compliant_observation(w, now)

            # 7. Visual Overlay Annotation
            annotated_frame = self.annotator.annotate(
                frame=frame,
                workers=verified_workers,
                zones=self.zone_detector.zones_for_frame(frame.shape),
                fps=self.fps
            )

            # Build Frame Summary
            total_workers = len(verified_workers)
            violation_workers = sum(1 for w in verified_workers if w.get("violations"))
            compliant_workers = total_workers - violation_workers

            worker_summaries = []
            for w in verified_workers:
                w_sum = {
                    "worker_id": w["worker_id"],
                    "label": w["label"],
                    "helmet": w.get("helmet", False),
                    "vest": w.get("vest", False),
                    "zone_type": w.get("zone_type", "SAFE"),
                    "zone_name": w.get("zone_name", "SAFE"),
                    "status": w.get("status", "COMPLIANT"),
                    "violations": [v["description"] for v in w.get("violations", [])]
                }
                # Append all 8 dynamic PPE categories
                for cat in PPE_CATEGORY_IDS:
                    w_sum[f"has_{cat}"] = w.get(f"has_{cat}", False)
                worker_summaries.append(w_sum)

            summary = {
                "fps": round(self.fps, 1),
                "worker_count": total_workers,
                "compliant_count": compliant_workers,
                "violation_count": violation_workers,
                "critical_count": sum(1 for w in worker_summaries if "CRITICAL" in w["status"]),
                "workers": worker_summaries,
                "new_incidents": logged_incidents
            }

            return annotated_frame, summary

    def process_single_image(self, image_input, source="Image Upload", site_id=None, camera_id=None):
        """
        Processes a single still image (without temporal tracking requirement).
        Immediate spatial association and rule checking applied.
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
                if iou > 0.50 or containment > 0.75:
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

        # Every image is its own session, so worker numbers from different uploads stay distinct
        image_tag = _session_tag()

        with self.lock:
            # 3. PPE Association
            enriched = self.ppe_associator.associate(instant_workers, detections, frame=frame)

            # 4. Zone Analysis
            enriched = self.zone_detector.evaluate_workers(enriched, frame.shape)

            # 5. Direct Rule Evaluation for single image (no temporal smoothing possible)
            rule_engine = self.violation_manager.rule_engine
            annotated_workers = []
            detected_incidents = []

            for w in enriched:
                zone_type = w.get("zone_type", "SAFE")
                eval_state = {
                    "worker_id": w["worker_id"],
                    "zone_type": zone_type,
                    "zone_frames": 1 if zone_type != "SAFE" else 0,
                    "min_zone_frames": 1,
                }
                for cat in PPE_CATEGORY_IDS:
                    eval_state[f"is_{cat}_present"] = w.get(f"has_{cat}", False)

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

                track_code = f"TRK-{image_tag}-{w['worker_id']:03d}"
                for v in v_list:
                    inc_entry = self.alert_manager.record_incident({
                        "worker_id": w["worker_id"],
                        "label": w["label"],
                        "violation_type": v["type"],
                        "description": v["description"],
                        "severity": v["severity"],
                        "confidence": w["confidence"],
                        "bbox": w["bbox"],
                        "zone_id": w.get("zone_id"),
                        "track_code": track_code
                    }, frame=frame, source=source, site_id=site_id, camera_id=camera_id)
                    detected_incidents.append(inc_entry)

                if not v_list:
                    WorkerTrackModel.record_observation(track_code=track_code, label=w["label"], is_compliant=True, has_violation=False)

            # Annotate
            annotated_frame = self.annotator.annotate(
                frame=frame,
                workers=annotated_workers,
                zones=self.zone_detector.zones_for_frame(frame.shape),
                fps=0
            )

        return annotated_frame, {
            "worker_count": len(annotated_workers),
            "compliant_count": sum(1 for w in annotated_workers if not w["violations"]),
            "violation_count": sum(1 for w in annotated_workers if w["violations"]),
            "workers": annotated_workers,
            "incidents": detected_incidents,
            "raw_detections": detections
        }
