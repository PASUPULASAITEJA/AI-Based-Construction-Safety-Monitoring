import time
import config
from safety.safety_rules import SafetyRuleEngine
from safety.ppe_config import PPE_CATEGORIES

class WorkerTemporalState:
    def __init__(self, worker_id):
        self.worker_id = worker_id

        # Track by Category ID (HEAD, VISIBILITY, etc.)
        self.ppe_categories = [cat["category"] for cat in PPE_CATEGORIES]

        # Consecutive frames each PPE category has NOT been detected on this worker
        self.missing_frames = {cat: 0 for cat in self.ppe_categories}

        # EWMA of per-frame detection confidence, exposed to the UI for debugging
        self.ewma_scores = {cat: 0.0 for cat in self.ppe_categories}
        self.is_first_update = True
        self.alpha = 0.15

        self.zone_frames = 0
        self.current_zone_type = "SAFE"
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()

        self.zone_entry_since = None
        self.active_incident_keys = set()  # violation types currently active for this worker
        self.frames_seen = 0
        self.resolved_at_frame = {}  # violation_type -> frames_seen when it last cleared

    def update(self, category_detections, zone_type):
        """
        category_detections: dict mapping category name to YOLO confidence score (0.0 if not detected)
        """
        now = time.time()
        self.last_seen_time = now
        self.current_zone_type = zone_type
        self.frames_seen += 1

        for cat in self.ppe_categories:
            raw_conf = category_detections.get(cat, 0.0)
            if raw_conf > 0.0:
                self.missing_frames[cat] = 0
            else:
                self.missing_frames[cat] += 1

            if self.is_first_update:
                self.ewma_scores[cat] = raw_conf
            else:
                self.ewma_scores[cat] = (self.alpha * raw_conf) + ((1.0 - self.alpha) * self.ewma_scores[cat])
        self.is_first_update = False

        # Update Zone Temporal Counter & Timestamp
        if zone_type != "SAFE":
            self.zone_frames += 1
            if self.zone_entry_since is None:
                self.zone_entry_since = now
        else:
            self.zone_frames = 0
            self.zone_entry_since = None

    @property
    def zone_duration(self):
        return (time.time() - self.zone_entry_since) if self.zone_entry_since else 0.0

class ViolationManager:
    def __init__(
        self,
        min_violation_frames=config.MIN_VIOLATION_FRAMES,
        min_zone_frames=config.MIN_ZONE_FRAMES,
        cooldown_frames=config.VIOLATION_COOLDOWN_FRAMES,
        track_timeout_sec=10.0
    ):
        self.min_violation_frames = int(min_violation_frames)
        self.min_zone_frames = int(min_zone_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.track_timeout_sec = track_timeout_sec
        self.worker_states = {}  # worker_id -> WorkerTemporalState
        self.rule_engine = SafetyRuleEngine()

    def set_thresholds(self, min_violation_frames=None, min_zone_frames=None):
        if min_violation_frames is not None:
            self.min_violation_frames = max(1, int(min_violation_frames))
        if min_zone_frames is not None:
            self.min_zone_frames = max(1, int(min_zone_frames))

    def reset(self):
        self.worker_states = {}

    def process_frame(self, enriched_workers):
        """
        Updates temporal state for all tracked workers and evaluates zone-based safety rules.
        A PPE item only counts as missing after min_violation_frames consecutive frames without it,
        and a violation that just cleared is not re-raised within cooldown_frames.

        Each returned worker gets "violations" (list of rule dicts: type, description, severity, ...),
        "status" and "is_compliant".
        """
        now = time.time()
        verified_workers = []
        new_incidents = []

        # Cleanup lost tracks
        dead_workers = [wid for wid, st in self.worker_states.items() if (now - st.last_seen_time) > self.track_timeout_sec]
        for wid in dead_workers:
            del self.worker_states[wid]

        for w in enriched_workers:
            wid = w.get("worker_id")
            if wid is None:
                verified_workers.append(w)
                continue

            if wid not in self.worker_states:
                self.worker_states[wid] = WorkerTemporalState(wid)
            state = self.worker_states[wid]

            # ppe_association maps the best item confidence to {cat}_conf (0.0 when not detected)
            category_detections = {cat: w.get(f"{cat}_conf", 0.0) for cat in state.ppe_categories}
            state.update(category_detections, w.get("zone_type", "SAFE"))

            rule_state = {
                "worker_id": wid,
                "zone_type": state.current_zone_type,
                "zone_frames": state.zone_frames,
                "zone_duration": state.zone_duration,
                "min_zone_frames": self.min_zone_frames,
            }
            for cat in state.ppe_categories:
                rule_state[f"is_{cat}_present"] = state.missing_frames[cat] < self.min_violation_frames

            violations = self.rule_engine.evaluate_rules(rule_state)

            # Raise NEW incidents (respecting the per-type cooldown after a violation cleared)
            for v in violations:
                v_type = v["type"]
                if v_type in state.active_incident_keys:
                    continue
                state.active_incident_keys.add(v_type)
                cleared_at = state.resolved_at_frame.get(v_type)
                if cleared_at is not None and state.frames_seen - cleared_at < self.cooldown_frames:
                    continue

                new_incidents.append({
                    "worker_id": wid,
                    "label": w.get("label", f"Worker #{wid}"),
                    "violation_type": v_type,
                    "description": v["description"],
                    "severity": v["severity"],
                    "is_critical": v["is_critical"],
                    "confidence": float(w.get("confidence", 0.0)),
                    "bbox": w.get("bbox"),
                    "zone_id": w.get("zone_id"),
                    "zone_name": w.get("zone_name", "SAFE"),
                    "duration_sec": round(now - state.first_seen_time, 2),
                    "timestamp_start": now
                })

            # Clear incidents that are no longer active
            current_v_types = {v["type"] for v in violations}
            for rt in state.active_incident_keys - current_v_types:
                state.active_incident_keys.remove(rt)
                state.resolved_at_frame[rt] = state.frames_seen

            w["violations"] = violations
            w["is_compliant"] = len(violations) == 0
            if any(v["severity"] == "CRITICAL" for v in violations):
                w["status"] = "CRITICAL VIOLATION"
            elif violations:
                w["status"] = "CONFIRMED VIOLATION"
            else:
                w["status"] = "COMPLIANT"
            # Inject EWMA scores into worker dict for debugging/UI if needed
            w["ewma_scores"] = state.ewma_scores.copy()

            verified_workers.append(w)

        return verified_workers, new_incidents
