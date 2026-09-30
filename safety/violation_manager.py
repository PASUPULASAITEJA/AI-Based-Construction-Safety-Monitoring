import time
import config
from safety.safety_rules import SafetyRuleEngine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.ppe_config import PPE_CATEGORIES, is_ppe_supported

class WorkerTemporalState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        
        # Track by Category ID (HEAD, VISIBILITY, etc.)
        self.ppe_categories = [cat["category"] for cat in PPE_CATEGORIES]
        
        # EWMA smoothed confidence scores (initialize to 0.0)
        self.ewma_scores = {cat: 0.0 for cat in self.ppe_categories}
        self.is_first_update = True
        
        # EWMA hyperparameter (alpha): lower means more smoothing/hysteresis
        self.alpha = 0.15 
        
        self.zone_frames = 0
        self.current_zone_type = "SAFE"
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        
        self.zone_entry_since = None
        self.active_incident_keys = set() # (violation_type) -> start_time
        self.incident_durations = {}
        self.frames_seen = 0
        self.resolved_at_frame = {} # violation_type -> frames_seen when it last cleared

    def update(self, category_detections, zone_type):
        """
        Updates the EWMA score for all PPE categories.
        category_detections: dict mapping category name to YOLO confidence score (0.0 if not detected)
        """
        now = time.time()
        self.last_seen_time = now
        self.current_zone_type = zone_type
        self.frames_seen += 1

        # Update EWMA for all PPE categories
        for cat in self.ppe_categories:
            raw_conf = category_detections.get(cat, 0.0)
            
            if self.is_first_update:
                self.ewma_scores[cat] = raw_conf
            else:
                # $S_t = \alpha \cdot P_t + (1 - \alpha) \cdot S_{t-1}$
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
        min_zone_frames=config.MIN_ZONE_FRAMES,
        min_zone_sec=1.0,
        ewma_threshold=0.45  # The threshold below which a smoothed item is considered missing
    ):
        self.min_zone_frames = min_zone_frames
        self.min_zone_sec = min_zone_sec
        self.ewma_threshold = ewma_threshold
        self.worker_states = {} # worker_id -> WorkerTemporalState
        self.rule_engine = SafetyRuleEngine()

    def process_frame(self, enriched_workers):
        """
        Updates temporal state for all tracked workers using EWMA and evaluates zone-based safety rules.
        """
        now = time.time()
        verified_workers = []
        new_incidents = []

        # Cleanup lost tracks (not seen for > 10 seconds)
        dead_workers = [wid for wid, st in self.worker_states.items() if (now - st.last_seen_time) > 10.0]
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
            
            # Extract category detections from enriched worker
            category_detections = {}
            for cat in state.ppe_categories:
                # The ppe_association.py script maps best item confidence to {cat}_conf
                # We use that if available, otherwise fallback to binary has_{cat} mapping
                conf = w.get(f"{cat}_conf", 0.0)
                if conf == 0.0 and w.get(f"has_{cat}", False):
                    conf = 0.9  # simulated high confidence if flag is present but no conf
                category_detections[cat] = conf

            # Update temporal state (EWMA)
            state.update(category_detections, w.get("zone_type", "SAFE"))

            # Build state dictionary for the rule engine
            rule_state = {
                "worker_id": wid,
                "zone_type": state.current_zone_type,
                "zone_frames": state.zone_frames,
                "zone_duration": state.zone_duration,
            }
            
            # Feed the EWMA threshold results to the rule engine
            for cat in state.ppe_categories:
                # If the smoothed score is above the threshold, consider it present
                rule_state[f"is_{cat}_present"] = (state.ewma_scores[cat] >= self.ewma_threshold)

            # Evaluate Rule Engine
            violations = self.rule_engine.evaluate_rules(rule_state)

            # Check if violations are NEW incidents
            for v in violations:
                v_type = v["type"]
                if v_type not in state.active_incident_keys:
                    state.active_incident_keys.add(v_type)
                    state.incident_durations[v_type] = now
                    
                    incident_record = {
                        "worker_id": wid,
                        "worker_label": w.get("label", f"Worker {wid}"),
                        "violation_type": v_type,
                        "description": v["description"],
                        "severity": v["severity"],
                        "is_critical": v["is_critical"],
                        "confidence": 1.0,  # Or use a derivation of the EWMA score
                        "zone_name": w.get("zone_name", "General"),
                        "timestamp_start": now
                    }
                    new_incidents.append(incident_record)

            # Clear incidents that are no longer active
            current_v_types = {v["type"] for v in violations}
            resolved_types = state.active_incident_keys - current_v_types
            for rt in resolved_types:
                state.active_incident_keys.remove(rt)

            w["violations"] = [v["description"] for v in violations]
            w["is_compliant"] = len(violations) == 0
            # Inject EWMA scores into worker dict for debugging/UI if needed
            w["ewma_scores"] = state.ewma_scores.copy()

            verified_workers.append(w)

        return verified_workers, new_incidents
