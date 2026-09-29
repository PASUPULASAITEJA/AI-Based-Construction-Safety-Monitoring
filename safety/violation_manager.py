"""
Safety Module: Temporal Violation Manager
Maintains temporal buffers per worker to verify persistent violations and eliminate transient false alerts using EWMA smoothing.
"""
import time
import config
from safety.safety_rules import SafetyRuleEngine

class WorkerTemporalState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        
        # All 8 PPE classes
        self.ppe_classes = ["helmet", "vest", "boots", "gloves", "goggles", "ear_muffs", "harness", "chaps"]
        
        # EWMA smoothed confidence scores (initialize to 0.0)
        self.ewma_scores = {ppe: 0.0 for ppe in self.ppe_classes}
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

    def update(self, ppe_detections, zone_type):
        """
        Updates the EWMA score for all PPE classes.
        ppe_detections: dict mapping ppe class name to YOLO confidence score (0.0 if not detected)
        """
        now = time.time()
        self.last_seen_time = now
        self.current_zone_type = zone_type
        self.frames_seen += 1

        # Update EWMA for all PPE
        for ppe in self.ppe_classes:
            raw_conf = ppe_detections.get(ppe, 0.0)
            
            if self.is_first_update:
                self.ewma_scores[ppe] = raw_conf
            else:
                # $S_t = \alpha \cdot P_t + (1 - \alpha) \cdot S_{t-1}$
                self.ewma_scores[ppe] = (self.alpha * raw_conf) + ((1.0 - self.alpha) * self.ewma_scores[ppe])
                
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
            
            # Map the raw detections into the ppe_detections dict expected by EWMA update
            ppe_detections = {}
            for ppe in state.ppe_classes:
                # In a real implementation, ppe_confidence would be the highest confidence among assigned bboxes
                # For this transition, we map binary state to confidence (1.0 or 0.0)
                # If they have actual confidence scores in w['assigned_ppe'], we'd use those.
                has_ppe = w.get(f"has_{ppe}", False)
                # If True, simulate high confidence. If False, simulate 0.0.
                # In future updates, this should pass the actual YOLO float confidence.
                ppe_detections[ppe] = 0.9 if has_ppe else 0.0

            # Update temporal state (EWMA)
            state.update(ppe_detections, w.get("zone_type", "SAFE"))

            # Build state dictionary for the rule engine
            rule_state = {
                "worker_id": wid,
                "zone_type": state.current_zone_type,
                "zone_frames": state.zone_frames,
                "zone_duration": state.zone_duration,
            }
            
            # Feed the EWMA threshold results to the rule engine
            for ppe in state.ppe_classes:
                # If the smoothed score is above the threshold, consider it present
                rule_state[f"is_{ppe}_present"] = (state.ewma_scores[ppe] >= self.ewma_threshold)

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
