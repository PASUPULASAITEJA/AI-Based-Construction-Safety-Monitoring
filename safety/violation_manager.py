"""
Safety Module: Temporal Violation Manager
Maintains temporal buffers per worker to verify persistent violations and eliminate transient false alerts.
"""
import time
import config
from safety.safety_rules import SafetyRuleEngine

class WorkerTemporalState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.helmet_missing_frames = 0
        self.vest_missing_frames = 0
        self.zone_frames = 0
        self.current_zone_type = "SAFE"
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.active_incident_keys = set() # (violation_type) -> start_time
        self.incident_durations = {}

    def update(self, has_helmet, has_vest, zone_type):
        self.last_seen_time = time.time()
        self.current_zone_type = zone_type

        # Update Helmet Temporal Counter
        if not has_helmet:
            self.helmet_missing_frames += 1
        else:
            # PPE detected again before threshold or resolved -> reset immediately
            self.helmet_missing_frames = 0

        # Update Vest Temporal Counter
        if not has_vest:
            self.vest_missing_frames += 1
        else:
            self.vest_missing_frames = 0

        # Update Zone Temporal Counter
        if zone_type != "SAFE":
            self.zone_frames += 1
        else:
            self.zone_frames = 0

class ViolationManager:
    def __init__(
        self,
        min_violation_frames=config.MIN_VIOLATION_FRAMES,
        min_zone_frames=config.MIN_ZONE_FRAMES
    ):
        self.min_violation_frames = min_violation_frames
        self.min_zone_frames = min_zone_frames
        self.worker_states = {} # worker_id -> WorkerTemporalState
        self.rule_engine = SafetyRuleEngine()

    def set_thresholds(self, min_violation_frames=None, min_zone_frames=None):
        if min_violation_frames is not None:
            self.min_violation_frames = int(min_violation_frames)
        if min_zone_frames is not None:
            self.min_zone_frames = int(min_zone_frames)

    def process_frame(self, enriched_workers):
        """
        Updates temporal state for all tracked workers and evaluates safety rules.
        
        enriched_workers: list of worker dicts from PPEAssociator & ZoneDetector
        
        Returns:
        (
            updated_workers: list of worker dicts with attached 'violations' list and 'status',
            new_confirmed_incidents: list of newly triggered incident events to log/alert
        )
        """
        current_time = time.time()
        new_incidents = []
        updated_workers = []

        seen_worker_ids = set()

        for w in enriched_workers:
            wid = w["worker_id"]
            seen_worker_ids.add(wid)

            if wid not in self.worker_states:
                self.worker_states[wid] = WorkerTemporalState(wid)

            state = self.worker_states[wid]
            has_helmet = w.get("helmet", False)
            has_vest = w.get("vest", False)
            zone_type = w.get("zone_type", "SAFE")

            # Update temporal counts
            state.update(has_helmet, has_vest, zone_type)

            # Build state object for rule engine
            eval_state = {
                "worker_id": wid,
                "helmet_missing_frames": state.helmet_missing_frames,
                "vest_missing_frames": state.vest_missing_frames,
                "zone_type": state.current_zone_type,
                "zone_frames": state.zone_frames,
                "min_violation_frames": self.min_violation_frames,
                "min_zone_frames": self.min_zone_frames,
                "is_helmet_present": has_helmet,
                "is_vest_present": has_vest
            }

            # Evaluate rules
            active_violations = self.rule_engine.evaluate_rules(eval_state)
            current_v_types = set()

            for v in active_violations:
                v_type = v["type"]
                current_v_types.add(v_type)

                # Check if this is a newly confirmed incident (not already logged)
                if v_type not in state.active_incident_keys:
                    state.active_incident_keys.add(v_type)
                    state.incident_durations[v_type] = current_time

                    new_incidents.append({
                        "worker_id": wid,
                        "label": w["label"],
                        "violation_type": v_type,
                        "description": v["description"],
                        "severity": v["severity"],
                        "confidence": w["confidence"],
                        "bbox": w["bbox"],
                        "timestamp": current_time,
                        "zone_name": w.get("zone_name", "SAFE")
                    })

            # Check if any previous violations have been resolved
            resolved_keys = state.active_incident_keys - current_v_types
            for rk in resolved_keys:
                state.active_incident_keys.remove(rk)
                state.incident_durations.pop(rk, None)

            # Assign status to worker
            worker_copy = dict(w)
            worker_copy["violations"] = active_violations
            if any(v["severity"] == "CRITICAL" for v in active_violations):
                worker_copy["status"] = "CRITICAL VIOLATION"
            elif active_violations:
                worker_copy["status"] = "CONFIRMED VIOLATION"
            elif state.helmet_missing_frames > 0 or state.vest_missing_frames > 0 or state.zone_frames > 0:
                worker_copy["status"] = "PENDING VERIFICATION"
            else:
                worker_copy["status"] = "COMPLIANT"

            updated_workers.append(worker_copy)

        # Cleanup stale worker states after 60 seconds of inactivity
        stale_ids = [wid for wid, s in self.worker_states.items() if current_time - s.last_seen_time > 60.0]
        for sid in stale_ids:
            del self.worker_states[sid]

        return updated_workers, new_incidents
