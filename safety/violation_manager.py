"""
Safety Module: Temporal Violation Manager
========================================
Maintains temporal persistence buffers per worker across all 8 PPE categories
to verify continuous violations and suppress single-frame transient false alarms.
"""
import time
import config
from safety.safety_rules import SafetyRuleEngine, PPE_CATEGORY_CONFIG


class WorkerTemporalState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        # Per-PPE missing frames counter dictionary for all 8 categories
        self.ppe_missing_frames = {item["key"]: 0 for item in PPE_CATEGORY_CONFIG}
        self.zone_frames = 0
        self.current_zone_type = "SAFE"
        self.current_zone_id = "general_site"
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.active_incident_keys = set()
        self.incident_durations = {}

    @property
    def helmet_missing_frames(self):
        return self.ppe_missing_frames.get("helmet", 0)

    @property
    def vest_missing_frames(self):
        return self.ppe_missing_frames.get("vest", 0)

    def update(self, worker_dict, zone_type, zone_id=None):
        self.last_seen_time = time.time()
        self.current_zone_type = zone_type
        if zone_id:
            self.current_zone_id = zone_id

        # Update each of the 8 PPE items
        for item in PPE_CATEGORY_CONFIG:
            key = item["key"]
            val = worker_dict.get(key, True)
            if val is False or val == "missing":
                self.ppe_missing_frames[key] += 1
            elif val is True:
                self.ppe_missing_frames[key] = 0
            # If uncertain, hold counter unchanged without incrementing or resetting

        # Update Zone Temporal Counter
        if zone_type not in ["SAFE", "NORMAL"]:
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
        """
        current_time = time.time()
        new_incidents = []
        updated_workers = []

        seen_worker_ids = set()

        for w in enriched_workers:
            raw_wid = w.get("worker_id", "1")
            wid = str(raw_wid)
            seen_worker_ids.add(wid)

            if wid not in self.worker_states:
                state_obj = WorkerTemporalState(wid)
                self.worker_states[wid] = state_obj
                self.worker_states[raw_wid] = state_obj

            state = self.worker_states[wid]
            zone_type = w.get("zone_type", "SAFE")
            zone_id = w.get("zone_id", "general_site")

            state.update(w, zone_type, zone_id)

            # Build state object for rule engine
            eval_state = {
                "worker_id": wid,
                "zone_type": state.current_zone_type,
                "zone_id": state.current_zone_id,
                "zone_frames": state.zone_frames,
                "min_violation_frames": self.min_violation_frames,
                "min_zone_frames": self.min_zone_frames,
                "missing_ppe": w.get("missing_ppe", []),
                "ppe_missing_frames": state.ppe_missing_frames
            }
            # Add all 8 PPE flags
            for item in PPE_CATEGORY_CONFIG:
                eval_state[item["key"]] = w.get(item["key"], True)

            # Evaluate rules
            active_violations = self.rule_engine.evaluate_rules(eval_state)
            current_v_types = set()

            for v in active_violations:
                v_type = v["type"]
                current_v_types.add(v_type)

                if v_type not in state.active_incident_keys:
                    state.active_incident_keys.add(v_type)
                    state.incident_durations[v_type] = current_time

                    new_incidents.append({
                        "worker_id": wid,
                        "label": w.get("label", f"Worker #{wid}"),
                        "violation_type": v_type,
                        "description": v["description"],
                        "severity": v["severity"],
                        "confidence": w.get("confidence", 0.9),
                        "bbox": w.get("bbox", [0, 0, 0, 0]),
                        "timestamp": current_time,
                        "zone_name": w.get("zone_name", "General Site")
                    })

            # Check if previous violations resolved
            resolved_keys = state.active_incident_keys - current_v_types
            for rk in resolved_keys:
                state.active_incident_keys.remove(rk)
                state.incident_durations.pop(rk, None)

            # Assign status to worker
            worker_copy = dict(w)
            worker_copy["violations"] = active_violations
            if any(v.get("severity") == "CRITICAL" for v in active_violations):
                worker_copy["status"] = "CRITICAL VIOLATION"
            elif active_violations:
                worker_copy["status"] = "CONFIRMED VIOLATION"
            elif any(cnt > 0 for cnt in state.ppe_missing_frames.values()) or state.zone_frames > 0:
                worker_copy["status"] = "PENDING VERIFICATION"
            else:
                worker_copy["status"] = "COMPLIANT"

            updated_workers.append(worker_copy)

        # Cleanup stale worker states after 60 seconds of inactivity
        stale_ids = [wid for wid, s in self.worker_states.items() if current_time - s.last_seen_time > 60.0]
        for sid in stale_ids:
            del self.worker_states[sid]

        return updated_workers, new_incidents
