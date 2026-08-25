"""
Safety Module: Safety Rule Engine
Evaluates safety compliance rules and determines violation types and severity levels.
"""

class Severity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SafetyRuleEngine:
    def __init__(self):
        pass

    def evaluate_rules(self, worker_state):
        """
        Evaluates safety rules on a single worker state.
        worker_state: dict containing:
          - 'worker_id': int
          - 'helmet_missing_frames': int
          - 'vest_missing_frames': int
          - 'zone_type': 'SAFE' | 'RESTRICTED' | 'HAZARD'
          - 'zone_frames': int
          - 'min_violation_frames': int
          - 'min_zone_frames': int
          - 'is_helmet_present': bool
          - 'is_vest_present': bool
          
        Returns a list of triggered violation dicts:
        [
            {
                "rule_id": 1,
                "type": "NO_HELMET",
                "description": "Worker not wearing safety helmet",
                "severity": "MEDIUM",
                "is_critical": False
            }, ...
        ]
        """
        violations = []
        min_v_frames = worker_state.get("min_violation_frames", 10)
        min_z_frames = worker_state.get("min_zone_frames", 8)

        helmet_missing_frames = worker_state.get("helmet_missing_frames", 0)
        vest_missing_frames = worker_state.get("vest_missing_frames", 0)
        zone_type = worker_state.get("zone_type", "SAFE")
        zone_frames = worker_state.get("zone_frames", 0)
        is_helmet = worker_state.get("is_helmet_present", False)
        is_vest = worker_state.get("is_vest_present", False)

        # Hazard Zone Compound Critical Rules (RULE 4 & RULE 5)
        if zone_type == "HAZARD" and zone_frames >= 2:
            if not is_helmet and not is_vest:
                violations.append({
                    "rule_id": 45,
                    "type": "HAZARD_NO_PPE",
                    "description": "Hazard Zone Breach without Helmet & Vest",
                    "severity": Severity.CRITICAL,
                    "is_critical": True
                })
            elif not is_helmet:
                violations.append({
                    "rule_id": 4,
                    "type": "HAZARD_NO_HELMET",
                    "description": "Hazard Zone Breach without Helmet",
                    "severity": Severity.CRITICAL,
                    "is_critical": True
                })
            elif not is_vest:
                violations.append({
                    "rule_id": 5,
                    "type": "HAZARD_NO_VEST",
                    "description": "Hazard Zone Breach without Safety Vest",
                    "severity": Severity.CRITICAL,
                    "is_critical": True
                })
            else:
                violations.append({
                    "rule_id": 35,
                    "type": "HAZARD_ZONE_ENTRY",
                    "description": "Authorized Worker in Hazard Zone",
                    "severity": Severity.HIGH,
                    "is_critical": False
                })

        # Restricted Zone Rule (RULE 3)
        elif zone_type == "RESTRICTED" and zone_frames >= min_z_frames:
            violations.append({
                "rule_id": 3,
                "type": "RESTRICTED_ZONE_BREACH",
                "description": "Worker entered Restricted Zone",
                "severity": Severity.HIGH,
                "is_critical": False
            })

        # RULE 1: Missing Helmet for more than MIN_VIOLATION_FRAMES
        if helmet_missing_frames >= min_v_frames:
            # Only add if not already part of hazard critical rule
            if not any("HAZARD_NO_HELMET" in v["type"] or "HAZARD_NO_PPE" in v["type"] for v in violations):
                violations.append({
                    "rule_id": 1,
                    "type": "NO_HELMET",
                    "description": "Missing Safety Helmet Violation",
                    "severity": Severity.MEDIUM,
                    "is_critical": False
                })

        # RULE 2: Missing Vest for more than MIN_VIOLATION_FRAMES
        if vest_missing_frames >= min_v_frames:
            if not any("HAZARD_NO_VEST" in v["type"] or "HAZARD_NO_PPE" in v["type"] for v in violations):
                violations.append({
                    "rule_id": 2,
                    "type": "NO_VEST",
                    "description": "Missing Safety Vest Violation",
                    "severity": Severity.MEDIUM,
                    "is_critical": False
                })

        return violations
