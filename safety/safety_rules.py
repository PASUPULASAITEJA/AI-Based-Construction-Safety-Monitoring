"""
Safety Module: Safety Rule Engine
=================================
Evaluates site safety compliance rules for all 8 PPE categories and zone boundaries.
"""

class Severity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

PPE_CATEGORY_CONFIG = [
    {"key": "helmet", "type": "NO_HELMET", "desc": "Missing Safety Helmet Violation", "severity": Severity.MEDIUM},
    {"key": "vest", "type": "NO_VEST", "desc": "Missing High-Vis Safety Vest Violation", "severity": Severity.MEDIUM},
    {"key": "goggles", "type": "NO_GOGGLES", "desc": "Missing Eye Protection / Goggles", "severity": Severity.MEDIUM},
    {"key": "earmuff", "type": "NO_EARMUFF", "desc": "Missing Hearing Protection / Earmuffs", "severity": Severity.MEDIUM},
    {"key": "harness", "type": "NO_HARNESS", "desc": "Missing Fall-Arrest Safety Harness", "severity": Severity.CRITICAL},
    {"key": "safety_pants", "type": "NO_SAFETY_PANTS", "desc": "Missing Protective Safety Pants", "severity": Severity.LOW},
    {"key": "gloves", "type": "NO_GLOVES", "desc": "Missing Heavy-Duty Work Gloves", "severity": Severity.LOW},
    {"key": "boots", "type": "NO_BOOTS", "desc": "Missing Steel-Toe Safety Boots", "severity": Severity.MEDIUM}
]


class SafetyRuleEngine:
    def __init__(self):
        pass

    def evaluate_rules(self, worker_state):
        """
        Evaluates safety rules on a single worker state across all 8 PPE items and zones.
        """
        violations = []
        min_v_frames = worker_state.get("min_violation_frames", 10)
        min_z_frames = worker_state.get("min_zone_frames", 8)

        zone_type = worker_state.get("zone_type", "SAFE")
        zone_frames = worker_state.get("zone_frames", 0)
        zone_id = worker_state.get("zone_id", "general_site")

        # 1. Critical Hazard & Height Zone Rules
        if zone_type in ["HAZARD", "CRITICAL"] and zone_frames >= 2:
            is_harness = worker_state.get("harness", True)
            is_helmet = worker_state.get("helmet", worker_state.get("is_helmet_present", True))
            is_vest = worker_state.get("vest", worker_state.get("is_vest_present", True))

            if not is_harness and zone_id == "heights_zone":
                violations.append({
                    "rule_id": 99,
                    "type": "HAZARD_NO_HARNESS",
                    "description": "Critical Fall Hazard: Missing Safety Harness in Elevated Zone",
                    "severity": Severity.CRITICAL,
                    "is_critical": True
                })

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

        # 2. Restricted Zone Boundary Breach Rule
        elif zone_type in ["RESTRICTED", "HIGH"] and zone_frames >= min_z_frames:
            violations.append({
                "rule_id": 3,
                "type": "RESTRICTED_ZONE_BREACH",
                "description": "Unauthorized Worker entered Restricted Zone",
                "severity": Severity.HIGH,
                "is_critical": False
            })

        # 3. PPE Category Evaluations (all 8 items)
        missing_ppe_list = worker_state.get("missing_ppe", [])
        ppe_missing_frames = worker_state.get("ppe_missing_frames", {})

        for item in PPE_CATEGORY_CONFIG:
            key = item["key"]
            is_present = worker_state.get(key, worker_state.get(f"is_{key}_present", True))
            frames_absent = ppe_missing_frames.get(key, worker_state.get(f"{key}_missing_frames", 0))

            # Confirm violation if absent for >= min_v_frames or explicitly listed in confirmed missing
            if (not is_present and frames_absent >= min_v_frames) or (key in missing_ppe_list and frames_absent >= 3):
                # Avoid duplicate if already covered by critical hazard rule
                if key == "helmet" and any("HAZARD_NO_HELMET" in v["type"] or "HAZARD_NO_PPE" in v["type"] for v in violations):
                    continue
                if key == "vest" and any("HAZARD_NO_VEST" in v["type"] or "HAZARD_NO_PPE" in v["type"] for v in violations):
                    continue
                if key == "harness" and any("HAZARD_NO_HARNESS" in v["type"] for v in violations):
                    continue

                violations.append({
                    "rule_id": 10 + len(violations),
                    "type": item["type"],
                    "description": item["desc"],
                    "severity": item["severity"],
                    "is_critical": item["severity"] == Severity.CRITICAL
                })

        return violations
