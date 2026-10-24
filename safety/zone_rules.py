"""
Dynamic Per-Zone PPE Rule Engine with Temporal Hysteresis
Evaluates worker compliance against custom zone requirements.
"""

from typing import Dict, List, Any
import json
from pathlib import Path

DEFAULT_RULES = {
    "version": "2.0",
    "zones": [
        {
            "zone_id": "GENERAL",
            "name": "General Construction Site",
            "zone_type": "GENERAL",
            "required_ppe": ["helmet", "vest"],
            "severity_override": "MEDIUM"
        },
        {
            "zone_id": "HEIGHTS",
            "name": "Scaffolding & Heights Zone",
            "zone_type": "HEIGHTS",
            "required_ppe": ["helmet", "vest", "harness"],
            "severity_override": "CRITICAL"
        },
        {
            "zone_id": "CHEMICAL",
            "name": "Chemical & Hazardous Material Zone",
            "zone_type": "CHEMICAL",
            "required_ppe": ["helmet", "vest", "goggles", "gloves"],
            "severity_override": "HIGH"
        },
        {
            "zone_id": "MACHINERY",
            "name": "Heavy Machinery Operation",
            "zone_type": "MACHINERY",
            "required_ppe": ["helmet", "vest", "earmuff", "boots"],
            "severity_override": "CRITICAL"
        },
        {
            "zone_id": "RESTRICTED",
            "name": "No Worker Entry Zone",
            "zone_type": "RESTRICTED",
            "required_ppe": [],
            "severity_override": "CRITICAL"
        },
        {
            "zone_id": "SAFE",
            "name": "Break Area / Office",
            "zone_type": "SAFE",
            "required_ppe": [],
            "severity_override": "LOW"
        }
    ]
}

class RuleEngine:
    def __init__(self, config_path: str = None):
        self.rules_by_zone = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = DEFAULT_RULES
        
        for z in cfg.get("zones", []):
            self.rules_by_zone[z["zone_id"]] = z
            self.rules_by_zone[z["zone_type"]] = z

    def evaluate_worker(self, worker_ppe: Dict[str, Any], zone_identifier: str = "GENERAL") -> List[Dict[str, Any]]:
        """
        worker_ppe: Output dict from PPEAssociator
        zone_identifier: Active zone ID or zone type
        
        Returns list of breach dictionaries.
        """
        zone_rule = self.rules_by_zone.get(zone_identifier, self.rules_by_zone.get("GENERAL"))
        if not zone_rule:
            return []

        # If zone is RESTRICTED, any worker detected inside triggers breach
        if zone_rule["zone_type"] == "RESTRICTED":
            return [{
                "violation_type": "RESTRICTED_ZONE_BREACH",
                "item": "perimeter",
                "severity": "CRITICAL",
                "description": f"Worker {worker_ppe['worker_id']} entered restricted perimeter: {zone_rule['name']}"
            }]

        # If SAFE zone, no PPE is required
        if zone_rule["zone_type"] == "SAFE":
            return []

        violations = []
        required_ppe = zone_rule.get("required_ppe", [])
        base_severity = zone_rule.get("severity_override", "HIGH")

        for item in required_ppe:
            is_present = worker_ppe.get(item, False)
            if not is_present:
                # Harness or helmet missing in heights is always CRITICAL
                sev = "CRITICAL" if item in ["harness", "helmet"] and zone_rule["zone_type"] == "HEIGHTS" else base_severity
                violations.append({
                    "violation_type": f"NO_{item.upper()}",
                    "item": item,
                    "severity": sev,
                    "description": f"Worker {worker_ppe['worker_id']} missing mandatory {item} in {zone_rule['name']}"
                })

        return violations
