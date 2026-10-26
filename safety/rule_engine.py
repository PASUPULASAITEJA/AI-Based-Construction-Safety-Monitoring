"""
AI-Based Construction Safety Monitoring — Per-Zone PPE Rule Engine
=================================================================
Evaluates worker PPE compliance per-zone and generates actionable violation records:
  (worker_ppe_status, zone_id) -> list of violations
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("RuleEngine")

DEFAULT_ZONE_RULES: Dict[str, Any] = {
    "version": "1.0.0",
    "zones": [
        {
            "zone_id": "general_site",
            "zone_name": "General Construction Site",
            "risk_level": "MEDIUM",
            "description": "Standard ground site safety boundary",
            "mandatory_ppe": ["helmet", "vest"]
        },
        {
            "zone_id": "chemical_zone",
            "zone_name": "Chemical & Hazardous Materials",
            "risk_level": "CRITICAL",
            "description": "Chemical storage and pressurized solvent wash area",
            "mandatory_ppe": ["helmet", "vest", "goggles", "gloves"]
        },
        {
            "zone_id": "heights_zone",
            "zone_name": "Scaffolding & Elevated Work",
            "risk_level": "CRITICAL",
            "description": "Areas exceeding 2 meters elevation requiring fall arrest",
            "mandatory_ppe": ["helmet", "vest", "harness"]
        },
        {
            "zone_id": "machinery_zone",
            "zone_name": "Heavy Machinery & Grinding",
            "risk_level": "HIGH",
            "description": "Operating heavy cranes, earth movers, and high-decibel tools",
            "mandatory_ppe": ["helmet", "vest", "earmuff", "boots"]
        }
    ]
}


class RuleEngine:
    """
    Site-configurable rule evaluation engine.
    Matches worker PPE status against per-zone policies.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.rules: Dict[str, Dict[str, Any]] = {}
        if config_path and os.path.isfile(config_path):
            self.load_from_file(config_path)
        else:
            self.load_from_dict(DEFAULT_ZONE_RULES)

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.rules = {z["zone_id"]: z for z in data.get("zones", [])}
        logger.info(f"Loaded {len(self.rules)} zone rules into RuleEngine.")

    def load_from_file(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_dict(data)

    def save_to_file(self, file_path: str) -> None:
        payload = {
            "version": "1.0.0",
            "zones": list(self.rules.values())
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_zone_rule(self, zone_id: str) -> Optional[Dict[str, Any]]:
        return self.rules.get(zone_id)

    def update_zone_rule(self, zone_id: str, zone_data: Dict[str, Any]) -> None:
        self.rules[zone_id] = zone_data

    def evaluate_worker(
        self,
        worker_ppe_status: Dict[str, Any],
        zone_id: str
    ) -> List[Dict[str, Any]]:
        """
        Takes (worker_ppe_status, zone_id) -> list of violations.

        worker_ppe_status dict schema:
          {
            "worker_id": "W-001",
            "helmet": True/False,
            "goggles": True/False,
            "earmuff": True/False,
            "vest": True/False,
            "harness": True/False,
            "safety_pants": True/False,
            "gloves": True/False,
            "boots": True/False,
            ...
          }

        Returns:
            List of violation dicts:
            [
                {
                    "worker_id": "W-001",
                    "zone_id": "heights_zone",
                    "zone_name": "Scaffolding & Elevated Work",
                    "missing_item": "harness",
                    "severity": "CRITICAL",
                    "description": "Worker W-001 missing mandatory harness in Scaffolding & Elevated Work"
                }, ...
            ]
        """
        zone = self.rules.get(zone_id)
        if not zone:
            zone = self.rules.get("general_site", {
                "zone_id": zone_id,
                "zone_name": f"Zone {zone_id}",
                "risk_level": "MEDIUM",
                "mandatory_ppe": ["helmet", "vest"]
            })

        mandatory = zone.get("mandatory_ppe", [])
        risk = zone.get("risk_level", "MEDIUM")
        worker_id = worker_ppe_status.get("worker_id", "Unknown")

        violations = []
        for ppe_item in mandatory:
            is_present = worker_ppe_status.get(ppe_item, False)
            # If item is missing (False or "missing")
            if is_present is False or is_present == "missing":
                clean_name = ppe_item.replace("_", " ")
                violations.append({
                    "worker_id": worker_id,
                    "zone_id": zone["zone_id"],
                    "zone_name": zone["zone_name"],
                    "missing_item": ppe_item,
                    "severity": risk,
                    "description": f"Worker {worker_id} missing mandatory {clean_name} in {zone['zone_name']}"
                })

        return violations


if __name__ == "__main__":
    engine = RuleEngine()
    dummy_worker = {
        "worker_id": "W-002",
        "helmet": True,
        "vest": True,
        "goggles": False,
        "gloves": False,
        "earmuff": False,
        "boots": True,
        "harness": False,
        "safety_pants": True
    }

    # Evaluate in chemical zone (requires helmet, vest, goggles, gloves)
    violations = engine.evaluate_worker(dummy_worker, "chemical_zone")
    print(f"Chemical Zone Violations for {dummy_worker['worker_id']}:")
    for v in violations:
        print(f" - [{v['severity']}] {v['description']}")

    # Evaluate in heights zone (requires helmet, vest, harness)
    violations_heights = engine.evaluate_worker(dummy_worker, "heights_zone")
    print(f"\nHeights Zone Violations for {dummy_worker['worker_id']}:")
    for v in violations_heights:
        print(f" - [{v['severity']}] {v['description']}")
