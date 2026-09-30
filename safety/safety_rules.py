"""
Safety Module: Safety Rule Engine
Evaluates safety compliance rules and determines violation types and severity levels based on context-aware zone rules.
"""

class Severity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

from .ppe_config import get_ppe_capabilities, is_ppe_supported

# Define mandatory PPE categories for each work zone
ZONE_RULES = {
    "GROUND": ["HEAD", "VISIBILITY", "FOOT", "HAND"],
    "HEIGHTS": ["HEAD", "FALL_PROTECTION", "FOOT", "HAND"],
    "MACHINERY": ["HEAD", "VISIBILITY", "FOOT", "EYE", "HEARING"],
    "WELDING": ["HEAD", "EYE", "HAND", "VISIBILITY"],
    "ELECTRICAL": ["HEAD", "HAND", "FOOT", "EYE"],
    "TRENCHING": ["HEAD", "VISIBILITY", "FOOT", "FALL_PROTECTION"],
    # Fallbacks for existing zones
    "SAFE": ["HEAD", "VISIBILITY"],
    "HAZARD": ["HEAD", "VISIBILITY", "FOOT", "EYE"],
    "RESTRICTED": []
}

class SafetyRuleEngine:
    def __init__(self):
        pass

    def get_required_ppe(self, zone_name):
        zone_upper = str(zone_name).upper()
        if zone_upper in ZONE_RULES:
            return ZONE_RULES[zone_upper]
        return ZONE_RULES["SAFE"]  # Default to basic PPE

    def determine_severity(self, missing_ppe, zone_type):
        """Dynamic severity scoring matrix based on the zone context."""
        zone_type = zone_type.upper()
        
        # CRITICAL conditions
        if zone_type in ["HEIGHTS", "TRENCHING"] and "FALL_PROTECTION" in missing_ppe:
            return Severity.CRITICAL, True
        if zone_type == "HAZARD" and ("HEAD" in missing_ppe or "VISIBILITY" in missing_ppe):
            return Severity.CRITICAL, True
            
        # HIGH conditions
        if zone_type == "WELDING" and "EYE" in missing_ppe:
            return Severity.HIGH, False
        if zone_type == "MACHINERY" and "HEAD" in missing_ppe:
            return Severity.HIGH, False
        if "HEAD" in missing_ppe:
            return Severity.HIGH, False
            
        # MEDIUM conditions
        if zone_type == "ELECTRICAL" and "HAND" in missing_ppe:
            return Severity.MEDIUM, False
        if "VISIBILITY" in missing_ppe:
            return Severity.MEDIUM, False
            
        # LOW conditions
        return Severity.LOW, False

    def evaluate_rules(self, worker_state):
        """
        Evaluates context-aware zone-based safety rules using EWMA smoothed confidence scores.
        Crucially, only evaluates PPE categories that the CURRENT YOLO MODEL actually supports.
        """
        violations = []
        zone_type = worker_state.get("zone_type", "SAFE")
        zone_frames = worker_state.get("zone_frames", 0)
        zone_duration = worker_state.get("zone_duration", 0.0)
        
        # Restricted Zone Breach (Independent of PPE)
        if zone_type == "RESTRICTED" and (zone_frames >= 5 or zone_duration >= 1.0):
            violations.append({
                "rule_id": 3,
                "type": "RESTRICTED_ZONE_BREACH",
                "description": "Worker entered Restricted Zone",
                "severity": Severity.CRITICAL,
                "is_critical": True
            })
            # No PPE checks needed if they shouldn't be there at all
            return violations

        # Get Required PPE for this specific zone
        required_ppe_categories = self.get_required_ppe(zone_type)
        missing_ppe = []
        
        for ppe_category in required_ppe_categories:
            # RULE: NEVER FAKE AI DETECTION.
            # If the current model cannot detect this category, we cannot flag it as missing.
            if not is_ppe_supported(ppe_category):
                continue
                
            # e.g., 'is_HEAD_present', 'is_VISIBILITY_present'
            key = f"is_{ppe_category}_present"
            is_present = worker_state.get(key, False)
            if not is_present:
                missing_ppe.append(ppe_category)
                
        # If PPE is missing, generate the appropriate violation
        if missing_ppe:
            severity, is_critical = self.determine_severity(missing_ppe, zone_type)
            
            # Format the violation string gracefully
            missing_str = ", ".join([p.replace("_", " ").title() for p in missing_ppe])
            
            violations.append({
                "rule_id": 100,
                "type": f"{zone_type}_NO_{missing_ppe[0].upper()}",
                "description": f"Missing mandatory PPE in {zone_type} Zone: {missing_str}",
                "severity": severity,
                "is_critical": is_critical
            })
            
        return violations
