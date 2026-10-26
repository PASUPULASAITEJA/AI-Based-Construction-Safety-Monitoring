import os
import sys
import unittest
import numpy as np
import cv2

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from cv.detector import ConstructionDetector
from cv.tracker import WorkerTracker
from cv.ppe_association import PPEAssociator, compute_box_overlap
from cv.zone_detector import ZoneDetector
from safety.safety_rules import SafetyRuleEngine, Severity
from safety.violation_manager import ViolationManager
from database.db import init_db
from database.models import IncidentModel, ZoneModel, SettingsModel
from app import app

class TestConstructionSafetySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_config_and_dataset(self):
        """Test configuration loading and class definitions."""
        self.assertTrue(os.path.exists(config.DATA_YAML_PATH), "data.yaml must exist")
        detector = ConstructionDetector()
        self.assertIn("person", detector.canonical_map)
        self.assertIn("helmet", detector.canonical_map)
        self.assertIn("vest", detector.canonical_map)

    def test_02_tracker_id_persistence(self):
        """Test WorkerTracker tracks person across frames and retains ID."""
        tracker = WorkerTracker(iou_threshold=0.3)
        
        # Frame 1: Worker at [100, 100, 200, 300]
        det_f1 = [{"bbox": [100, 100, 200, 300], "confidence": 0.9}]
        w_f1 = tracker.update(det_f1)
        self.assertEqual(len(w_f1), 1)
        wid_1 = w_f1[0]["worker_id"]

        # Frame 2: Worker slightly moved [105, 102, 205, 302]
        det_f2 = [{"bbox": [105, 102, 205, 302], "confidence": 0.88}]
        w_f2 = tracker.update(det_f2)
        self.assertEqual(len(w_f2), 1)
        self.assertEqual(w_f2[0]["worker_id"], wid_1, "Tracking ID must persist across consecutive frames")

    def test_03_ppe_spatial_association(self):
        """Test spatial PPE association for helmet (head) and vest (torso)."""
        associator = PPEAssociator()
        worker = [{"worker_id": 1, "label": "Worker #1", "bbox": [100, 100, 200, 400], "confidence": 0.9}]

        # Head region is roughly top 35%: y: 100 to 205
        # Torso region is roughly 15%-75%: y: 145 to 325
        detections = [
            {"bbox": [110, 95, 190, 160], "confidence": 0.95, "category": "helmet"},
            {"bbox": [105, 160, 195, 310], "confidence": 0.92, "category": "vest"}
        ]

        enriched = associator.associate(worker, detections)
        self.assertEqual(len(enriched), 1)
        w = enriched[0]
        self.assertTrue(w["helmet"], "Helmet must be associated to worker head")
        self.assertTrue(w["vest"], "Vest must be associated to worker torso")

    def test_04_missing_ppe_detection(self):
        """Test detection of missing helmet / vest without hallucinated items."""
        associator = PPEAssociator()
        worker = [{"worker_id": 2, "label": "Worker #2", "bbox": [100, 100, 200, 400], "confidence": 0.9}]

        # No PPE detected nearby
        detections = []
        enriched = associator.associate(worker, detections)
        self.assertFalse(enriched[0]["helmet"], "Must report missing helmet when none detected")
        self.assertFalse(enriched[0]["vest"], "Must report missing vest when none detected")

    def test_05_zone_detector(self):
        """Test restricted and hazard zone bottom-center anchor point containment."""
        zone_detector = ZoneDetector()
        zone_detector.set_zones([
            {
                "id": 1,
                "name": "Excavation Pit",
                "type": "HAZARD",
                "coordinates": [[300, 300], [500, 300], [500, 500], [300, 500]]
            },
            {
                "id": 2,
                "name": "Crane Swing Radius",
                "type": "RESTRICTED",
                "coordinates": [[600, 100], [800, 100], [800, 400], [600, 400]]
            }
        ])

        # Worker 1 feet at (400, 450) -> inside Hazard zone
        w1_box = [350, 250, 450, 450]
        z_type, z_obj = zone_detector.check_worker_zone(w1_box)
        self.assertEqual(z_type, "HAZARD")
        self.assertEqual(z_obj["name"], "Excavation Pit")

        # Worker 2 feet at (700, 350) -> inside Restricted zone
        w2_box = [650, 150, 750, 350]
        z_type, z_obj = zone_detector.check_worker_zone(w2_box)
        self.assertEqual(z_type, "RESTRICTED")

        # Worker 3 feet at (100, 200) -> SAFE
        w3_box = [50, 50, 150, 200]
        z_type, _ = zone_detector.check_worker_zone(w3_box)
        self.assertEqual(z_type, "SAFE")

    def test_06_safety_rule_engine(self):
        """Test the 5 safety rules and compound severity triggers."""
        engine = SafetyRuleEngine()

        # Rule 1: Missing Helmet > 10 frames
        state_r1 = {
            "worker_id": 1,
            "helmet_missing_frames": 10,
            "vest_missing_frames": 0,
            "zone_type": "SAFE",
            "zone_frames": 0,
            "is_helmet_present": False,
            "is_vest_present": True
        }
        v1 = engine.evaluate_rules(state_r1)
        self.assertTrue(any(v["type"] == "NO_HELMET" for v in v1))

        # Rule 4 & 5: Hazard Zone Breach without PPE -> CRITICAL
        state_crit = {
            "worker_id": 2,
            "helmet_missing_frames": 5,
            "vest_missing_frames": 5,
            "zone_type": "HAZARD",
            "zone_frames": 3,
            "is_helmet_present": False,
            "is_vest_present": False
        }
        v_crit = engine.evaluate_rules(state_crit)
        self.assertTrue(any(v["severity"] == Severity.CRITICAL for v in v_crit))

    def test_07_temporal_verification_reset(self):
        """Test temporal verification does NOT alert on single-frame dropout, and resets when PPE returns."""
        vm = ViolationManager(min_violation_frames=5)
        worker = {"worker_id": 99, "label": "Worker #99", "bbox": [100, 100, 200, 300], "confidence": 0.9, "helmet": False, "vest": True, "zone_type": "SAFE"}

        # Frames 1 to 4: Missing helmet -> No confirmed incident yet
        for f in range(4):
            _, incs = vm.process_frame([worker])
            self.assertEqual(len(incs), 0, f"Frame {f+1} should not trigger confirmed alert before threshold")

        # Frame 5: PPE is detected again -> Counter resets
        worker["helmet"] = True
        _, incs = vm.process_frame([worker])
        self.assertEqual(len(incs), 0)
        self.assertEqual(vm.worker_states[99].helmet_missing_frames, 0, "Counter must reset upon PPE detection")

    def test_08_database_crud(self):
        """Test incident logging and retrieval in SQLite."""
        inc_id = IncidentModel.create(
            incident_code="INC-TEST01",
            worker_id=1,
            worker_label="Worker #1",
            violation_type="NO_HELMET",
            description="Test violation",
            severity="MEDIUM",
            confidence=0.92,
            source="Unit Test"
        )
        self.assertIsNotNone(inc_id)

        incidents = IncidentModel.get_all(limit=10)
        self.assertTrue(any(i["incident_code"] == "INC-TEST01" for i in incidents))

    def test_09_flask_routes(self):
        """Test all Flask web application endpoints."""
        client = app.test_client()

        # Page routes
        for endpoint in ["/", "/monitor", "/upload", "/incidents", "/analytics", "/settings"]:
            res = client.get(endpoint)
            self.assertEqual(res.status_code, 200, f"Endpoint {endpoint} must return HTTP 200")

        # Status API
        res_status = client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertIn("worker_count", res_status.get_json())

        # Zones API
        res_zones = client.get("/api/zones")
        self.assertEqual(res_zones.status_code, 200)

if __name__ == "__main__":
    unittest.main()
