"""
CV Module: Restricted & Hazard Zone Detector
Performs point-in-polygon testing using the bottom-center anchor point of worker bounding boxes.
"""
import cv2
import numpy as np

class ZoneDetector:
    def __init__(self):
        self.zones = [] # list of dicts: {"id": str/int, "name": str, "type": "SAFE"|"RESTRICTED"|"HAZARD", "points": np.ndarray}

    def set_zones(self, zone_list):
        """
        zone_list: list of dicts:
        [
            {
                "id": 1,
                "name": "Heavy Machinery Area",
                "type": "RESTRICTED" | "HAZARD" | "SAFE",
                "coordinates": [[x1, y1], [x2, y2], [x3, y3], ...]
            }, ...
        ]
        """
        self.zones = []
        for z in zone_list:
            coords = z.get("coordinates", [])
            if len(coords) >= 3:
                pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                self.zones.append({
                    "id": z.get("id"),
                    "name": z.get("name", f"Zone {z.get('id')}"),
                    "type": str(z.get("type", "RESTRICTED")).upper(),
                    "points": pts,
                    "raw_coords": coords
                })

    def check_worker_zone(self, worker_box):
        """
        Tests if worker's bottom-center point ((x1+x2)/2, y2) falls inside any defined zone.
        Returns the highest severity zone breached:
        ("SAFE" | "RESTRICTED" | "HAZARD", zone_dict_or_None)
        """
        x1, y1, x2, y2 = worker_box
        # Bottom-center anchor point (feet contact point)
        anchor_point = (float((x1 + x2) / 2.0), float(y2))

        highest_zone_type = "SAFE"
        matched_zone = None

        for z in self.zones:
            # pointPolygonTest returns >= 0 if inside or on edge
            dist = cv2.pointPolygonTest(z["points"], anchor_point, False)
            if dist >= 0:
                z_type = z["type"]
                if z_type == "HAZARD":
                    return "HAZARD", z
                elif z_type == "RESTRICTED" and highest_zone_type != "HAZARD":
                    highest_zone_type = "RESTRICTED"
                    matched_zone = z

        return highest_zone_type, matched_zone

    def evaluate_workers(self, workers):
        """
        Enriches workers with their current zone status.
        """
        for w in workers:
            z_type, z_obj = self.check_worker_zone(w["bbox"])
            w["zone_type"] = z_type
            w["zone_name"] = z_obj["name"] if z_obj else "SAFE"
            w["zone_id"] = z_obj["id"] if z_obj else None
        return workers
