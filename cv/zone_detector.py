"""
CV Module: Context-Aware Zone Detector
Performs point-in-polygon testing using the bottom-center anchor point of worker bounding boxes.

Zone coordinates are stored normalized to [0, 1] relative to the frame, so the same zone
applies to any resolution (webcam, simulator images, uploaded media). Zones saved by older
versions in 640x480 pixel space are detected and converted automatically.
"""
import cv2
import numpy as np

# Frame size used by legacy pixel-space zones (and the default when no frame shape is given)
LEGACY_FRAME_SIZE = (640, 480)  # (width, height)

def normalize_coordinates(coords):
    """Returns coordinates normalized to [0, 1]. Values above 1 are treated as legacy 640x480 pixels."""
    if any(abs(float(v)) > 1.0 for pt in coords for v in pt):
        lw, lh = LEGACY_FRAME_SIZE
        return [[float(x) / lw, float(y) / lh] for x, y in coords]
    return [[float(x), float(y)] for x, y in coords]

class ZoneDetector:
    def __init__(self):
        self.zones = [] # list of dicts
        self._pixel_cache = {}

    def set_zones(self, zone_list):
        """
        zone_list: list of dicts:
        [
            {
                "id": 1,
                "name": "Heavy Machinery Area",
                "type": "MACHINERY",
                "coordinates": [[x1, y1], [x2, y2], [x3, y3], ...]
            }, ...
        ]
        """
        self.zones = []
        self._pixel_cache = {}
        for z in zone_list:
            coords = z.get("coordinates", [])
            if len(coords) >= 3:
                self.zones.append({
                    "id": z.get("id"),
                    "name": z.get("name", f"Zone {z.get('id')}"),
                    "type": str(z.get("zone_type") or z.get("type", "SAFE")).upper(),
                    "norm_coords": normalize_coordinates(coords)
                })

    def zones_for_frame(self, frame_shape=None):
        """Returns zones with "points" scaled to the given frame shape (h, w, ...), for testing and drawing."""
        if frame_shape is None:
            w, h = LEGACY_FRAME_SIZE
        else:
            h, w = frame_shape[:2]

        if (w, h) not in self._pixel_cache:
            scaled = []
            for z in self.zones:
                pts = np.array([[round(x * w), round(y * h)] for x, y in z["norm_coords"]], dtype=np.int32)
                scaled.append({**z, "points": pts.reshape((-1, 1, 2))})
            self._pixel_cache[(w, h)] = scaled
        return self._pixel_cache[(w, h)]

    def check_worker_zone(self, worker_box, frame_shape=None):
        """
        Tests if worker's bottom-center point ((x1+x2)/2, y2) falls inside any defined zone.
        Returns the specific zone type defined by the safety officer (e.g., WELDING, HEIGHTS).
        If multiple overlap, returns the first match.
        """
        x1, y1, x2, y2 = worker_box
        # Bottom-center anchor point (feet contact point)
        anchor_point = (float((x1 + x2) / 2.0), float(y2))

        for z in self.zones_for_frame(frame_shape):
            # pointPolygonTest returns >= 0 if inside or on edge
            dist = cv2.pointPolygonTest(z["points"], anchor_point, False)
            if dist >= 0:
                # Inside this zone
                return z["type"], z

        # If not inside any drawn polygon, default to SAFE (or GROUND if preferred)
        return "SAFE", None

    def evaluate_workers(self, workers, frame_shape=None):
        """
        Enriches workers with their current zone status.
        """
        for w in workers:
            z_type, z_obj = self.check_worker_zone(w["bbox"], frame_shape)
            w["zone_type"] = z_type
            w["zone_name"] = z_obj["name"] if z_obj else "SAFE"
            w["zone_id"] = z_obj["id"] if z_obj else None
        return workers
