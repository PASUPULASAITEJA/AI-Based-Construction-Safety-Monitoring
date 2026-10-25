"""
Spatial Restricted-Zone Geometry Module
Evaluates worker presence inside 2D polygon boundaries strictly at the worker's bottom-center
foot ground-contact coordinate (P_feet) to eliminate camera perspective projection distortion.
"""
from typing import List, Dict, Any, Tuple
import json
import cv2
import numpy as np

def compute_foot_anchor(bbox: List[int]) -> Tuple[int, int]:
    """
    Computes bottom-center foot coordinate:
    P_feet = (int((x1 + x2) / 2), int(y2))
    """
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def evaluate_spatial_containment(worker_bbox: List[int], zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tests worker ground-anchor point P_feet against configured 2D polygons.
    
    Args:
        worker_bbox: [x1, y1, x2, y2]
        zones: List of zone dictionaries with 'id', 'name', 'severity', and 'coordinates'
        
    Returns:
        List of breached zone dicts containing breached zone metadata.
    """
    P_feet = compute_foot_anchor(worker_bbox)
    breached_zones: List[Dict[str, Any]] = []

    for z in zones:
        coords = z.get("coordinates")
        if not coords:
            continue

        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                continue

        if len(coords) < 3:
            continue

        # Format polygon vertices for OpenCV pointPolygonTest
        polygon_np = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

        # Test containment: >= 0 indicates inside or on boundary
        is_inside = cv2.pointPolygonTest(polygon_np, P_feet, False) >= 0

        if is_inside:
            breached_zones.append({
                "zone_id": z.get("id"),
                "zone_name": z.get("name", "Restricted Area"),
                "severity": z.get("severity", "DANGER"),
                "foot_point": P_feet
            })

    return breached_zones
