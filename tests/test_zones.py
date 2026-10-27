"""
Tests for Spatial Safety-Zone Detection and Ray-Casting Point-in-Polygon.
"""
import pytest
from cv.zone_detector import ZoneDetector, normalize_coordinates

def test_coordinate_normalization():
    # Already normalized coords in [0, 1]
    norm = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    res = normalize_coordinates(norm)
    assert res[0] == [0.1, 0.1]

    # Legacy 640x480 pixel coordinates
    pixels = [[64, 48], [576, 48], [576, 432], [64, 432]]
    res_pixels = normalize_coordinates(pixels)
    assert res_pixels[0] == pytest.approx([0.1, 0.1])
    assert res_pixels[2] == pytest.approx([0.9, 0.9])

def test_zone_detector_point_in_polygon():
    detector = ZoneDetector()
    zones = [
        {
            "id": 1,
            "name": "Restricted Crane Area",
            "zone_type": "RESTRICTED",
            "coordinates": [[100, 100], [300, 100], [300, 300], [100, 300]]
        }
    ]
    detector.set_zones(zones)

    # Worker whose feet (bottom-center) fall inside the restricted polygon [200, 200]
    # bbox: [x1, y1, x2, y2] -> bottom-center = ((150+250)/2, 200) = (200, 200)
    inside_worker = [150, 100, 250, 200]
    zone_type, zone_obj = detector.check_worker_zone(inside_worker, frame_shape=(480, 640, 3))
    assert zone_type == "RESTRICTED"
    assert zone_obj["name"] == "Restricted Crane Area"

    # Worker whose feet are outside the zone
    outside_worker = [400, 200, 500, 400]
    zone_type_out, zone_obj_out = detector.check_worker_zone(outside_worker, frame_shape=(480, 640, 3))
    assert zone_type_out == "SAFE"
    assert zone_obj_out is None
