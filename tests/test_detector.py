"""
Tests for YOLOv8 Construction Safety Object Detector.
"""
import os
import numpy as np
import pytest
import config
from cv.detector import ConstructionDetector

def test_detector_initialization():
    detector = ConstructionDetector(conf=0.35, iou=0.45)
    assert detector is not None
    assert detector.conf == 0.35
    assert detector.iou == 0.45
    assert detector.device is not None
    assert len(detector.classes) > 0

def test_detector_category_mapping():
    detector = ConstructionDetector()
    assert hasattr(detector, "category_map")
    assert isinstance(detector.category_map, dict)
    assert len(detector.person_cids) > 0

def test_detector_inference_on_synthetic_frame():
    detector = ConstructionDetector(conf=0.20)
    # 640x480 RGB image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a colored rectangle
    dummy_frame[100:300, 200:400] = [120, 180, 220]
    
    results = detector.detect(dummy_frame)
    assert isinstance(results, list)
    for det in results:
        assert "bbox" in det
        assert len(det["bbox"]) == 4
        assert "confidence" in det
        assert 0.0 <= det["confidence"] <= 1.0
        assert "category" in det
        assert "class_name" in det
