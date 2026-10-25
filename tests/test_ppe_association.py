"""
Tests for Anatomical PPE-to-Worker Association.
"""
import pytest
from cv.ppe_association import PPEAssociator, compute_box_overlap, is_point_inside

def test_box_overlap_calculation():
    inner = [110, 110, 190, 190]
    outer = [100, 100, 200, 200]
    # Inner box completely inside outer box
    assert compute_box_overlap(inner, outer) == pytest.approx(1.0)

    disjoint = [300, 300, 350, 350]
    assert compute_box_overlap(disjoint, outer) == pytest.approx(0.0)

def test_point_inside():
    box = [50, 50, 150, 150]
    assert is_point_inside((100, 100), box) is True
    assert is_point_inside((40, 100), box) is False
    assert is_point_inside((100, 160), box) is False

def test_anatomical_head_and_torso_spans():
    associator = PPEAssociator()
    person_box = [100, 100, 200, 400]  # height = 300, width = 100
    
    head_box = associator.get_head_box(person_box)
    torso_box = associator.get_torso_box(person_box)

    # Head box should be at top of person
    assert head_box[1] <= person_box[1]
    assert head_box[3] < person_box[3]

    # Torso box should be between head and feet
    assert torso_box[1] > person_box[1]
    assert torso_box[3] < person_box[3]

def test_worker_ppe_association_helmet_and_vest():
    associator = PPEAssociator()
    workers = [
        {
            "worker_id": 1,
            "label": "Worker #1",
            "bbox": [100, 100, 200, 400],
            "confidence": 0.90
        }
    ]
    detections = [
        # Person
        {"category": "person", "class_name": "Person", "bbox": [100, 100, 200, 400], "confidence": 0.90},
        # Helmet situated on head
        {"category": "helmet", "class_name": "helmet", "bbox": [120, 90, 180, 150], "confidence": 0.94},
        # Vest situated on torso
        {"category": "vest", "class_name": "vest", "bbox": [110, 160, 190, 280], "confidence": 0.88}
    ]

    enriched = associator.associate(workers, detections)
    assert len(enriched) == 1
    w = enriched[0]
    assert w.get("helmet") is True
    assert w.get("vest") is True
    assert w.get("has_HEAD") is True
    assert w.get("has_VISIBILITY") is True
