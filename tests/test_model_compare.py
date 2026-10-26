"""
Unit Tests: Model Comparison Engine & Plain Annotation (SH17 Paper Fig. 1, 5, 6)
"""

import os
import unittest
import numpy as np
import cv2

import config
from cv.annotator import plain_annotate, get_class_color, FrameAnnotator
from cv.model_compare import (
    MODEL_REGISTRY,
    MODEL_ALIASES,
    MODEL_METADATA,
    resolve_model_name,
    build_comparison_grid
)


class TestModelCompare(unittest.TestCase):
    def setUp(self):
        self.dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        self.detections = [
            {
                "bbox": [50, 60, 200, 400],
                "confidence": 0.88,
                "class_name": "person",
                "category": "person"
            },
            {
                "bbox": [80, 70, 160, 140],
                "confidence": 0.92,
                "class_name": "helmet",
                "category": "helmet"
            },
            {
                "bbox": [75, 140, 185, 290],
                "confidence": 0.85,
                "class_name": "vest",
                "category": "vest"
            }
        ]

    def test_alias_resolution(self):
        """Verifies shorthand aliases resolve to canonical model names."""
        self.assertEqual(resolve_model_name("v8n"), "YOLOv8-n")
        self.assertEqual(resolve_model_name("yolov8m"), "YOLOv8-m")
        self.assertEqual(resolve_model_name("v8x"), "YOLOv8-x")
        self.assertEqual(resolve_model_name("v9e"), "YOLOv9-e")
        self.assertEqual(resolve_model_name("YOLOv8-n"), "YOLOv8-n")

    def test_class_color_consistency(self):
        """Verifies consistent fixed color mapping per class."""
        c1 = get_class_color("helmet")
        c2 = get_class_color("helmet")
        self.assertEqual(c1, c2)
        # Verify normalization of hyphens/underscores
        c3 = get_class_color("no_helmet")
        c4 = get_class_color("no-helmet")
        self.assertEqual(c3, c4)

    def test_plain_annotate(self):
        """Verifies plain_annotate draws boxes and returns ndarray without HUD."""
        out = plain_annotate(self.dummy_frame, self.detections)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.dummy_frame.shape)
        # Check that canvas pixels were modified
        self.assertFalse(np.array_equal(out, self.dummy_frame))

    def test_frame_annotator_plain_style(self):
        """Verifies FrameAnnotator respects style='plain' flag."""
        annotator = FrameAnnotator()
        workers = [
            {
                "worker_id": 1,
                "label": "Worker #1",
                "bbox": [50, 60, 200, 400],
                "confidence": 0.88,
                "helmet": True,
                "vest": True,
                "violations": []
            }
        ]
        plain_out = annotator.annotate(self.dummy_frame, workers, style="plain")
        full_out = annotator.annotate(self.dummy_frame, workers, style="full")
        self.assertIsInstance(plain_out, np.ndarray)
        self.assertIsInstance(full_out, np.ndarray)
        # Plain style output should differ from full HUD overlay
        self.assertFalse(np.array_equal(plain_out, full_out))

    def test_build_comparison_grid(self):
        """Verifies build_comparison_grid renders valid image with headers and row indices."""
        test_dir = os.path.join(config.OUTPUT_FOLDER, "test_comparison")
        os.makedirs(test_dir, exist_ok=True)

        # Mock results structure
        ann1 = self.dummy_frame.copy()
        ann2 = self.dummy_frame.copy()

        mock_results = {
            "model_names": ["YOLOv8-n", "YOLOv8-m"],
            "skipped_models": [],
            "results_by_image": [
                {
                    "image_index": 1,
                    "image_path": "mock1.jpg",
                    "image_name": "mock1.jpg",
                    "model_outputs": {
                        "YOLOv8-n": {
                            "annotated_frame": ann1,
                            "detection_count": 3,
                            "inference_ms": 15.2,
                            "is_fallback": False
                        },
                        "YOLOv8-m": {
                            "annotated_frame": ann2,
                            "detection_count": 4,
                            "inference_ms": 28.5,
                            "is_fallback": False
                        }
                    }
                }
            ],
            "model_stats": {
                "YOLOv8-n": {"params_m": 3.2, "avg_inference_ms": 15.2, "avg_detections": 3.0, "total_detections": 3},
                "YOLOv8-m": {"params_m": 25.9, "avg_inference_ms": 28.5, "avg_detections": 4.0, "total_detections": 4}
            }
        }

        grid_path, tile_paths = build_comparison_grid(
            comparison_results=mock_results,
            tile_width=240,
            output_dir=test_dir
        )

        self.assertTrue(os.path.exists(grid_path))
        grid_img = cv2.imread(grid_path)
        self.assertIsNotNone(grid_img)
        self.assertGreater(grid_img.shape[0], 100)
        self.assertGreater(grid_img.shape[1], 100)


if __name__ == "__main__":
    unittest.main()
