import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from auto_follow.detection.targets import TargetIBVS
from auto_follow.detection.yolo_ibvs import YoloEngineIBVS


class TestYoloEngineIBVS(unittest.TestCase):

    def setUp(self):
        """Set up IBVS engine with mocked YOLO weights."""
        with patch('ultralytics.YOLO') as mocked_yolo:
            self.engine = YoloEngineIBVS(model_path="mock_path.pt")
            self.engine.model = MagicMock()

    def test_compute_bbox_oriented(self):
        """Test that compute_bbox_oriented returns 4 corner points for a square mask."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        xy_seg = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)

        bbox = self.engine._compute_bbox_oriented(frame, xy_seg)

        self.assertEqual(len(bbox), 4)
        for point in bbox:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)

    def test_reorder_bbox_oriented(self):
        """Test that the reordering logic maintains a consistent starting point."""
        unordered_box = [(50, 50), (10, 10), (50, 10), (10, 50)]

        reordered = self.engine._reorder_bbox_oriented(unordered_box)

        self.assertEqual(len(reordered), 4)
        self.assertEqual(min(p[1] for p in unordered_box), reordered[0][1])

    def test_find_best_target_ibvs_success(self):
        """Test find_best_target with valid detections and masks."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_results = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.xyxy = torch.tensor([[10, 10, 50, 50]])
        mock_boxes.__len__.return_value = 1
        mock_results.boxes = mock_boxes

        mock_results.masks.xy = [np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)]

        target = self.engine.find_best_target(frame, mock_results)

        self.assertIsInstance(target, TargetIBVS)
        self.assertAlmostEqual(target.confidence, 0.9, places=5)
        self.assertIsNotNone(target.bbox_oriented)
        self.assertEqual(len(target.bbox_oriented), 4)
        self.assertFalse(target.is_lost)

    def test_find_best_target_low_confidence(self):
        """Test that target is marked lost if confidence is below threshold."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_results = MagicMock()
        mock_boxes = MagicMock()

        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.xyxy = torch.tensor([[10, 10, 50, 50]])
        mock_results.boxes.__len__.return_value = 1

        target = self.engine.find_best_target(frame, mock_results)

        self.assertEqual(target.confidence, -1.0)
        self.assertTrue(target.is_lost)

    def test_compute_bbox_no_contours_error(self):
        """Test that ValueError is raised if an empty mask is provided."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        empty_seg = np.array([], dtype=np.int32)

        with self.assertRaises(cv2.error):
            self.engine._compute_bbox_oriented(frame, empty_seg)


if __name__ == '__main__':
    unittest.main()
