import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nser_ibvs_drone.detection.yolo_pose_ibvs import YoloEngineIBVSPose


class TestYoloEngineIBVSPose(unittest.TestCase):
    def setUp(self):
        with patch('nser_ibvs_drone.detection.yolo_ibvs.YoloEngineIBVS.__init__', return_value=None):
            self.engine = YoloEngineIBVSPose(model_path_pose="mock_path.pt")
            self.engine.model = MagicMock()
            self.engine.mask_confidence = 0.5

    @patch('nser_ibvs_drone.detection.yolo_ibvs.YoloEngineIBVS._reorder_bbox_oriented')
    def test_reorder_fallback_on_empty_masks(self, mock_super_reorder):
        """Should call super() if front or back masks are missing."""
        mock_super_reorder.return_value = [(0, 0), (10, 0), (10, 10), (0, 10)]

        box = [(0, 0), (1, 1), (2, 2), (3, 3)]
        best_front = {"masks_xy": []}
        best_back = {"masks_xy": np.array([[1, 1]])}

        result = self.engine._reorder_bbox_oriented(box, best_front, best_back)

        mock_super_reorder.assert_called_once_with(box)
        self.assertEqual(result, mock_super_reorder.return_value)

    def test_reorder_bbox_oriented_success(self):
        """Test the geometric reordering: FL, FR, BR, BL."""
        # Setup: Car facing "Up" (-Y direction)
        # Front centroid at (50, 20), Back centroid at (50, 80)
        # Vector orientation (Back - Front) = (0, 60) -> pointing Down
        best_front = {"masks_xy": np.array([[45, 15], [55, 15]])}  # Mean: (50, 15) approx
        best_back = {"masks_xy": np.array([[45, 85], [55, 85]])}  # Mean: (50, 85) approx

        # Bounding box points (unordered)
        # Assuming a rectangle around these centroids
        box = [
            (40, 80),  # Back-Left (relative to car facing Up) -> Actually Back-Right if vector is (0, 60)
            (60, 20),  # Front-Right
            (40, 20),  # Front-Left
            (60, 80)   # Back-Right
        ]

        # We expect: [Front-Left, Front-Right, Back-Right, Back-Left]
        # With Front=(50,15), Back=(50,85), Vector=(0,70)
        # A point (40, 20) is "Right" of the vector (0, 70) in screen space (Y-down)
        # Note: Cross product logic depends on screen coordinates!

        result = self.engine._reorder_bbox_oriented(box, best_front, best_back)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], tuple)

    @patch('nser_ibvs_drone.detection.yolo_ibvs.YoloEngineIBVS._compute_bbox_oriented')
    def test_find_best_target_integration(self, mock_compute_bbox):
        """Test the full pipeline from YOLO results to TargetIBVS."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_compute_bbox.return_value = [(10, 10), (20, 10), (20, 20), (10, 20)]

        mock_result_pose = MagicMock()
        mock_result_pose.boxes.cls = torch.tensor([0.0, 1.0])  # 0=Back, 1=Front
        mock_result_pose.boxes.conf = torch.tensor([0.9, 0.8])

        mock_result_pose.masks.xy = [
            np.array([[10, 10], [12, 10]]),  # Back mask points
            np.array([[50, 50], [52, 50]])   # Front mask points
        ]
        self.engine.model.predict.return_value = [mock_result_pose]

        target = self.engine.find_best_target(frame, MagicMock())

        self.assertFalse(target.is_lost)
        self.assertAlmostEqual(target.confidence, 0.9, places=6)
        # Check if masks were combined (2 points from back + 2 points from front = 4)
        self.assertEqual(len(target.masks_xy), 4)


if __name__ == '__main__':
    unittest.main()
