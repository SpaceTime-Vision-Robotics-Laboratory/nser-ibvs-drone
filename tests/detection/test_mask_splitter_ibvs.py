import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nser_ibvs_drone.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS


class TestMaskSplitterEngineIBVS(unittest.TestCase):
    def setUp(self):
        # Prevent actual model loading
        with patch('nser_ibvs_drone.detection.yolo_ibvs.YoloEngineIBVS.__init__', return_value=None), \
                patch('mask_splitter.nn.infer.MaskSplitterInference.__init__', return_value=None):
            self.engine = MaskSplitterEngineIBVS()
            self.engine.splitter_model = MagicMock()
            self.engine._default_target = MagicMock()
            self.engine.contour_2d_dimensions = 2
            self.engine.max_number_of_points = 2
            self.engine.tol_err_norm = 1e-3

    def test_reorder_bbox_oriented_rotation(self):
        """Verify the list rotation logic based on front points."""
        box = [(10, 10), (20, 10), (20, 30), (10, 30)]

        # Scenario: Front points are actually (20, 10) and (20, 30)
        # (The right side of the box)
        best_front = {"masks_xy": np.array([[21, 15], [21, 25]])}  # Close to right edge
        best_back = {"masks_xy": np.array([[9, 15], [9, 25]])}  # Close to left edge

        # The logic finds candidates: front_pts = [(20, 10), (20, 30)]
        # It should rotate the box so these are at the front.
        result = self.engine._reorder_bbox_oriented(box, best_front, best_back)

        # Check if the result starts with one of the front candidates
        self.assertIn(result[0], [(20, 10), (20, 30)])
        self.assertEqual(len(result), 4)

    @patch('nser_ibvs_drone.detection.yolo_ibvs.YoloEngineIBVS._compute_bbox_oriented')
    def test_find_best_target_full_flow(self, mock_compute_bbox):
        """Test the logic from initial mask to split masks to target."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # 1. Mock base segmentation return (image, mask, masks_xy)
        dummy_mask = np.zeros((360, 640), dtype=np.uint8)
        dummy_mask[100:200, 100:200] = 255
        self.engine.segment_image = MagicMock(return_value=(frame, dummy_mask, np.array([[100, 100], [200, 200]])))

        # 2. Mock Splitter output (front_mask, back_mask)
        # Create small squares representing split masks
        f_mask = np.zeros((360, 640), dtype=np.uint8)
        f_mask[100:110, 100:110] = 255
        b_mask = np.zeros((360, 640), dtype=np.uint8)
        b_mask[190:200, 190:200] = 255
        self.engine.splitter_model.infer.return_value = (f_mask, b_mask)

        # 3. Mock oriented bbox return
        mock_compute_bbox.return_value = [(100, 100), (200, 100), (200, 200), (100, 200)]

        target = self.engine.find_best_target(frame, None)

        self.assertFalse(target.is_lost)
        self.assertEqual(target.confidence, 0.8)
        self.assertTrue(len(target.masks_xy) > 0)
        self.engine.splitter_model.infer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
