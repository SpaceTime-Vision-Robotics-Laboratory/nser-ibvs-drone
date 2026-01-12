import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nser_ibvs_drone.detection.yolo_engine import YoloEngine


class TestYoloEngine(unittest.TestCase):

    def setUp(self):
        """Set up the engine with a mocked YOLO model."""
        with patch('ultralytics.YOLO') as mocked_yolo:
            self.engine = YoloEngine(model_path="mock_path.pt")
            self.engine.model = MagicMock()
            self.engine.confidence_threshold = 0.5

    def test_find_best_target_with_detections(self):
        """Test that the best target is selected based on highest confidence."""
        mock_results = MagicMock()
        mock_boxes = MagicMock()

        mock_boxes.conf = torch.tensor([0.7, 0.9])
        mock_boxes.xyxy = torch.tensor([
            [10, 10, 50, 50],
            [100, 100, 200, 200]
        ])
        mock_results.boxes = mock_boxes

        target = self.engine.find_best_target(mock_results)

        self.assertAlmostEqual(target.confidence, 0.9, places=6)
        self.assertEqual(target.center, (150, 150))
        self.assertEqual(target.size, (100, 100))
        self.assertFalse(target.is_lost)

    def test_find_best_target_empty(self):
        """Test behavior when no boxes are detected."""
        mock_results = MagicMock()
        mock_results.boxes = None

        target = self.engine.find_best_target(mock_results)

        self.assertEqual(target.confidence, -1.0)

    def test_process_mask(self):
        """Test the conversion of a tensor mask to a binary uint8 mask."""
        mask_tensor = torch.zeros((10, 10))
        mask_tensor[5, 5] = 1.0

        processed = self.engine._process_mask(mask_tensor, 20, 20)

        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, (20, 20))
        self.assertEqual(processed.dtype, np.uint8)
        self.assertTrue(255 in processed)

    def test_segment_image_no_detections(self):
        """Test segment_image handles frames with no objects correctly."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_results = MagicMock()
        mock_results.boxes.conf = torch.tensor([0.1])
        mock_results.boxes.max = lambda: torch.tensor(0.1)

        self.engine.model.predict.return_value = [mock_results]

        annotated, mask, masks_xy = self.engine.segment_image(frame)

        np.testing.assert_array_equal(annotated, frame)
        self.assertEqual(mask.sum(), 0)

    def test_draw_boxes(self):
        """Ensure drawing logic doesn't crash and modifies the frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[10, 10, 20, 20]])
        mock_boxes.cls = torch.tensor([0])
        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.__len__.return_value = 1

        mock_results = MagicMock()
        mock_results.boxes = mock_boxes
        mock_results.names = {0: "car"}

        self.engine._draw_boxes(frame, mock_results)

        self.assertGreater(frame.sum(), 0, "Frame should have boxes drawn on it")


if __name__ == '__main__':
    unittest.main()
