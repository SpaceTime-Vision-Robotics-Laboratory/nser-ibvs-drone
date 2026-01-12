import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from nser_ibvs_drone.detection.frame_visualizer import FrameVisualizer, FrameVisualizerIBVS

from nser_ibvs_drone.detection.targets import Target, TargetIBVS
from drone_base.config.video import VideoConfig


class TestFrameVisualizer(unittest.TestCase):
    def setUp(self):
        self.video_config = VideoConfig(width=640, height=360)
        self.visualizer = FrameVisualizer(self.video_config)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_draw_frame_lost(self):
        """Test drawing when no target is detected."""
        target = Target(confidence=0.8, is_lost=True)
        updated_frame, window_name = self.visualizer.draw_frame(self.frame, target)

        self.assertEqual(window_name, "Drone View")
        self.assertIsInstance(updated_frame, np.ndarray)
        self.assertTrue(np.any(updated_frame > 0))

    def test_draw_frame_tracking(self):
        """Test drawing when target is found."""
        target = Target(
            is_lost=False,
            center=(320, 180),
            box=(100, 100, 200, 200),
            confidence=0.85
        )

        updated_frame, _ = self.visualizer.draw_frame(self.frame, target)

        self.assertTrue(np.any(updated_frame > 0))


class TestFrameVisualizerIBVS(unittest.TestCase):
    def setUp(self):
        self.video_config = VideoConfig(width=640, height=360)
        self.visualizer = FrameVisualizerIBVS(self.video_config)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    @patch('nser_ibvs_drone.detection.frame_visualizer.plot_bbox_keypoints')
    def test_display_frame_ibvs(self, mock_plot_keypoints):
        """Test IBVS-specific drawing (oriented bbox and keypoints)."""
        target = TargetIBVS(
            is_lost=False,
            bbox_oriented=[(100, 100), (200, 100), (200, 200), (100, 200)],
            center=(150, 150),
            box=(100, 100, 200, 200),
            masks_xy=np.array([[100, 100]]),
            confidence=0.9
        )
        goal_points = [(110, 110), (190, 110), (190, 190), (110, 190)]
        mock_controller = MagicMock()

        self.visualizer.display_frame(self.frame, target, mock_controller, goal_points)

        # Verify that plot_bbox_keypoints was called twice (once for target, once for goal)
        self.assertEqual(mock_plot_keypoints.call_count, 2)


if __name__ == '__main__':
    unittest.main()
