import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from auto_follow.detection.target_tracker import TargetTracker, TargetTrackerIBVS
from drone_base.config.video import VideoConfig


class TestTargetTracker(unittest.TestCase):
    def setUp(self):
        self.video_config = VideoConfig(width=640, height=360)
        self.tracker = TargetTracker(self.video_config)

    def test_calculate_movement_no_target(self):
        """Should return hover commands when target is lost."""
        cmd = self.tracker.calculate_movement(object_center=(320, 180), target_lost=True)
        self.assertEqual(cmd.status, "Lost/Hover")
        self.assertEqual(cmd.x_cmd, 0)
        self.assertEqual(cmd.rot_cmd, 0)

    def test_proportional_rotation(self):
        """Target to the right of center should trigger positive rotation."""
        # Offset to the right: (480 - 320) / 320 = 0.5
        # Command = kp_rot (20) * 0.5 = 10
        cmd = self.tracker.calculate_movement(
            object_center=(480, 180),
            box_size=(50, 50),
            target_lost=False
        )
        self.assertEqual(cmd.status, "Tracking")
        self.assertGreater(cmd.rot_cmd, 0)
        self.assertAlmostEqual(cmd.x_offset, 0.5)

    def test_derivative_rotation(self):
        """Rapid movement should increase rotation command via D-term."""
        self.tracker.calculate_movement((320, 180), (50, 50), False)

        with patch('time.time', return_value=self.tracker.last_command_time + 0.1):
            # Move target significantly to the right
            cmd = self.tracker.calculate_movement((400, 180), (50, 50), False)

            # d_rot = kd_rot * (delta_offset / dt)
            # delta_offset = (400 - 320) / 320 = 0.25. dt = 0.1.
            # d_rot = 5 * (0.25 / 0.1) = 12.5
            self.assertAlmostEqual(cmd.d_rot, 12.5, places=4)


class TestTargetTrackerIBVS(unittest.TestCase):
    def setUp(self):
        self.video_config = VideoConfig(width=640, height=360)
        self.mock_ibvs = MagicMock()
        self.tracker = TargetTrackerIBVS(self.video_config, self.mock_ibvs)

    def test_ibvs_velocity_scaling(self):
        """Test conversion from m/s to percentage commands."""
        # Setup: IBVS suggests moving forward at 1m/s
        # max_linear_speed is 2m/s, so pitch should be 50%
        # velocities order: [vx, vy, vz, wz] (usually roll, pitch, gaz, yaw)
        self.mock_ibvs.compute_velocities.return_value = (
            np.array([0.0, -1.0, 0.0, 0.0]),
            {"error": 0.1}
        )

        target_mock = MagicMock()
        target_mock.bbox_oriented = [(0, 0), (10, 0), (10, 10), (0, 10)]

        cmd, logs = self.tracker.calculate_movement(target_mock)

        # pitch = ceil(-100 * -1.0 / 2.0) = 50
        self.assertEqual(cmd.y_cmd, 50)
        self.assertEqual(cmd.status, "IBVS")

    def test_yaw_threshold(self):
        """Yaw below threshold should be zeroed out."""
        # Max angular speed is 60 deg/s.
        self.mock_ibvs.compute_velocities.return_value = (
            np.array([0.0, 0.0, 0.01, 0.0]),  # Small yaw
            {}
        )

        target_mock = MagicMock()
        cmd, _ = self.tracker.calculate_movement(target_mock)

        # Since yaw (0.01 rad/s) scaled to percentage is very small,
        # it should fall below const_yaw_threshold (8) and become 0.
        self.assertEqual(cmd.rot_cmd, 0)


if __name__ == '__main__':
    unittest.main()
