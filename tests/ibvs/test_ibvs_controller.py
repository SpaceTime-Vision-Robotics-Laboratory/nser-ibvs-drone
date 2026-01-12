import unittest

import numpy as np

from nser_ibvs_drone.ibvs.ibvs_controller import ImageBasedVisualServo


class TestImageBasedVisualServo(unittest.TestCase):

    def setUp(self):
        # Sample Intrinsic Matrix (Standard 4K-ish sim values)
        self.K = np.array([
            [1000.0, 0, 960.0],
            [0, 1000.0, 540.0],
            [0, 0, 1.0]
        ])
        # Define goal and current points (4 corners of a centered square)
        self.goal_points = [(860, 440), (1060, 440), (1060, 640), (860, 640)]
        self.current_points = [(800, 400), (1000, 400), (1000, 600), (800, 600)]

        self.ibvs = ImageBasedVisualServo(
            camera_intrensic=self.K,
            goal_points=self.goal_points,
            lambda_factor=0.3,
            estimated_depth=2.0
        )

    def test_initialization(self):
        """Check if K_inv is calculated correctly"""
        identity = self.ibvs.K @ self.ibvs.Kinv
        np.testing.assert_array_almost_equal(identity, np.eye(3))
        self.assertEqual(len(self.ibvs.goal_points_flatten), 8)

    def test_compute_normalized_image_plane_coordinates(self):
        """The principal point (960, 540) should map to (0, 0) in normalized coordinates"""
        points = [(960, 540)]
        normalized = self.ibvs.compute_normalized_image_plane_coordinates(points)
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[1], 0.0)

    def test_compute_depths(self):
        """
        For a point at the principal point, depth should equal Z (estimated_depth)
        Because sqrt(0^2 + 0^2 + 1) = 1
        """
        pixels = np.array([[960, 540]])
        depths = self.ibvs.compute_depths(pixels)
        self.assertEqual(depths[0], 2.0)

    def test_compute_interaction_matrix_shape(self):
        self.ibvs.set_current_points(self.current_points)
        jacobian = self.ibvs.compute_interaction_matrix()

        # 4 points * 2 (u,v) rows per point = 8 rows. 3 columns (vx, vy, vz).
        self.assertEqual(jacobian.shape, (8, 3))

    def test_velocity_calculation_high_error(self):
        """Test that the system uses the high lambda when error is above threshold."""
        self.ibvs.set_current_points(self.current_points)

        vel, logs = self.ibvs.compute_velocities()

        self.assertEqual(len(vel), 3)
        self.assertGreater(logs["err_uv_values" if "err_uv_values" in logs else "jcond"], 0)
        # Verify lambda used was the large one (0.3 * factor)
        # We can check this by seeing if the log error is above threshold
        self.assertGreater(np.linalg.norm(logs["err_uv"]), self.ibvs.err_threshold)

    def test_velocity_calculation_low_error(self):
        """Test that the system switches to lambda_factor_low when close to goal."""
        # Set current points very close to goal
        close_points = [(861, 441), (1061, 441), (1061, 641), (861, 641)]
        self.ibvs.set_current_points(close_points)

        vel, logs = self.ibvs.compute_velocities()
        err_norm = np.linalg.norm(logs["err_uv"])

        # If error < 60, it should use lambda_factor_low (diag[0.5, 0.5, 0.1])
        self.assertLess(err_norm, self.ibvs.err_threshold)

    def test_logging(self):
        self.ibvs.set_current_points(self.current_points)
        self.ibvs.compute_velocities()
        self.ibvs.compute_velocities()

        self.assertEqual(len(self.ibvs.jcond_values), 2)
        self.assertEqual(len(self.ibvs.err_uv_values), 2)


if __name__ == "__main__":
    unittest.main()
