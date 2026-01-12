import unittest
from unittest.mock import patch

import numpy as np

from nser_ibvs_drone.ibvs.ibvs_math_fcn import (
    e2h,
    get_rectangle_sides,
    plot_bbox_keypoints,
    check_stability,
    compute_distance
)


class TestMathFunctionsIBVS(unittest.TestCase):

    def test_e2h_valid_matrix(self):
        v = np.array([[1, 2, 3],
                      [4, 5, 6]])
        expected = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [1, 1, 1]])
        result = e2h(v)
        np.testing.assert_array_equal(result, expected)

    def test_e2h_invalid_input(self):
        with self.assertRaises(ValueError):
            e2h(np.array([1, 2]))

    def test_get_rectangle_sides(self):
        corners = [(0, 0), (3, 0), (3, 4), (0, 4)]
        short, long = get_rectangle_sides(corners)
        self.assertEqual(short, 3.0)
        self.assertEqual(long, 4.0)

    def test_compute_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertAlmostEqual(compute_distance(p1, p2), 5.0)

    def test_check_stability_true(self):
        """Sides 2 and 4 -> Ratio = 2.0. Mean 2.1, Std 0.2. (0.1 diff < 0.2)"""
        corners = [(0, 0), (2, 0), (2, 4), (0, 4)]
        self.assertTrue(check_stability(corners, 2.1, 0.2))

    def test_check_stability_false(self):
        """Ratio = 2.0. Mean 2.5, Std 0.2. (0.5 diff > 0.2)"""
        corners = [(0, 0), (2, 0), (2, 4), (0, 4)]
        self.assertFalse(check_stability(corners, 2.5, 0.2))

    @patch('cv2.circle')
    def test_plot_bbox_keypoints(self, mock_circle):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        keypoints = [(10, 10), (20, 20), (30, 30), (40, 40)]

        plot_bbox_keypoints(frame, keypoints)

        self.assertEqual(mock_circle.call_count, 4)

        mock_circle.assert_any_call(frame, (10, 10), 5, (255, 0, 0), -1)


if __name__ == "__main__":
    unittest.main()
