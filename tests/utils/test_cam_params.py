import unittest
from unittest.mock import patch, mock_open

import numpy as np

from nser_ibvs_drone.utils.cam_params import (
    read_pkl_file,
    scale_intrinsic_matrix_by_factor,
    scale_intrinsic_matrix_by_size,
    infer_intrinsic_matrix
)


class TestCameraUtils(unittest.TestCase):

    def setUp(self):
        self.sample_k = np.array([
            [1000, 0, 500],
            [0, 1000, 400],
            [0, 0, 1]
        ], dtype=float)

    def test_scale_intrinsic_matrix_by_factor(self):
        factor = 0.5
        scaled = scale_intrinsic_matrix_by_factor(self.sample_k, factor)

        # Expected: fx, fy, cx, cy are halved. [2,2] remains 1.0
        self.assertEqual(scaled[0, 0], 500.0)
        self.assertEqual(scaled[1, 1], 500.0)
        self.assertEqual(scaled[0, 2], 250.0)
        self.assertEqual(scaled[1, 2], 200.0)
        self.assertEqual(scaled[2, 2], 1.0)

    def test_scale_intrinsic_matrix_by_size(self):
        orig_size = (1000, 800)
        new_size = (500, 200)  # Width halved (0.5), Height quartered (0.25)

        scaled = scale_intrinsic_matrix_by_size(self.sample_k, orig_size, new_size)

        self.assertEqual(scaled[0, 0], 500.0)  # 1000 * 0.5
        self.assertEqual(scaled[1, 1], 250.0)  # 1000 * 0.25
        self.assertEqual(scaled[0, 2], 250.0)  # 500 * 0.5
        self.assertEqual(scaled[1, 2], 100.0)  # 400 * 0.25

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_read_pkl_file(self, mock_pickle_load, mock_file):
        mock_pickle_load.return_value = {"data": 123}

        result = read_pkl_file("fake_path.pkl")

        mock_file.assert_called_once_with("fake_path.pkl", 'rb')
        self.assertEqual(result, {"data": 123})

    @patch("nser_ibvs_drone.utils.cam_params.read_pkl_file")
    def test_infer_intrinsic_matrix_real(self, mock_read):
        mock_read.return_value = self.sample_k

        result = infer_intrinsic_matrix("/path/to/real_camera_params")

        self.assertTrue(np.array_equal(result, self.sample_k))
        self.assertEqual(mock_read.call_count, 1)

    @patch("nser_ibvs_drone.utils.cam_params.read_pkl_file")
    def test_infer_intrinsic_matrix_sim(self, mock_read):
        mock_read.side_effect = [self.sample_k, self.sample_k * 0.5]

        res_full, res_half = infer_intrinsic_matrix("/path/to/sim_camera_params")

        self.assertTrue(np.array_equal(res_full, self.sample_k))
        self.assertTrue(np.array_equal(res_half, self.sample_k * 0.5))
        self.assertEqual(mock_read.call_count, 2)

    def test_infer_intrinsic_matrix_invalid(self):
        with self.assertRaises(ValueError):
            infer_intrinsic_matrix("/path/to/unknown_format")


if __name__ == "__main__":
    unittest.main()
