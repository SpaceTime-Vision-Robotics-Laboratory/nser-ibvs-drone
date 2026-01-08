import unittest
from unittest.mock import MagicMock

import numpy as np

from auto_follow.pid.filters import LowPassFilter, FilterConfig


class TestLowPassFilter(unittest.TestCase):

    def setUp(self):
        """Set up a default config and filter instance for testing."""
        self.default_config = FilterConfig(
            alpha=0.5,
            wp=0.2,
            ws=0.3,
            filter_order=4,
            mode="exp"
        )
        self.filter = LowPassFilter(self.default_config)

    def test_initialization(self):
        """Check if wn and coefficients are calculated correctly during init."""
        expected_wn = (0.2 + 0.3) / 2
        self.assertEqual(self.filter.wn, expected_wn)
        # FIR order 4 means 5 coefficients
        self.assertEqual(len(self.filter.coefficients), 5)
        self.assertEqual(self.filter.buffer.maxlen, 5)

    def test_exponential_filter_first_value(self):
        """The first value processed should return itself (initialization)."""
        result = self.filter.exp_filter(10.0)
        self.assertEqual(result, 10.0)
        self.assertEqual(self.filter.filtered_value, 10.0)

    def test_exponential_filter_smoothing(self):
        """Verify the math: alpha * raw + (1 - alpha) * prev."""
        self.filter.exp_filter(10.0)
        # alpha = 0.5, raw = 20.0 -> 0.5 * 20 + 0.5 * 10 = 15.0
        result = self.filter.exp_filter(20.0)
        self.assertEqual(result, 15.0)

    def test_fir_fallback_to_exp(self):
        """FIR should use exp_filter until the buffer is full."""
        self.filter.config.mode = "fir"
        # Buffer needs 5 samples (order 4 + 1)
        result = self.filter(10.0)
        self.assertEqual(result, 10.0)

    def test_fir_filter_full_calculation(self):
        """Verify FIR convolution happens when buffer is full."""
        self.filter.config.mode = "fir"
        coeffs = self.filter.coefficients

        # Fill buffer with 1.0s (except the last one)
        for _ in range(self.filter.config.filter_order):
            self.filter(1.0)

        # The next call triggers the dot product
        # If all inputs are 1.0, the result of a normalized FIR filter is ~1.0
        result = self.filter(1.0)

        expected = np.dot(coeffs, np.ones(len(coeffs)))
        self.assertAlmostEqual(result, expected, places=5)

    def test_reset(self):
        """Test that reset clears the history."""
        self.filter.exp_filter(100.0)
        self.filter.buffer.append(50.0)

        self.filter.reset()

        self.assertIsNone(self.filter.filtered_value)
        self.assertEqual(len(self.filter.buffer), 0)

        # After reset, the next value should be treated as the first (no smoothing)
        self.assertEqual(self.filter.exp_filter(25.0), 25.0)

    def test_routing(self):
        """Test that __call__ routes to the correct method based on mode."""
        self.filter.exp_filter = MagicMock(return_value=1.0)
        self.filter.fir_filter = MagicMock(return_value=2.0)

        self.filter.config.mode = "exp"
        self.assertEqual(self.filter(10.0), 1.0)

        self.filter.config.mode = "fir"
        self.assertEqual(self.filter(10.0), 2.0)


if __name__ == "__main__":
    unittest.main()
