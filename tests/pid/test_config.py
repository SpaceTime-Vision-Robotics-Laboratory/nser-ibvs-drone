import os
import tempfile
import unittest

from auto_follow.pid.config import PIDConfig, FilterConfig


class TestPIDConfig(unittest.TestCase):

    def test_valid_initialization(self):
        """Test that a valid PIDConfig is created correctly."""
        config = PIDConfig(kp=1.0, ki=0.1, kd=0.05, dead_zone=0.1, thresholds=(-10.0, 10.0))
        self.assertEqual(config.kp, 1.0)
        self.assertEqual(config.thresholds, (-10.0, 10.0))

    def test_invalid_thresholds(self):
        """Test that invalid thresholds (wrong length and min > max) raise ValueError."""
        with self.assertRaises(ValueError):
            PIDConfig(1, 0, 0, 0, thresholds=(1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            PIDConfig(1, 0, 0, 0, thresholds=(10.0, 5.0))

    def test_negative_values(self):
        """Test that negative dead_zone or max_integral raise ValueError."""
        with self.assertRaises(ValueError):
            PIDConfig(1, 0, 0, dead_zone=-1.0)
        with self.assertRaises(ValueError):
            PIDConfig(1, 0, 0, dead_zone=0, max_integral=-5.0)

    def test_yaml_serialization(self):
        """Test saving to and loading from YAML."""
        config = PIDConfig(kp=2.5, ki=0.5, kd=0.1, dead_zone=0.01, thresholds=(-5.0, 5.0))

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config.to_yaml(tmp_path)
            loaded_config = PIDConfig.from_yaml(tmp_path)
            self.assertEqual(config.kp, loaded_config.kp)
            self.assertEqual(config.thresholds, loaded_config.thresholds)
            self.assertIsInstance(loaded_config.thresholds, tuple)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFilterConfig(unittest.TestCase):

    def test_valid_defaults(self):
        """Test that default values are applied correctly."""
        config = FilterConfig()
        self.assertEqual(config.mode, "fir")
        self.assertEqual(config.alpha, 0.2)

    def test_invalid_ranges(self):
        """Test that out-of-bounds parameters raise ValueError."""
        with self.assertRaises(ValueError):
            FilterConfig(alpha=1.5)
        with self.assertRaises(ValueError):
            FilterConfig(wp=-0.1)
        with self.assertRaises(ValueError):
            FilterConfig(ws=1.1)
        with self.assertRaises(ValueError):
            FilterConfig(filter_order=-1)

    def test_invalid_mode(self):
        """Test that unsupported modes raise ValueError."""
        with self.assertRaises(ValueError):
            FilterConfig(mode="kalman")

    def test_yaml_serialization(self):
        """Test saving to and loading from YAML."""
        config = FilterConfig(alpha=0.5, mode="exp")

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config.to_yaml(tmp_path)
            loaded_config = FilterConfig.from_yaml(tmp_path)
            self.assertEqual(loaded_config.alpha, 0.5)
            self.assertEqual(loaded_config.mode, "exp")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
