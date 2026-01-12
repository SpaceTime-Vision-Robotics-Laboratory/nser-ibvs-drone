import unittest
from unittest.mock import patch

from nser_ibvs_drone.pid.pid_controller import PIDController, PIDConfig, FilterConfig


class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.pid_cfg = PIDConfig(
            kp=1.0, ki=0.5, kd=0.1,
            dead_zone=1.0,
            thresholds=(-10.0, 10.0),
            max_integral=50.0
        )
        self.filter_cfg = FilterConfig(mode="exp", alpha=1.0)

        self.patcher = patch('time.perf_counter')
        self.mock_time = self.patcher.start()
        self.mock_time.return_value = 0.0

        self.controller = PIDController(self.pid_cfg, self.filter_cfg)

    def tearDown(self):
        self.patcher.stop()

    def test_dead_zone(self):
        """If error is within dead_zone, output should be 0 and state reset."""
        self.controller.integral = 10.0

        output = self.controller.compute(0.5)

        self.assertEqual(output, 0.0)
        self.assertEqual(self.controller.integral, 0.0)
        self.assertEqual(self.controller.last_error, 0.0)

    def test_basic_p_action(self):
        """Test simple Proportional action (dt=1s, ki=0, kd=0)."""
        self.controller.config.thresholds = (-100.0, 100.0)
        self.controller.config.ki = 0.0
        self.controller.config.kd = 0.0
        self.controller.config.kp = 1.0

        # Move time forward by 1 second
        self.mock_time.return_value = 1.0

        # process_error = -10 (Target is 0, so filtered_error = 0 - (-10) = 10)
        # Adaptive Kp will kick in, making it > 1.0
        output = self.controller.compute(-10.0)

        self.assertGreater(output, 10.0)  # Kp * 10 where Kp is adapted upwards

    def test_integral_windup_and_clipping(self):
        """Test that the integral does not exceed max_integral."""
        self.controller.config.kp = 0.0
        self.controller.config.kd = 0.0
        self.controller.config.ki = 1.0

        for i in range(1, 10):
            self.mock_time.return_value = float(i)
            self.controller.compute(-20.0)

        self.assertLessEqual(abs(self.controller.integral), self.pid_cfg.max_integral)

    def test_output_threshold_clipping(self):
        """Test that the output is clipped to the thresholds."""
        # Force a huuuge output via high Kp
        self.controller.config.kp = 1000.0
        self.mock_time.return_value = 1.0

        output = self.controller.compute(-50.0)

        self.assertEqual(output, 10.0)

    def test_adaptive_gains(self):
        """Verify that Kp increases and Ki decreases with larger errors."""
        self.controller.config.dead_zone = 1.0
        kp_small = self.controller._adapt_kp(2.0)
        ki_small = self.controller._adapt_ki(2.0)

        kp_large = self.controller._adapt_kp(100.0)
        ki_large = self.controller._adapt_ki(100.0)

        self.assertGreater(kp_large, kp_small)
        self.assertLess(ki_large, ki_small)

    def test_update_config(self):
        """Test that update_config correctly modifies values."""
        self.controller.update_config(kp=5.0, target_error=10.0)
        self.assertEqual(self.controller.config.kp, 5.0)
        self.assertEqual(self.controller.config.target_error, 10.0)
        self.assertEqual(self.controller.config.ki, 0.5)

    def test_reset(self):
        """Test that reset clears the internal state and resets time."""
        self.controller.integral = 25.0
        self.controller.last_error = 5.0

        self.mock_time.return_value = 500.0
        self.controller.reset()

        self.assertEqual(self.controller.integral, 0.0)
        self.assertEqual(self.controller.last_error, 0.0)
        self.assertEqual(self.controller.prev_time, 500.0)


if __name__ == "__main__":
    unittest.main()
