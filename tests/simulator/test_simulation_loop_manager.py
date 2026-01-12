import unittest
from unittest.mock import patch

from nser_ibvs_drone.simulator.simulation_loop_manager import SimulationLoopManager


class TestSimulationLoopManager(unittest.TestCase):

    def setUp(self):
        """Set up a fresh manager before each test."""
        self.sphinx_dir = "/mock/path"
        self.manager = SimulationLoopManager(
            sphinx_base_dir=self.sphinx_dir,
            target_runs=2,
            is_student=False
        )
        self.manager.delay_between_runs = 0

    def test_prepare_scene_name_standard(self):
        """Verify scene name remains unchanged when is_student is False."""
        scene = "test_scene.yaml"
        result = self.manager._prepare_scene_name(scene)
        self.assertEqual(result, "test_scene.yaml")

    def test_prepare_scene_name_student(self):
        """Verify student suffix is added correctly."""
        self.manager.is_student = True
        scene = "test_scene.yaml"
        result = self.manager._prepare_scene_name(scene)
        self.assertEqual(result, "test_scene-student.yaml")

    @patch("nser_ibvs_drone.simulator.simulation_loop_manager.main_simulation_runner")
    @patch("time.sleep", return_value=None)
    def test_run_single_config_success(self, mock_sleep, mock_runner):
        """Test a successful run where target_runs is met."""
        mock_runner.return_value = True

        self.manager.run_single_config("scene1.yaml")

        self.assertEqual(mock_runner.call_count, 2)
        self.assertEqual(self.manager.successful_runs, 2)
        self.assertEqual(len(self.manager.failures), 0)

    @patch("nser_ibvs_drone.simulator.simulation_loop_manager.main_simulation_runner")
    @patch("time.sleep", return_value=None)
    def test_run_single_config_with_failures(self, mock_sleep, mock_runner):
        """Test tracking failures when the simulation fails once then succeeds."""
        mock_runner.side_effect = [False, True, True]

        self.manager.run_single_config("scene1.yaml")

        # Total attempts should be 3 to get 2 successes
        self.assertEqual(self.manager.total_attempts, 3)
        self.assertEqual(self.manager.successful_runs, 2)
        self.assertEqual(len(self.manager.failures), 1)
        self.assertEqual(self.manager.failures[0], 1)  # Failed on the first attempt

    @patch("nser_ibvs_drone.simulator.simulation_loop_manager.SimulationLoopManager.run_single_config")
    def test_run_suite(self, mock_run_single):
        """Verify that the suite iterates through all provided scenes."""
        scenes = ["s1.yaml", "s2.yaml", "s3.yaml"]
        self.manager.run_suite(scenes)

        self.assertEqual(mock_run_single.call_count, 3)


if __name__ == "__main__":
    unittest.main()
