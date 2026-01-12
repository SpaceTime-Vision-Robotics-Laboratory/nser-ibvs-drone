import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
from pathlib import Path

from nser_ibvs_drone.processors.distilled_temporal_network_processor import DistilledTemporalNetworkProcessor
from nser_ibvs_drone.detection.target_tracker import CommandInfo


class TestDistilledTemporalNetworkProcessor(unittest.TestCase):

    def setUp(self):
        def mocked_parent_init(self, *args, **kwargs):
            self.frame_saver = MagicMock()
            self.frame_saver.output_dir = Path("/tmp/test/frames")
            self.detector = MagicMock()
            self.drone_commander = kwargs.get('drone_commander')
            self._frame_count = 0
            self.logger = MagicMock()
            self.config = kwargs.get('video_config')

        self.patcher_init = patch('nser_ibvs_drone.processors.ibvs_yolo_processor.IBVSYoloProcessor.__init__',
                                  autospec=True, side_effect=mocked_parent_init)
        self.patcher_engine = patch('nser_ibvs_drone.processors.distilled_temporal_network_processor.TemporalStudentEngine')
        self.patcher_parquet = patch.object(pd.DataFrame, 'to_parquet')

        self.patcher_init.start()
        self.patcher_engine.start()
        self.mock_parquet_save = self.patcher_parquet.start()

        self.addCleanup(patch.stopall)

        self.mock_config = MagicMock()
        self.mock_commander = MagicMock()

        with patch.object(Path, 'mkdir'):
            self.processor = DistilledTemporalNetworkProcessor(
                video_config=self.mock_config,
                frame_queue=MagicMock(),
                drone_commander=self.mock_commander,
                logs_parquet_path="/tmp/dummy_logs"
            )

        # Ensure the results path exists in our mock environment
        self.processor.results_path = Path("/tmp/test/flight_duration.json")

    def test_process_frame_full_flow(self):
        """Test the standard processing loop for a single frame."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)

        # Mock detector finding a target
        mock_target = MagicMock()
        mock_target.confidence = 0.9
        self.processor.detector.find_best_target.return_value = mock_target

        # Mock Student Engine output
        self.processor.student_engine.predict.return_value = np.array([0.6, 1.4, 2.1])
        self.processor.int_threshold = 0.5

        # We mock perform_movement on the processor to see if it was triggered
        self.processor.perform_movement = MagicMock()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Perform processing
        self.processor._process_frame(frame)

        # 1. Verify Command Creation in the buffer
        self.assertEqual(self.processor.recent_commands[-1].tolist(), [1, 1, 2])

        # 2. Verify perform_movement was called with a CommandInfo object
        self.processor.perform_movement.assert_called_once()

        # 3. Verify the values inside the command passed to perform_movement
        called_command = self.processor.perform_movement.call_args[0][0]
        self.assertIsInstance(called_command, CommandInfo)
        self.assertEqual(called_command.x_cmd, 1)
        self.assertEqual(called_command.y_cmd, 1)
        self.assertEqual(called_command.rot_cmd, 2)

        # 4. Verify Parquet saving was triggered
        self.mock_parquet_save.assert_called()

    def test_is_stable_at_goal_strict(self):
        """Verify the strict 'all zeros' goal stability check."""
        # Case 1: All zeros
        self.processor.recent_commands = np.zeros((self.processor.error_window_size, 3))
        self.assertTrue(self.processor._is_stable_at_goal())

        # Case 2: One non-zero value (even small)
        self.processor.recent_commands = np.zeros((self.processor.error_window_size, 3))
        self.processor.recent_commands[0, 0] = 1
        self.assertFalse(self.processor._is_stable_at_goal())

    def test_timeout_landing(self):
        """Ensure drone lands after the timeout period."""
        self.processor._flight_start_time = 100.0

        with patch.object(Path, "open", mock_open()):
            # T=176 is 76s after start (limit is 75s)
            self.processor.check_timout_landing(176.0)

            self.mock_commander.land.assert_called_once()
            self.assertEqual(self.processor._flight_end_time, 176.0)

    def test_target_lost_resets_goal_timer(self):
        """If target is lost (confidence -1), goal timer should reset."""
        self.processor._command_zero_time = 500.0

        mock_target = MagicMock()
        mock_target.confidence = -1
        self.processor.detector.find_best_target.return_value = mock_target
        self.processor._check_start_drone_state = MagicMock(return_value=True)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(frame)

        self.assertIsNone(self.processor._command_zero_time)

    def test_parquet_logging_content(self):
        """Verify the data appended to the parquet dataframe."""
        cmd = CommandInfo(
            timestamp=1.0, x_cmd=10, y_cmd=20, z_cmd=0, rot_cmd=30,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Test"
        )
        row = {"timestamp": 123.45, "frame_idx": 5}

        self.processor._save_parquet_logs(row, cmd, {})

        last_row = self.processor.log_parquet.iloc[-1]
        self.assertEqual(last_row["x_cmd"], 10)
        self.assertEqual(last_row["rot_cmd"], 30)
        self.assertEqual(last_row["frame_idx"], 5)


if __name__ == '__main__':
    unittest.main()