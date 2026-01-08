import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from auto_follow.detection.target_tracker import CommandInfo
from auto_follow.processors.eval_distilled_network_processor import EvalDistilledNetworkProcessor


class TestEvalDistilledNetworkProcessor(unittest.TestCase):

    def setUp(self):
        def mocked_parent_init(self, *args, **kwargs):
            self.detector = MagicMock()
            self.target_tracker = MagicMock()
            self.ibvs_controller = MagicMock()
            self.drone_commander = kwargs.get('drone_commander')
            self._frame_count = 0
            self.logger = MagicMock()
            self.recent_errors = deque(maxlen=5)
            self.recent_commands = np.zeros((5, 3))
            self.parquet_path = Path("/tmp/eval_logs")
            self.log_parquet = pd.DataFrame()
            self._flight_start_time = None
            self.frame_saver = MagicMock()
            self.frame_saver.output_dir = Path("/tmp/frames/dummy.jpg")

        self.patcher_init = patch('auto_follow.processors.ibvs_splitter_processor.IBVSSplitterProcessor.__init__',
                                  autospec=True, side_effect=mocked_parent_init)
        self.patcher_engine = patch('auto_follow.processors.eval_distilled_network_processor.StudentEngine')
        self.patcher_parquet = patch.object(pd.DataFrame, 'to_parquet')

        self.patcher_init.start()
        self.patcher_engine.start()
        self.mock_parquet_save = self.patcher_parquet.start()

        self.addCleanup(patch.stopall)

        self.mock_commander = MagicMock()
        self.processor = EvalDistilledNetworkProcessor(
            drone_commander=self.mock_commander,
            student_model_path="dummy_path"
        )

        self.processor.perform_movement = MagicMock()
        self.processor.check_goal_reached = MagicMock()
        self.processor.check_timout_landing = MagicMock()

    def test_process_frame_dual_logic_flow(self):
        """Verify that both student prediction and splitter calculation occur."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        self.processor._frame_count = 0

        self.processor.student_engine.predict.return_value = np.array([0.6, 1.4, 2.8])

        mock_target = MagicMock()
        mock_target.confidence = 0.9
        self.processor.detector.find_best_target.return_value = mock_target

        splitter_cmd = CommandInfo(
            timestamp=0, x_cmd=5, y_cmd=5, z_cmd=0, rot_cmd=5,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Splitter"
        )
        dummy_logs = {
            "jacobian_matrix": [0], "jcond": 0, "current_points_flatten": [0],
            "goal_points_flatten": [0], "err_uv": [0], "velocity": [0]
        }
        self.processor.target_tracker.calculate_movement.return_value = (splitter_cmd, dummy_logs)
        self.processor.ibvs_controller.err_uv_values = [[0, 0]]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(frame)

        self.processor.student_engine.predict.assert_called_once()
        called_cmd = self.processor.perform_movement.call_args[0][0]
        self.assertEqual(called_cmd.x_cmd, 1)
        self.mock_parquet_save.assert_called()

    def test_skip_odd_frames(self):
        """Ensure odd frames skip heavy inference but still show visualization."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        self.processor._frame_count = 1
        self.processor.last_command_info = CommandInfo(
            timestamp=0, x_cmd=1, y_cmd=1, z_cmd=0, rot_cmd=1,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Test"
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(self.processor, '_add_cmd_visualization') as mock_viz:
            self.processor._process_frame(frame)
            mock_viz.assert_called_once()
            self.processor.student_engine.predict.assert_not_called()

    def test_save_parquet_logs_student_columns(self):
        """Verify the specific 'student vs splitter' comparison columns in the log."""
        student_cmd = CommandInfo(
            timestamp=0, x_cmd=1, y_cmd=1, z_cmd=0, rot_cmd=1,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Student"
        )
        splitter_cmd = CommandInfo(
            timestamp=0, x_cmd=2, y_cmd=2, z_cmd=0, rot_cmd=2,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Splitter"
        )
        logs = {
            "jacobian_matrix": "jac", "jcond": 1.0, "current_points_flatten": "pts",
            "goal_points_flatten": "goal", "err_uv": "err", "velocity": "vel"
        }

        self.processor.log_parquet = pd.DataFrame()
        self.processor._save_parquet_logs_student({"timestamp": 10}, student_cmd, splitter_cmd, logs)

        df = self.processor.log_parquet
        self.assertEqual(df.iloc[0]["x_cmd"], 1)
        self.assertEqual(df.iloc[0]["splitter_x_cmd"], 2)

    def test_target_lost_resets_state(self):
        """Check that losing the target clears recent errors and goal timers."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        self.processor._frame_count = 0
        self.processor._soft_goal_enter_time = 123.45
        self.processor.recent_errors.append([1, 1])

        self.processor.student_engine.predict.return_value = np.array([0, 0, 0])

        mock_target = MagicMock()
        mock_target.confidence = -1
        self.processor.detector.find_best_target.return_value = mock_target

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(frame)

        self.assertIsNone(self.processor._soft_goal_enter_time)
        self.assertEqual(len(self.processor.recent_errors), 0)


if __name__ == '__main__':
    unittest.main()
