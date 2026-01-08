import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
from pathlib import Path

from auto_follow.processors.distilled_network_processor import DistilledNetworkProcessor
from auto_follow.detection.target_tracker import CommandInfo


class TestDistilledNetworkProcessor(unittest.TestCase):

    def setUp(self):
        def mocked_parent_init(self, *args, **kwargs):
            self.frame_saver = kwargs.get('video_config').frame_saver
            self.detector = MagicMock()
            self.drone_commander = kwargs.get('drone_commander')
            self._frame_count = 0
            self.logger = MagicMock()
            self.config = kwargs.get('video_config')
            self.frame_saver.output_dir = Path("/tmp/test/frames")

        self.patcher_init = patch('auto_follow.processors.ibvs_yolo_processor.IBVSYoloProcessor.__init__', autospec=True, side_effect=mocked_parent_init)
        self.patcher_engine = patch('auto_follow.processors.distilled_network_processor.StudentEngine')
        self.patcher_intrinsic = patch('auto_follow.processors.ibvs_yolo_processor.infer_intrinsic_matrix', return_value=np.eye(3))
        self.patcher_parquet = patch.object(pd.DataFrame, 'to_parquet')

        self.patcher_init.start()
        self.patcher_engine.start()
        self.patcher_intrinsic.start()
        self.mock_parquet_save = self.patcher_parquet.start()

        self.addCleanup(patch.stopall)

        self.mock_config = MagicMock()
        self.mock_commander = MagicMock()
        self.mock_config.frame_saver.output_dir = Path("/tmp/test/frames")

        with patch('builtins.open', mock_open(read_data='{"bbox_oriented_points": [[0,0], [1,1], [2,2], [3,3]]}')):
            self.processor = DistilledNetworkProcessor(
                video_config=self.mock_config,
                frame_queue=MagicMock(),
                drone_commander=self.mock_commander,
                logs_parquet_path=None
            )

        self.processor.log_parquet = pd.DataFrame(columns=[
            "timestamp", "frame_idx", "x_cmd", "y_cmd", "z_cmd", "rot_cmd"
        ])
        self.processor.parquet_path = Path("/tmp/dummy")

    def test_command_rounding_logic(self):
        """Verify the threshold-based rounding for student engine output."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)

        mock_target = MagicMock()
        mock_target.confidence = 0.9
        self.processor.detector.find_best_target.return_value = mock_target

        # StudentEngine returns [x, y, rot]
        # 0.6 (>0.5) -> 1.0
        # 1.4 (<0.5) -> 1.0
        # 2.8 (>0.5) -> 3.0
        self.processor.student_engine.predict.return_value = np.array([0.6, 1.4, 2.8])
        self.processor.int_threshold = 0.5

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(frame)

        last_cmd = self.processor.last_command_info
        self.assertEqual(last_cmd.x_cmd, 1)
        self.assertEqual(last_cmd.y_cmd, 1)
        self.assertEqual(last_cmd.rot_cmd, 3)

    def test_is_stable_at_goal(self):
        """Verify the 'at goal' check (commands <= 1)."""
        # Case 1: All commands within [-1, 1]
        self.processor.recent_commands = np.array([[1, 0, -1], [0, 1, 0]])
        self.assertTrue(self.processor._is_stable_at_goal())

        # Case 2: One command exceeds threshold
        self.processor.recent_commands = np.array([[1, 0, 2], [0, 1, 0]])
        self.assertFalse(self.processor._is_stable_at_goal())

    def test_skip_processing_on_odd_frames(self):
        """Verify odd frames only draw visualization and don't run inference."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        self.processor._frame_count = 1
        self.processor.last_command_info = CommandInfo(
            timestamp=0, x_cmd=5, y_cmd=5, z_cmd=0, rot_cmd=5,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Test"
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(self.processor, '_add_cmd_visualization') as mock_viz:
            self.processor._process_frame(frame)
            mock_viz.assert_called_once()
            # In frame 1, student_engine should NOT be called
            self.processor.student_engine.predict.assert_not_called()

    def test_goal_reached_and_landing(self):
        """Test the sequence from entering goal to landing."""
        self.processor._flight_start_time = 100.0
        self.processor._is_stable_at_goal = MagicMock(return_value=True)

        with patch.object(Path, 'open', mock_open()) as m_open:
            # Enters goal at T=105
            self.processor.check_goal_reached(105.0)
            self.assertEqual(self.processor._command_zero_time, 105.0)
            self.mock_commander.land.assert_not_called()

            # T=109 (> 3s threshold)
            self.processor.check_goal_reached(109.0)
            self.mock_commander.land.assert_called_once()

            handle = m_open()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertIn('"status": "complete-goal"', written_data)

    def test_parquet_logging(self):
        """Ensure parquet logging creates the expected columns and triggers the save call."""
        cmd = CommandInfo(
            timestamp=0, x_cmd=1, y_cmd=2, z_cmd=0, rot_cmd=3,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Test"
        )

        self.processor._save_parquet_logs({"timestamp": 123, "frame_idx": 1}, cmd, {})

        self.assertEqual(len(self.processor.log_parquet), 1)
        self.assertEqual(self.processor.log_parquet.iloc[0]['x_cmd'], 1)
        self.mock_parquet_save.assert_called()


if __name__ == '__main__':
    unittest.main()