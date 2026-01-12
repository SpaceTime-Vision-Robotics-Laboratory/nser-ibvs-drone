import unittest
from collections import deque
from pathlib import Path
from time import perf_counter
from unittest.mock import MagicMock, patch, mock_open

import numpy as np

from nser_ibvs_drone.detection.target_tracker import CommandInfo
from nser_ibvs_drone.processors.ibvs_splitter_processor import IBVSSplitterProcessor


class TestIBVSSplitterProcessor(unittest.TestCase):

    def setUp(self):
        def mocked_parent_init(self, *args, **kwargs):
            self.frame_saver = kwargs.get('video_config').frame_saver
            self.ibvs_controller = MagicMock()
            self.target_tracker = MagicMock()
            self.visualizer = MagicMock()
            self.parquet_path = kwargs.get('logs_parquet_path')
            self._frame_count = 0
            self.drone_commander = kwargs.get('drone_commander')
            self.logger = MagicMock()

        with patch('nser_ibvs_drone.processors.ibvs_yolo_processor.IBVSYoloProcessor.__init__', autospec=True,
                   side_effect=mocked_parent_init), \
                patch('nser_ibvs_drone.processors.ibvs_yolo_processor.infer_intrinsic_matrix', return_value=np.eye(3)), \
                patch('nser_ibvs_drone.processors.ibvs_splitter_processor.MaskSplitterEngineIBVS'), \
                patch('nser_ibvs_drone.processors.ibvs_yolo_processor.Paths'), \
                patch('builtins.open', mock_open(read_data='{"bbox_oriented_points": [[0,0], [1,1], [2,2], [3,3]]}')):
            self.mock_config = MagicMock()
            self.mock_commander = MagicMock()
            self.mock_config.frame_saver = MagicMock()
            self.mock_config.frame_saver.output_dir = Path("/tmp/test/frames")

            self.processor = IBVSSplitterProcessor(
                video_config=self.mock_config,
                frame_queue=MagicMock(),
                drone_commander=self.mock_commander,
                logs_parquet_path=None
            )

            self.processor.ibvs_controller.err_uv_values = [0.0]

            self.default_cmd = CommandInfo(
                timestamp=perf_counter(),
                x_cmd=0, y_cmd=0, z_cmd=0, rot_cmd=0,
                x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Unknown"
            )
            mock_logs = {"velocity": [0] * 6, "jcond": 0.0, "err_uv": np.array([0, 0])}

            self.processor.target_tracker.calculate_movement.return_value = (self.default_cmd, mock_logs)

            mock_target = MagicMock()
            mock_target.confidence = 0.9
            self.processor.detector.find_best_target.return_value = mock_target
            self.processor.logger = MagicMock()

    def test_flight_start_time_initialization(self):
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        with patch('time.perf_counter', return_value=100.0):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.processor._process_frame(frame)
            self.assertEqual(self.processor._flight_start_time, 100.0)

    def test_skip_processing_on_odd_frames(self):
        """Every second frame should skip detection and only show visualization."""
        self.processor._check_start_drone_state = MagicMock(return_value=True)
        self.processor._frame_count = 1

        self.processor.last_command_info = CommandInfo(
            timestamp=perf_counter(),
            x_cmd=10, y_cmd=20, z_cmd=30, rot_cmd=5,
            x_offset=0, y_offset=0, p_rot=0, d_rot=0, status="Active"
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(frame)
        self.processor.detector.find_best_target.assert_not_called()

    def test_is_stable_at_goal_logic(self):
        """Test the median error and command-zero check."""
        self.processor.recent_errors = deque([30.0, 25.0, 35.0], maxlen=5)
        self.processor.recent_commands = np.zeros((5, 3))
        hard, soft = self.processor._is_stable_at_goal()
        self.assertTrue(hard)
        self.assertTrue(soft)

    def test_hard_goal_landing_trigger(self):
        """Verify drone lands and writes JSON."""
        self.processor._flight_start_time = 0.0
        with patch.object(Path, 'open', mock_open()) as m_open:
            # T=10 Enter goal
            self.processor.handle_hard_goal_reach(10.0, True)
            # T=13 (3s later) -> Trigger land
            self.processor.handle_hard_goal_reach(13.0, True)

            self.mock_commander.land.assert_called_once()
            handle = m_open()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertIn('"status": "complete"', written_data)

    def test_timeout_landing(self):
        """Verify drone lands if flight duration exceeds timeout."""
        self.processor._flight_start_time = 10.0
        self.processor.timeout_seconds = 75

        with patch.object(Path, 'open', mock_open()) as m_open:
            # 90 - 10 = 80s (> 75s timeout)
            self.processor.check_timout_landing(90.0)
            self.mock_commander.land.assert_called_once()

            handle = m_open()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertIn('"status": "timeout"', written_data)


if __name__ == '__main__':
    unittest.main()
