import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import json
import queue
from pathlib import Path

from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from drone_base.control.operations import PilotingCommand


class TestIBVSYoloProcessor(unittest.TestCase):

    def setUp(self):
        self.dummy_goal_points = {
            "bbox_oriented_points": [[100, 100], [200, 100], [200, 200], [100, 200]]
        }
        self.mock_json_content = json.dumps(self.dummy_goal_points)

        with patch('auto_follow.processors.ibvs_yolo_processor.infer_intrinsic_matrix') as mock_infer, \
                patch('auto_follow.processors.ibvs_yolo_processor.YoloEngineIBVS'), \
                patch('auto_follow.processors.ibvs_yolo_processor.TargetTrackerIBVS'), \
                patch('auto_follow.processors.ibvs_yolo_processor.FrameVisualizerIBVS'), \
                patch('auto_follow.processors.ibvs_yolo_processor.ImageBasedVisualServo'), \
                patch('auto_follow.processors.ibvs_yolo_processor.Paths') as mock_paths, \
                patch('builtins.open', mock_open(read_data=self.mock_json_content)):
            mock_infer.return_value = np.eye(3)

            mock_paths.SIM_CAR_IBVS_YOLO_PATH = "model.pt"
            mock_paths.GOAL_FRAME_POINTS_PATH_45 = "goal.json"
            mock_paths.LOG_PARQUET_DIR = None

            self.mock_config = MagicMock()
            self.mock_config.save_extension = ".jpg"
            self.mock_commander = MagicMock()
            self.mock_queue = queue.Queue()

            self.processor = IBVSYoloProcessor(
                video_config=self.mock_config,
                frame_queue=self.mock_queue,
                drone_commander=self.mock_commander,
                logs_parquet_path=None
            )
            self.processor.logger = MagicMock()

    def test_intrinsic_matrix_handling(self):
        """Verify that the processor correctly extracts the K matrix."""
        self.assertIsInstance(self.processor.K, np.ndarray)
        self.assertEqual(self.processor.K.shape, (3, 3))

    def test_process_frame_no_target_found(self):
        """If target_data.confidence is -1, it should return original frame without movement."""
        mock_target = MagicMock()
        mock_target.confidence = -1
        self.processor.detector.find_best_target.return_value = mock_target

        input_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.processor._process_frame(input_frame)

        self.mock_commander.execute_command.assert_not_called()
        self.assertIs(result, input_frame)

    def test_process_frame_with_target_and_logging(self):
        """Tests the full pipeline from detection to command execution."""
        mock_target = MagicMock()
        mock_target.confidence = 0.8
        self.processor.detector.find_best_target.return_value = mock_target

        mock_cmd = MagicMock()
        mock_cmd.x_cmd, mock_cmd.y_cmd, mock_cmd.z_cmd, mock_cmd.rot_cmd = 0.1, 0.2, 0.3, 0.4

        mock_logs = {
            "jacobian_matrix": np.zeros((8, 6)).tolist(),
            "jcond": 10.0,
            "current_points_flatten": [0] * 8,
            "goal_points_flatten": [0] * 8,
            "err_uv": np.array([1.0, 1.0]),
            "velocity": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }
        self.processor.target_tracker.calculate_movement.return_value = (mock_cmd, mock_logs)

        with patch.object(pd.DataFrame, 'to_parquet'):
            self.processor.parquet_path = Path("./test_logs")
            self.processor.log_parquet = pd.DataFrame(columns=[
                "timestamp", "frame_idx", "x_cmd", "y_cmd", "z_cmd", "rot_cmd",
                "jacobian_matrix", "jcond", "current_points_flatten",
                "goal_points_flatten", "err_uv", "velocity"
            ])
            input_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            self.processor._process_frame(input_frame)

            self.mock_commander.execute_command.assert_called_once()
            self.assertEqual(len(self.processor.log_parquet), 1)
            self.assertEqual(self.processor.log_parquet.iloc[0]['jcond'], 10.0)

    def test_perform_movement_blocking_status(self):
        """Ensure piloting commands are always sent as non-blocking."""
        mock_cmd = MagicMock()
        mock_cmd.x_cmd = 0.5

        self.processor.perform_movement(mock_cmd)

        _, kwargs = self.mock_commander.execute_command.call_args
        self.assertFalse(kwargs['is_blocking'])
        self.assertIsInstance(kwargs['command'], PilotingCommand)

    def test_drone_state_validation(self):
        """Verify safety check returns False if drone is not flying or gimbal not tilted."""
        from drone_base.control.states import FlightState, GimbalOrientation

        # Scenario: Flying but Gimbal is Horizontal
        self.mock_commander.state.get_state.return_value = (FlightState.FLYING, GimbalOrientation.HORIZONTAL)
        self.assertFalse(self.processor._check_start_drone_state())

        # Scenario: Flying and Gimbal Tilted
        self.mock_commander.state.get_state.return_value = (FlightState.FLYING, GimbalOrientation.TILTED)
        self.assertTrue(self.processor._check_start_drone_state())


if __name__ == '__main__':
    unittest.main()