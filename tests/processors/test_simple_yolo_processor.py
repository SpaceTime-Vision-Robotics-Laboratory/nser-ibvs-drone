import queue
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from nser_ibvs_drone.processors.simple_yolo_processor import SimpleYoloProcessor
from drone_base.control.operations import PilotingCommand


class TestSimpleYoloProcessor(unittest.TestCase):

    def setUp(self):
        with patch('nser_ibvs_drone.processors.simple_yolo_processor.YoloEngine'), \
                patch('nser_ibvs_drone.processors.simple_yolo_processor.TargetTracker'), \
                patch('nser_ibvs_drone.processors.simple_yolo_processor.FrameVisualizer'), \
                patch('nser_ibvs_drone.processors.simple_yolo_processor.Paths') as mock_paths:
            mock_paths.SIM_CAR_YOLO_PATH = "dummy.pt"
            mock_paths.DETECTOR_LOG_DIR = None

            # Base class requirements
            self.mock_video_config = MagicMock()
            self.mock_video_config.save_extension = ".jpg"
            self.mock_queue = queue.Queue()
            self.mock_drone_commander = MagicMock()

            self.processor = SimpleYoloProcessor(
                model_path="dummy_path",
                detector_log_dir=None,
                video_config=self.mock_video_config,
                frame_queue=self.mock_queue,
                drone_commander=self.mock_drone_commander
            )
            self.processor.logger = MagicMock()

    def test_process_frame_flow(self):
        """Tests that a frame triggers detection, tracking, movement, and visualization."""
        mock_target = MagicMock()
        mock_target.center = (100, 100)
        mock_target.size = (50, 50)
        mock_target.is_lost = False
        self.processor.detector.find_best_target.return_value = mock_target

        mock_command_info = MagicMock()
        self.processor.target_tracker.calculate_movement.return_value = mock_command_info

        dummy_output_frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.processor.visualizer.draw_frame.return_value = (dummy_output_frame, None)

        input_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result_frame = self.processor._process_frame(input_frame)

        self.processor.detector.detect.assert_called_once()
        self.processor.target_tracker.calculate_movement.assert_called_once_with(
            object_center=(100, 100), box_size=(50, 50), target_lost=False
        )
        self.mock_drone_commander.execute_command.assert_called_once()
        np.testing.assert_array_equal(result_frame, dummy_output_frame)

    def test_perform_movement_sends_piloting_command(self):
        """Verifies that movement commands are correctly translated to PilotingCommand."""
        mock_cmd_info = MagicMock()
        mock_cmd_info.x_cmd = 0.1
        mock_cmd_info.y_cmd = -0.2
        mock_cmd_info.z_cmd = 0.3
        mock_cmd_info.rot_cmd = 0.5

        self.processor.perform_movement(mock_cmd_info)

        called_args = self.mock_drone_commander.execute_command.call_args
        command = called_args.kwargs['command']

        self.assertIsInstance(command, PilotingCommand)
        self.assertEqual(command.x, 0.1)
        self.assertEqual(command.rotation, 0.5)
        self.assertFalse(called_args.kwargs['is_blocking'])

    @patch('nser_ibvs_drone.processors.simple_yolo_processor.pd.DataFrame.to_csv')
    @patch('nser_ibvs_drone.processors.simple_yolo_processor.os.path.exists')
    def test_csv_logging(self, mock_exists, mock_to_csv):
        """Tests that the processor attempts to log to CSV when a log directory is provided."""
        self.processor.detector_log_dir = Path("test_log.csv")
        mock_exists.return_value = True

        mock_cmd_info = MagicMock()
        mock_cmd_info.timestamp = 1.0
        mock_cmd_info.status = "tracking"

        self.processor.perform_movement(mock_cmd_info)

        mock_to_csv.assert_called_once()
        self.assertEqual(mock_to_csv.call_args.kwargs['mode'], 'a')


if __name__ == '__main__':
    unittest.main()
