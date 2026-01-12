import queue
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nser_ibvs_drone.processors.state_processor import StateYoloProcessor, SearchState
from drone_base.control.operations import MovementByCommand
from drone_base.control.states import FlightState, GimbalOrientation


class TestStateYoloProcessor(unittest.TestCase):

    def setUp(self):
        with patch('nser_ibvs_drone.processors.state_processor.YoloEngine'), \
                patch('nser_ibvs_drone.processors.state_processor.TargetTracker'), \
                patch('nser_ibvs_drone.processors.state_processor.FrameVisualizer'), \
                patch('nser_ibvs_drone.processors.state_processor.Paths') as mock_paths:
            mock_paths.SIM_CAR_YOLO_PATH = "dummy_model.pt"
            mock_paths.DETECTOR_LOG_DIR = None

            self.mock_video_config = MagicMock()
            self.mock_video_config.save_extension = ".jpg"
            self.mock_queue = queue.Queue()
            self.mock_drone_commander = MagicMock()

            self.processor = StateYoloProcessor(
                model_path="dummy_path",
                detector_log_dir=None,
                video_config=self.mock_video_config,
                frame_queue=self.mock_queue,
                drone_commander=self.mock_drone_commander
            )

            self.processor.logger = MagicMock()

    def _set_drone_state(self, flight_state, gimbal_state):
        """Helper to mock the drone's telemetry via the actual drone_commander."""
        self.mock_drone_commander.state.get_state.return_value = (flight_state, gimbal_state)

    def test_initial_state(self):
        """Verify the processor starts in DO_NOTHING."""
        self.assertEqual(self.processor.current_state, SearchState.DO_NOTHING)

    def test_transition_to_follow_car(self):
        """Should transition from INITIAL to FOLLOW_CAR when a target is detected."""
        self._set_drone_state(FlightState.FLYING, GimbalOrientation.TILTED)
        self.processor.current_state = SearchState.INITIAL

        mock_target = MagicMock()
        mock_target.is_lost = False
        self.processor.detector.find_best_target.return_value = mock_target

        self.processor.target_tracker.calculate_movement.return_value = MagicMock()
        self.processor.visualizer.draw_frame.return_value = (np.zeros((10, 10)), None)

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.processor._process_frame(dummy_frame)

        self.assertEqual(self.processor.current_state, SearchState.FOLLOW_CAR)

    def test_target_lost_counter_logic(self):
        """Should transition to MOVE_UP only after MAX_IS_LOST_COUNT is reached."""
        self.processor.current_state = SearchState.FOLLOW_CAR
        self.processor._is_lost_count = 0

        for _ in range(self.processor.MAX_IS_LOST_COUNT - 1):
            self.processor._check_target_lost(is_lost=True)
            self.assertEqual(self.processor.current_state, SearchState.FOLLOW_CAR)

        self.processor._check_target_lost(is_lost=True)
        self.assertEqual(self.processor.current_state, SearchState.MOVE_UP)

    def test_drone_not_flying_resets_state(self):
        """If drone lands, state should reset to DO_NOTHING."""
        self.processor.current_state = SearchState.FOLLOW_CAR
        self._set_drone_state(FlightState.LANDED, GimbalOrientation.TILTED)

        result = self.processor._check_start_drone_state()

        self.assertFalse(result)
        self.assertEqual(self.processor.current_state, SearchState.DO_NOTHING)

    def test_perform_movement_commands(self):
        """Ensure correct MovementByCommand is sent in MOVE_UP state."""
        self.processor.current_state = SearchState.MOVE_UP
        self.processor.perform_movement(None)

        called_args = self.mock_drone_commander.execute_command.call_args[1]
        command = called_args['command']

        self.assertIsInstance(command, MovementByCommand)
        self.assertEqual(command.down, -0.5)


if __name__ == '__main__':
    unittest.main()
