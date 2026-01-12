import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from nser_ibvs_drone.detection.frame_visualizer import FrameVisualizer
from nser_ibvs_drone.detection.target_tracker import TargetTracker, CommandInfo
from nser_ibvs_drone.detection.targets import Target
from nser_ibvs_drone.detection.yolo_engine import YoloEngine
from nser_ibvs_drone.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand, MovementByCommand
from drone_base.control.states import FlightState, GimbalOrientation
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class SearchState(Enum):
    FOLLOW_CAR = 1
    MOVE_UP = 2
    FOLLOW_AFTER_UP = 3
    MOVE_DOWN = 4
    DO_NOTHING = 5
    INITIAL = 6


class StateYoloProcessor(BaseVideoProcessor):
    """
    State Transitions:

      INITIAL / DO_NOTHING
              ↓ (target detected)
         → FOLLOW_CAR
              ↓ (lost target)
         → MOVE_UP
              ↓ (target seen)
         → MOVE_DOWN
              ↓
         → FOLLOW_CAR

    Other conditions:
    - If not flying                    → DO_NOTHING
    - If gimbal not tilted             → INITIAL
    - After MOVE_UP/DOWN and no target → DO_NOTHING
    """

    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_YOLO_PATH,
            detector_log_dir: str | Path | None = Paths.DETECTOR_LOG_DIR,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.detector = YoloEngine(model_path)
        self.target_tracker = TargetTracker(video_config=self.config)
        self.visualizer = FrameVisualizer(video_config=self.config)
        self.detector_log_dir = detector_log_dir
        if self.detector_log_dir is not None:
            self.detector_log_dir = Path(self.detector_log_dir) / date_time_now_to_file_name()
            self.detector_log_dir.mkdir(parents=True, exist_ok=True)
            self.detector_log_dir /= "yolo_camera_log.csv"

        self.current_state = SearchState.DO_NOTHING
        self.move_done = False  # Track if MOVE_UP/DOWN has already executed
        self._is_lost_count = 0
        self.MAX_IS_LOST_COUNT = 45

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:  # noqa: PLR0911
        if self._frame_count % 15 == 0:
            self.logger.info("Current state: %s", self.current_state)
        if not self._check_start_drone_state():
            return frame

        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(results)
        if self.current_state in (SearchState.INITIAL, SearchState.DO_NOTHING):
            return self._handle_initial(frame, target_data)

        if self.current_state == SearchState.FOLLOW_CAR:
            has_transitioned = self._check_target_lost(target_data.is_lost)
            if has_transitioned:
                return frame
            return self._follow_car(frame, target_data)
        elif self.current_state == SearchState.MOVE_UP:
            if not self.move_done:
                self.perform_movement(None)
                self.move_done = True
                return frame
            if not target_data.is_lost:
                self.current_state = SearchState.FOLLOW_AFTER_UP
                self.move_done = False
                return frame
            return frame
        elif self.current_state == SearchState.FOLLOW_AFTER_UP:
            return self._handle_follow_after_up(frame, target_data)
        elif self.current_state == SearchState.MOVE_DOWN:
            if not self.move_done:
                self.perform_movement(None)
                self.move_done = True
            else:
                self.current_state = SearchState.FOLLOW_CAR
                self.move_done = False
        return frame

    def _check_start_drone_state(self) -> bool:
        """
        Validates that the drone is ready to process commands and sets states.

        If the drone is flying and has tilted camera then it can accept commands and will return true. False otherwise.
        """
        fly_state, gimbal_state = self.drone_commander.state.get_state()
        if fly_state != FlightState.FLYING:
            self.current_state = SearchState.DO_NOTHING
            return False

        if gimbal_state != GimbalOrientation.TILTED:
            self.reset_to_initial()
            return False

        return True

    def _handle_initial(self, frame: np.ndarray, target_data: Target) -> np.ndarray:
        """
        Wait until we see the car in the frame. Manually guide the car or the drone.
        """
        if not target_data.is_lost:
            self.current_state = SearchState.FOLLOW_CAR
            return self._follow_car(frame, target_data)
        return frame

    def _check_target_lost(self, is_lost: bool) -> bool:
        """
        Checks if the target is not seen in the frame for MAX_IS_LOST_COUNT consecutive times.
        If it is not seed for that many frames it will then consider the target lost and move to the MOVE UP procedure.

        :param is_lost: Is the target seen in frame?
        :return: True if the state transitioned False otherwise.
        """
        if is_lost:
            self._is_lost_count += 1
            if self._is_lost_count >= self.MAX_IS_LOST_COUNT:
                self.current_state = SearchState.MOVE_UP
                self.move_done = False
                return True
        else:
            self._is_lost_count = 0
        return False

    def _follow_car(self, frame: np.ndarray, target_data: Target) -> np.ndarray:
        """
        Will infer the command to take from the target data.
        Will log, and perform the follow command together with its drawing.
        """
        command_info = self.target_tracker.calculate_movement(
            object_center=target_data.center, box_size=target_data.size, target_lost=target_data.is_lost
        )
        self.log_command(command_info)
        self.perform_movement(command_info)
        return self.visualizer.draw_frame(frame=frame, target_data=target_data, moved_up=False)[0]

    def _handle_follow_after_up(self, frame: np.ndarray, target_data: Target) -> np.ndarray:
        if target_data.is_lost:
            return frame  # Wait until we see the target
        command_info = self.target_tracker.calculate_movement(
            object_center=target_data.center, box_size=target_data.size, target_lost=target_data.is_lost
        )

        # If its centered enough:
        if command_info.y_cmd == 0.0 and command_info.rot_cmd < 0.0:
            self.current_state = SearchState.MOVE_DOWN
            self.move_done = False
            return frame

        self.log_command(command_info)
        self.perform_movement(command_info)
        return self.visualizer.draw_frame(frame=frame, target_data=target_data, moved_up=False)[0]

    def perform_movement(self, command_info: CommandInfo | None) -> None:
        if self.current_state == SearchState.FOLLOW_CAR:
            self.drone_commander.execute_command(
                command=PilotingCommand(
                    x=command_info.x_cmd,
                    y=command_info.y_cmd,
                    z=command_info.z_cmd,
                    rotation=command_info.rot_cmd,
                    dt=0.15
                ),
                is_blocking=False
            )
        elif self.current_state == SearchState.MOVE_UP:
            self.drone_commander.execute_command(command=MovementByCommand(down=-0.5), is_blocking=False)
        elif self.current_state == SearchState.FOLLOW_AFTER_UP:
            self.drone_commander.execute_command(
                command=PilotingCommand(
                    x=command_info.x_cmd,
                    y=command_info.y_cmd,
                    z=command_info.z_cmd,
                    rotation=command_info.rot_cmd,
                    dt=0.15
                ),
                is_blocking=False
            )
        elif self.current_state == SearchState.MOVE_DOWN:
            self.drone_commander.execute_command(command=MovementByCommand(down=0.5), is_blocking=False)
        elif self.current_state == SearchState.DO_NOTHING:
            return

    def log_command(self, command_info: CommandInfo) -> None:
        if self.detector_log_dir is None:
            return
        new_data = pd.DataFrame([{
            'timestamp': command_info.timestamp,
            'x_cmd': command_info.x_cmd,
            'y_cmd': command_info.y_cmd,
            'z_cmd': command_info.z_cmd,
            'rot_cmd': command_info.rot_cmd,
            'x_offset': command_info.x_offset,
            'y_offset': command_info.y_offset,
            'p_rot': command_info.p_rot,
            'd_rot': command_info.d_rot,
            'status': command_info.status
        }])
        new_data.to_csv(
            self.detector_log_dir, mode='a', header=not os.path.exists(self.detector_log_dir), index=False
        )

    def reset_to_initial(self):
        self.current_state = SearchState.INITIAL
        self.move_done = False


if __name__ == '__main__':
    from drone_base.stream.base_streaming_controller import BaseStreamingController

    controller = BaseStreamingController(
        ip=DroneIp.SIMULATED,
        processor_class=StateYoloProcessor,
    )
