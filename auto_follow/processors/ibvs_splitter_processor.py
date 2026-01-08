import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from auto_follow.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from auto_follow.detection.target_tracker import CommandInfo
from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from auto_follow.utils.path_manager import Paths


class IBVSSplitterProcessor(IBVSYoloProcessor):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            error_window_size: int = 5,
            **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.detector = MaskSplitterEngineIBVS(model_path=model_path, splitter_model_path=splitter_model_path)
        self.time_to_keep_in_frame = 3
        self.stop_error_hard_threshold = 40.0
        self.stop_error_soft_threshold = 80.0
        self.timeout_seconds = 75
        self._soft_goal_enter_time = None
        self._hard_goal_enter_time = None
        self._flight_start_time = None
        self._flight_end_time = None

        self.error_window_size = error_window_size
        self.recent_errors = deque(maxlen=self.error_window_size)
        self.results_path = self.frame_saver.output_dir.parent / "flight_duration.json"
        self.recent_commands = np.ones((self.error_window_size, 3))

        self.last_command_info = None
        self.last_target_data = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX


    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._check_start_drone_state():
            return frame

        timestamp = time.perf_counter()

        if self._flight_start_time is None:
            self._flight_start_time = timestamp
            self.logger.info("Flight started at: %s", self._flight_start_time)

        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": self._frame_count,
        }

        if self._frame_count % 2 == 1:
            if self.last_command_info is not None:
                self._add_cmd_visualization(frame, self.last_command_info)
            if self.last_target_data is not None:
                self.visualizer.display_frame(
                    frame, self.last_target_data, self.ibvs_controller, self.ibvs_controller.goal_points
                )
            return frame

        target_data = self.detector.find_best_target(frame, None)
        self.last_target_data = target_data

        if target_data.confidence == -1:
            self._soft_goal_enter_time = None
            self.recent_errors.clear()
            self.check_timout_landing(timestamp)
            return frame

        self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)

        command_info, logs = self.target_tracker.calculate_movement(target_data)

        self.logger.info("Command info: %s", command_info)
        self.logger.info(
            "Velocities IBVS: %s, Jcond: %.5f, Err norm: %.5f",
            logs["velocity"],
            logs["jcond"],
            np.linalg.norm(logs["err_uv"])
        )

        self.recent_commands[:-1] = self.recent_commands[1:]
        self.recent_commands[-1] = np.array([command_info.x_cmd, command_info.y_cmd, command_info.rot_cmd])

        if self.parquet_path is not None:
            self._save_parquet_logs(parquet_row, command_info, logs)
        self.recent_errors.append(self.ibvs_controller.err_uv_values[-1])

        self.check_goal_reached(timestamp)
        self.check_timout_landing(timestamp)

        self.last_command_info = command_info
        self.perform_movement(command_info)
        self._add_cmd_visualization(frame, command_info)

        return frame

    def check_goal_reached(self, timestamp: float):
        is_hard_error_reached, is_soft_error_reached = self._is_stable_at_goal()
        self.handle_hard_goal_reach(timestamp, is_hard_error_reached)
        self.handle_soft_goal_reach(timestamp, is_soft_error_reached)

    def handle_hard_goal_reach(self, timestamp: float, is_hard_error_reached: bool):
        if not is_hard_error_reached:
            self._hard_goal_enter_time = None
            return

        if self._hard_goal_enter_time is None:
            self._hard_goal_enter_time = timestamp
            self.logger.info("Goal enter time: %s", self._hard_goal_enter_time)
        elif (timestamp - self._hard_goal_enter_time) >= self.time_to_keep_in_frame:
            self.logger.info("Target has been in goal threshold for %s [s].", self.time_to_keep_in_frame)

            flight_duration = timestamp - self._flight_start_time
            self.logger.info("Flight ended at: %s", timestamp)
            self.logger.info("Total flight duration: %.5f [s]", flight_duration)

            with self.results_path.open("w") as results_file:
                json.dump({
                    "start_time": self._flight_start_time,
                    "end_time": timestamp,
                    "flight_duration": flight_duration,
                    "status": "complete"
                }, results_file, indent=4)

            self.drone_commander.land()

    def handle_soft_goal_reach(self, timestamp: float, is_soft_error_reached: bool):
        if not is_soft_error_reached:
            self._soft_goal_enter_time = None
            return

        if self._soft_goal_enter_time is None:
            self._soft_goal_enter_time = timestamp
            self.logger.info("Hard Goal enter time: %s", self._soft_goal_enter_time)
        elif (timestamp - self._soft_goal_enter_time) >= self.time_to_keep_in_frame:
            self.logger.info("Target has been in goal threshold (hard) for %s [s].", self.time_to_keep_in_frame)
            flight_duration = timestamp - self._flight_start_time
            self.logger.info("Flight ended at: %s", timestamp)
            self.logger.info("Total flight duration: %.5f [s]", flight_duration)

            with self.results_path.open("w") as results_file:
                json.dump({
                    "start_time": self._flight_start_time,
                    "end_time": timestamp,
                    "flight_duration": flight_duration,
                    "status": "complete-soft"
                }, results_file, indent=4)

            self.drone_commander.land()

    def check_timout_landing(self, timestamp: float):
        if not (self._flight_start_time is not None and (timestamp - self._flight_start_time) >= self.timeout_seconds):
            return

        if self._flight_end_time is None:
            self._flight_end_time = timestamp
            flight_duration = self._flight_end_time - self._flight_start_time

            self.logger.info("Timeout reached. Landing now.")
            self.logger.info("Flight ended at: %s", self._flight_end_time)
            self.logger.info("Total flight duration: %.5f [s]", flight_duration)

            with self.results_path.open("w") as results_file:
                json.dump({
                    "start_time": self._flight_start_time,
                    "end_time": self._flight_end_time,
                    "flight_duration": flight_duration,
                    "status": "timeout"
                }, results_file, indent=4)

            self.drone_commander.land()

    def _is_stable_at_goal(self) -> tuple[bool, bool]:
        """
        Check if at goal using median of recent errors for stability. Need at least 3 values for meaningful median.
        :returns: A tuple of (If goal reached, If reached within a threshold and if all commands are 0)
        """
        if len(self.recent_errors) < 3:  # noqa: PLR2004
            return False, False

        median_error = np.median(list(self.recent_errors))
        return (median_error < self.stop_error_hard_threshold,
                np.all(self.recent_commands == 0) and median_error < self.stop_error_soft_threshold)

    def _add_cmd_visualization(self, frame: np.ndarray, drone_command: CommandInfo) -> None:
        overlay = np.array(frame)
        cv2.rectangle(overlay, (5, 10), (105, 100), (0, 0, 0), -1)  # Black rectangle
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 40% overlay opacity
        cv2.putText(
            frame, f"X: {drone_command.x_cmd:+4d}", (10, 30), self.font, 0.7, (0, 0, 255), 2
        )
        cv2.putText(
            frame, f"Y: {drone_command.y_cmd:+4d}", (10, 60), self.font, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"R: {drone_command.rot_cmd:+4d}", (10, 90), self.font, 0.7, (255, 255, 0), 2
        )
