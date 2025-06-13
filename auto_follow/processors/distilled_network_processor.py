import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from auto_follow.detection.target_tracker import CommandInfo
from auto_follow.distiled_network.distil_engine import StudentEngine
from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from auto_follow.utils.path_manager import Paths


class DistilledNetworkProcessor(IBVSYoloProcessor):
    """Basic video processor that only displays frames from the video stream."""

    def __init__(
            self,
            model_path: str | Path = Paths.SIM_STUDENT_NET_PATH,
            logs_parquet_path: str | Path | None = Paths.LOG_PARQUET_DIR,
            error_window_size: int = 5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.student_engine = StudentEngine(model_path)
        self.int_threshold = 0.5

        self.parquet_path = logs_parquet_path
        if self.parquet_path is not None:
            self.parquet_path = Path(self.parquet_path)
            self.parquet_path.mkdir(parents=True, exist_ok=True)
            self.log_parquet = pd.DataFrame(columns=[
                "timestamp",
                "frame_idx",
                "x_cmd",
                "y_cmd",
                "z_cmd",
                "rot_cmd",
            ])
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.time_to_keep_in_frame = 3
        self.timeout_seconds = 75
        self._flight_start_time = None
        self._flight_end_time = None
        self._command_zero_time = None

        self.error_window_size = error_window_size
        self.results_path = self.frame_saver.output_dir.parent / "flight_duration.json"
        self.recent_commands = np.ones((self.error_window_size, 3))

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

        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(frame, results)
        if target_data.confidence == -1:
            self._command_zero_time = None
            
            self.check_timout_landing(timestamp)
            
            return frame

        command = self.student_engine.predict(frame)
        command = np.where(
            np.abs(command - np.floor(command)) > self.int_threshold,
            np.ceil(command),
            np.floor(command)
        )

        drone_command = CommandInfo(
            x_cmd=int(command[0]),
            y_cmd=int(command[1]),
            z_cmd=0,
            rot_cmd=int(command[2]),
            timestamp=time.time(),
            x_offset=0,
            y_offset=0,
            p_rot=0,
            d_rot=0,
            status="StudentNet"
        )

        self.recent_commands[:-1] = self.recent_commands[1:]
        self.recent_commands[-1] = np.array([drone_command.x_cmd, drone_command.y_cmd, drone_command.rot_cmd])

        self._save_parquet_logs(parquet_row, drone_command, {})

        self.check_goal_reached(timestamp)
        self.check_timout_landing(timestamp)

        self.perform_movement(drone_command)
        self._add_cmd_visualization(frame, drone_command)

        return frame

    def check_goal_reached(self, timestamp: float):
        if not self._is_stable_at_goal():
            self._command_zero_time = None
            return

        if self._command_zero_time is None:
            self._command_zero_time = timestamp
            self.logger.info("Goal enter time: %s", self._command_zero_time)
        elif (timestamp - self._command_zero_time) >= self.time_to_keep_in_frame:
            self.logger.info("Target has been in goal threshold (hard) for %s [s].", self.time_to_keep_in_frame)
            flight_duration = timestamp - self._flight_start_time
            self.logger.info("Flight ended at: %s", timestamp)
            self.logger.info("Total flight duration: %.5f [s]", flight_duration)

            with self.results_path.open("w") as results_file:
                json.dump({
                    "start_time": self._flight_start_time,
                    "end_time": timestamp,
                    "flight_duration": flight_duration,
                    "status": "complete-goal"
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

    def _is_stable_at_goal(self) -> bool:
        """
        Check if at goal using median of recent errors for stability. Need at least 3 values for meaningful median.
        :returns: A tuple of (If goal reached, If reached within a threshold and if all commands are 0)
        """
        return np.all(self.recent_commands == 0)

    def _save_parquet_logs(self, parquet_row: dict, command_info: CommandInfo, logs: dict) -> None:
        parquet_row["x_cmd"] = command_info.x_cmd
        parquet_row["y_cmd"] = command_info.y_cmd
        parquet_row["z_cmd"] = command_info.z_cmd
        parquet_row["rot_cmd"] = command_info.rot_cmd

        self.log_parquet = pd.concat([self.log_parquet, pd.DataFrame([parquet_row])], ignore_index=True)
        self.log_parquet.to_parquet(self.parquet_path / "logs.parquet", index=False)

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
