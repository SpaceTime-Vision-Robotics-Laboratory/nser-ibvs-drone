import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from auto_follow.detection.target_tracker import CommandInfo
from auto_follow.distiled_network.distil_engine import StudentEngine
from auto_follow.processors.ibvs_splitter_processor import IBVSSplitterProcessor
from auto_follow.utils.path_manager import Paths


class EvalDistilledNetworkProcessor(IBVSSplitterProcessor):
    """Basic video processor that only displays frames from the video stream."""

    def __init__(
            self,
            student_model_path: Path | str = Paths.SIM_STUDENT_NET_PATH,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            error_window_size: int = 5,
            **kwargs
    ):
        super().__init__(
            model_path=model_path,
            splitter_model_path=splitter_model_path,
            error_window_size=error_window_size,
            **kwargs
        )
        self.student_engine = StudentEngine(student_model_path)
        self.int_threshold = 0.5
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_command_info = None

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

        self._add_cmd_visualization(frame, self.last_command_info)
        if self._frame_count % 2 == 1:
            return frame

        # Student command: Real one
        command = self.student_engine.predict(np.array(frame))
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

        target_data = self.detector.find_best_target(np.array(frame), None)

        if target_data.confidence == -1:
            self._soft_goal_enter_time = None
            self.recent_errors.clear()
            return frame

        # self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)
        # Splitter Command: Note it is not used.
        splitter_command_info, logs = self.target_tracker.calculate_movement(target_data)
        self.recent_errors.append(self.ibvs_controller.err_uv_values[-1])

        self.recent_commands[:-1] = self.recent_commands[1:]
        self.recent_commands[-1] = np.array([drone_command.x_cmd, drone_command.y_cmd, drone_command.rot_cmd])
        self._save_parquet_logs_student(parquet_row, drone_command, splitter_command_info, logs)

        self.check_goal_reached(timestamp)
        self.check_timout_landing(timestamp)

        self.last_command_info = drone_command
        self.perform_movement(drone_command)

        return frame

    def _save_parquet_logs_student(
            self, parquet_row: dict, student_command_info: CommandInfo, splitter_command_info: CommandInfo, logs: dict
    ) -> None:
        parquet_row["x_cmd"] = student_command_info.x_cmd
        parquet_row["y_cmd"] = student_command_info.y_cmd
        parquet_row["z_cmd"] = student_command_info.z_cmd
        parquet_row["rot_cmd"] = student_command_info.rot_cmd

        parquet_row["splitter_x_cmd"] = splitter_command_info.x_cmd
        parquet_row["splitter_y_cmd"] = splitter_command_info.y_cmd
        parquet_row["splitter_z_cmd"] = splitter_command_info.z_cmd
        parquet_row["splitter_rot_cmd"] = splitter_command_info.rot_cmd
        parquet_row["jacobian_matrix"] = logs["jacobian_matrix"]
        parquet_row["jcond"] = logs["jcond"]
        parquet_row["current_points_flatten"] = logs["current_points_flatten"]
        parquet_row["goal_points_flatten"] = logs["goal_points_flatten"]
        parquet_row["err_uv"] = logs["err_uv"]
        parquet_row["velocity"] = logs["velocity"]

        self.log_parquet = pd.concat([self.log_parquet, pd.DataFrame([parquet_row])], ignore_index=True)
        self.log_parquet.to_parquet(self.parquet_path / "logs-student.parquet", index=False)

    def _add_cmd_visualization(self, frame: np.ndarray, drone_command: CommandInfo) -> None:
        if drone_command is None:
            return
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
