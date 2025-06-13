import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auto_follow.detection.frame_visualizer import FrameVisualizerIBVS
from auto_follow.detection.target_tracker import CommandInfo, TargetTrackerIBVS
from auto_follow.detection.yolo_ibvs import YoloEngineIBVS
from auto_follow.ibvs.ibvs_controller import ImageBasedVisualServo
from auto_follow.utils.cam_params import infer_intrinsic_matrix
from auto_follow.utils.path_manager import Paths
from drone_base.control.operations import PilotingCommand
from drone_base.control.states import FlightState, GimbalOrientation
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class IBVSYoloProcessor(BaseVideoProcessor):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            detector_log_dir: str | Path | None = Paths.DETECTOR_LOG_DIR,
            goal_frame_points_path: str | Path | None = Paths.GOAL_FRAME_POINTS_PATH_45,
            camera_params_path: str | Path | None = Paths.CAMERA_SIM_ANAFI_4k_DIR,
            logs_parquet_path: str | Path | None = Paths.LOG_PARQUET_DIR,
            **kwargs
    ):
        super().__init__(**kwargs)

        camera_params = infer_intrinsic_matrix(camera_params_path)
        if isinstance(camera_params, tuple):
            _, self.K, self.K_rec = camera_params
        else:
            self.K = camera_params

        with open(goal_frame_points_path, "r") as f:
            goal_points = json.load(f)
        goal_points_bbox = goal_points["bbox_oriented_points"]
        goal_points_bbox = goal_points_bbox[:4]

        self.ibvs_controller = ImageBasedVisualServo(self.K, goal_points_bbox)
        self.detector = YoloEngineIBVS(model_path)
        self.target_tracker = TargetTrackerIBVS(self.config, self.ibvs_controller)
        self.visualizer = FrameVisualizerIBVS(self.config)

        self.detector_log_dir = detector_log_dir
        if self.detector_log_dir is not None:
            self.detector_log_dir = Path(self.detector_log_dir) / date_time_now_to_file_name()

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
                "jacobian_matrix",
                "jcond",
                "current_points_flatten",
                "goal_points_flatten",
                "err_uv",
                "velocity"
            ])

    def _save_parquet_logs(self, parquet_row: dict, command_info: CommandInfo, logs: dict) -> None:
        parquet_row["x_cmd"] = command_info.x_cmd
        parquet_row["y_cmd"] = command_info.y_cmd
        parquet_row["z_cmd"] = command_info.z_cmd
        parquet_row["rot_cmd"] = command_info.rot_cmd
        parquet_row["jacobian_matrix"] = logs["jacobian_matrix"]
        parquet_row["jcond"] = logs["jcond"]
        parquet_row["current_points_flatten"] = logs["current_points_flatten"]
        parquet_row["goal_points_flatten"] = logs["goal_points_flatten"]
        parquet_row["err_uv"] = logs["err_uv"]
        parquet_row["velocity"] = logs["velocity"]

        self.log_parquet = pd.concat([self.log_parquet, pd.DataFrame([parquet_row])], ignore_index=True)
        self.log_parquet.to_parquet(self.parquet_path / "logs.parquet", index=False)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        timestamp = time.perf_counter()

        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": self._frame_count,
        }

        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(frame, results)

        if target_data.confidence == -1:
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

        self._save_parquet_logs(parquet_row, command_info, logs)

        self.perform_movement(command_info)

        return frame

    def perform_movement(self, command_info: CommandInfo) -> None:
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

    def _check_start_drone_state(self) -> bool:
        """
        Validates that the drone is ready to process commands and sets states.

        If the drone is flying and has tilted camera then it can accept commands and will return true. False otherwise.
        """
        fly_state, gimbal_state = self.drone_commander.state.get_state()
        if fly_state != FlightState.FLYING:
            return False

        if gimbal_state != GimbalOrientation.TILTED:
            return False

        return True
