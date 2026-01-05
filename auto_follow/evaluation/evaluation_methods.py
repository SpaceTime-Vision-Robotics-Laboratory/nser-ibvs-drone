import json
import time
from pathlib import Path

import numpy as np

from auto_follow.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from auto_follow.detection.target_tracker import TargetTrackerIBVS, CommandInfo
from auto_follow.detection.yolo_ibvs import YoloEngineIBVS
from auto_follow.distiled_network.distil_engine import StudentEngine
from auto_follow.ibvs.ibvs_controller import ImageBasedVisualServo
from auto_follow.utils.cam_params import infer_intrinsic_matrix
from auto_follow.utils.path_manager import Paths
from drone_base.config.video import VideoConfig


class StudentEvaluator:
    def __init__(
            self,
            student_model_path: str | Path = Paths.SIM_STUDENT_NEW_PATH_REAL_WORLD_DISTRIBUTION,
            segmentation_model_path: str | Path | None = None,
            device: str | None = None
    ):
        self.student_model_path = Path(student_model_path)
        if segmentation_model_path is not None:
            self.segmentation_model_path = Path(segmentation_model_path)
            self.detector = YoloEngineIBVS(self.segmentation_model_path)

        self.student_engine = StudentEngine(student_model_path, device=device)
        self.int_threshold = 0.5

    def predict_command_on_frame(self, frame: np.ndarray) -> tuple[int, int, int] | CommandInfo:
        command = self.student_engine.predict(frame)
        command = np.where(
            np.abs(command - np.floor(command)) > self.int_threshold,
            np.ceil(command),
            np.floor(command)
        )
        return int(command[0]), int(command[1]), int(command[2])

    def predict_with_segmentation(self, frame: np.ndarray) -> tuple[int, int, int] | CommandInfo:
        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(frame, results)
        if target_data.confidence == -1:
            return 0, 0, 0
        command = self.student_engine.predict(frame)
        command = np.where(
            np.abs(command - np.floor(command)) > self.int_threshold,
            np.ceil(command),
            np.floor(command)
        )

        return CommandInfo(
            timestamp=time.time(),
            x_cmd=int(command[0]),
            y_cmd=int(command[1]),
            z_cmd=0,
            rot_cmd=int(command[2]),
            x_offset=0,
            y_offset=0,
            p_rot=0,
            d_rot=0,
            status="Student"
        )


class IBVSEvaluator:
    def __init__(
            self,
            video_config: VideoConfig = VideoConfig(cam_mode="recording"),
            segmentation_model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            goal_frame_points_path: str | Path | None = Paths.GOAL_FRAME_POINTS_PATH_45,
            camera_params_path: str | Path | None = Paths.CAMERA_SIM_ANAFI_4k_DIR,
    ):
        camera_params = infer_intrinsic_matrix(camera_params_path)
        if isinstance(camera_params, tuple):
            _, self.K = camera_params
        else:
            self.K = camera_params

        with open(goal_frame_points_path, "r") as f:
            goal_points = json.load(f)
        goal_points_bbox = goal_points["bbox_oriented_points"]
        goal_points_bbox = goal_points_bbox[:4]

        self.ibvs_controller = ImageBasedVisualServo(self.K, goal_points_bbox)
        self.detector = MaskSplitterEngineIBVS(model_path=segmentation_model_path,
                                               splitter_model_path=splitter_model_path)
        self.target_tracker = TargetTrackerIBVS(video_config, self.ibvs_controller)

    def predict_command_on_frame(self, frame: np.ndarray) -> tuple[int, int, int] | CommandInfo:
        target_data = self.detector.find_best_target(frame, None)
        if target_data.confidence == -1:
            return 0, 0, 0

        command_info, logs = self.target_tracker.calculate_movement(target_data)
        return command_info
