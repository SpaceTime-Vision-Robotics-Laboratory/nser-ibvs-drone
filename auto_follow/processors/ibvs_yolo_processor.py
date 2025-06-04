import pickle
import json
from pathlib import Path

import numpy as np

from auto_follow.detection.frame_visualizer import FrameVisualizerIBVS
from auto_follow.detection.target_tracker import CommandInfo, TargetTrackerIBVS
from auto_follow.detection.yolo_engine import YoloEngineIBVS, YoloEngineIBVSPose
from auto_follow.utils.path_manager import Paths
from auto_follow.controllers.ibvs_controller import ImageBasedVisualServo

from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class IBVSYoloProcessor(BaseVideoProcessor):
    def __init__(
        self,
        model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
        detector_log_dir: str | Path | None = Paths.DETECTOR_LOG_DIR,
        goal_frame_points_pth: str | Path | None = Paths.GOAL_FRAME_POINTS_PATH_45,
        camera_params_pth: str | Path | None = Paths.CAMERA_PARAMS_HALF_SIZE_PATH,
        **kwargs
    ):
        super().__init__(**kwargs)

        with open(camera_params_pth, 'rb') as f:
            camera_params = pickle.load(f)
        self.K = camera_params

        with open(goal_frame_points_pth, "r") as f:
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

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(frame, results)

        if (target_data.confidence == -1):
            return frame

        self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)

        command_info = self.target_tracker.calculate_movement(target_data)
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

class IBVSPoseYoloProcessor(IBVSYoloProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = YoloEngineIBVSPose(model_path_pose=Paths.SIM_CAR_POSE_IBVS_YOLO_PATH)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(frame, results)

        if (target_data.confidence == -1):
            return frame

        self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)

        return frame
