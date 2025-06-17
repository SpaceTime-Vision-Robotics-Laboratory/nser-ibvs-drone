import json
import time
from collections import deque
from pathlib import Path

import os
from typing import Callable, Any

import numpy as np
import pandas as pd

import glob

import cv2

from auto_follow.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from auto_follow.utils.path_manager import Paths

from auto_follow.detection.frame_visualizer import FrameVisualizerIBVS
from auto_follow.detection.target_tracker import CommandInfo, TargetTrackerIBVS
from auto_follow.detection.yolo_ibvs import YoloEngineIBVS
from auto_follow.ibvs.ibvs_controller import ImageBasedVisualServo
from auto_follow.utils.cam_params import infer_intrinsic_matrix

'''

TODO plots
- norm (y) vs initial (?) distance from pose d (x)
  ^ should be a horizontal line (final norm should not increase for a long distance)

- add failure cases for student // when you get it out of its distribution
  ^ in random position

'''

class IBVSSplitterAnnotate():
    def __init__(
            self,
            model_path: str | Path = Paths.REAL_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.REAL_MASK_SPLITTER_CAR,
            goal_frame_points_path: str | Path | None = Paths.GOAL_FRAME_POINTS_PATH_45,
            camera_params_path: str | Path | None = Paths.CAMERA_SIM_ANAFI_4k_DIR,
            logs_parquet_path: str | Path | None = Paths.LOG_PARQUET_DIR,
    ):
        self.detector = MaskSplitterEngineIBVS(model_path=model_path, splitter_model_path=splitter_model_path)

        print(f"{model_path=} | {splitter_model_path=}")

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
        self.target_tracker = TargetTrackerIBVS(video_config=None, ibvs_controller=self.ibvs_controller)

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
        self.log_parquet.to_parquet(self.parquet_path / "logs-teacher-output.parquet", index=False)

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        timestamp = time.perf_counter()

        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
        }

        target_data = self.detector.find_best_target(frame, None)

        if target_data.confidence == -1:
            return frame

        command_info, logs = self.target_tracker.calculate_movement(target_data)

        # self.logger.info("Command info: %s", command_info)
        # print(
        #     "Velocities IBVS: %s, Jcond: %.5f, Err norm: %.5f",
        #     logs["velocity"],
        #     logs["jcond"],
        #     np.linalg.norm(logs["err_uv"])
        # )

        self._save_parquet_logs(parquet_row, command_info, logs)

        return frame

def process_all_frames(base_directory: str) -> None:
    """
    Process all frames in the results directory structure.
    
    Args:
        base_directory: Path to the main directory containing parquet_logs and results
        frame_processor: Function to apply to each frame
    """
    base_path = Path(base_directory)

    logs_parquet_path = base_path / "parquet-logs"
    results_path = base_path / "results"
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    # Iterate through each folder in results
    for folder in list(results_path.iterdir()):
        if folder.is_dir():
            frames_path = folder / "frames"

            print("---------------------------------")
            print(f"{frames_path=}")

            logs_parquet_folder = logs_parquet_path / folder.name
            print(f"{logs_parquet_folder=}")
            if not logs_parquet_folder.is_dir():
                # the logs do not exist for this run
                continue

            print("\nGetting frame_processor")
            print("---------------------------------")

            frame_processor = IBVSSplitterAnnotate(logs_parquet_path=logs_parquet_folder)
            
            if frames_path.exists() and frames_path.is_dir():
                ## read the parquet
                ## ---------------------------------

                try:
                    df = pd.read_parquet(logs_parquet_folder / "logs.parquet")
                    
                    frame_idx_values = []
                    if 'frame_idx' not in df.columns:
                        print(f"Warning: 'frame_idx' column not found in {logs_parquet_folder}")
                        print(f"Available columns: {list(df.columns)}")
                    
                    frame_idx_values = df['frame_idx'].tolist()
                    
                except Exception as e:
                    print(f"Error reading parquet file {logs_parquet_folder}: {e}")

                ## ---------------------------------
                
                # Process each frame in the frames directory
                for frame_idx in frame_idx_values:
                    prefix = f"frame_{frame_idx:06d}"

                    pattern = os.path.join(frames_path, f"{prefix}*")
                    matching_files = list(glob.glob(pattern))
                    
                    frame_file = matching_files[0]

                    frame = cv2.imread(str(frame_file))
                    frame_processor._process_frame(frame, frame_idx)
            else:
                print(f"No frames directory found in: {folder.name}")

if __name__ == "__main__":
    paths_run = Path("/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/real-student-all")
    for folder in list(paths_run.iterdir()):
        if folder.is_dir():
            print(f"\n Running teacher for: {folder}")
            print("==========================================")

            process_all_frames(folder)
