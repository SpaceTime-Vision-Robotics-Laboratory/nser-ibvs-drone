import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from nser_ibvs_drone.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from nser_ibvs_drone.detection.target_tracker import CommandInfo, TargetTrackerIBVS
from nser_ibvs_drone.ibvs.ibvs_controller import ImageBasedVisualServo
from nser_ibvs_drone.utils.cam_params import infer_intrinsic_matrix
from nser_ibvs_drone.utils.path_manager import Paths
from drone_base.config.video import VideoConfig


class IBVSSplitterAnnotate:
    """NSER-IBVS class to generate IBVS logs for a scene (used to compare student network)."""
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            goal_frame_points_path: str | Path | None = Paths.GOAL_FRAME_POINTS_PATH_45,
            camera_params_path: str | Path | None = Paths.CAMERA_SIM_ANAFI_4k_DIR,
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
        self.target_tracker = TargetTrackerIBVS(video_config=VideoConfig(), ibvs_controller=self.ibvs_controller)
        self.log_parquet = []

    def _save_parquet_logs(self, parquet_row: dict, command_info: CommandInfo, logs: dict) -> None:
        row_data = {
            "timestamp": parquet_row["timestamp"],
            "frame_idx": parquet_row["frame_idx"],
            "x_cmd": command_info.x_cmd,
            "y_cmd": command_info.y_cmd,
            "z_cmd": command_info.z_cmd,
            "rot_cmd": command_info.rot_cmd,
            "jacobian_matrix": logs["jacobian_matrix"],
            "jcond": logs["jcond"],
            "current_points_flatten": logs["current_points_flatten"],
            "goal_points_flatten": logs["goal_points_flatten"],
            "err_uv": logs["err_uv"],
            "velocity": logs["velocity"],
        }
        self.log_parquet.append(row_data)

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> np.ndarray:
        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
        }

        target_data = self.detector.find_best_target(frame, None)
        if target_data.confidence == -1:
            return frame

        command_info, logs = self.target_tracker.calculate_movement(target_data)
        self._save_parquet_logs(parquet_row, command_info, logs)

        return frame


def process_all_frames_on_scene(base_directory: str | Path) -> None:
    """
    Process all frames in the results directory structure.

    :param base_directory: Path to the main directory containing parquet_logs and results directories.
    """
    base_directory = Path(base_directory)
    logs_parquet_path = base_directory / "parquet-logs"
    results_path = base_directory / "results"

    if not results_path.exists() or not logs_parquet_path.exists():
        print(f"Results or parquet directories not found: {base_directory}")
        return

    for run_directory in list(results_path.iterdir()):
        if not run_directory.is_dir():
            continue

        frames_path = run_directory / "frames"
        logs_parquet_folder = logs_parquet_path / run_directory.name

        print("---------------------------------")
        print(f"Processing {run_directory.name} at:\n\t- {frames_path}\n\t- {logs_parquet_folder}")
        if not logs_parquet_folder.is_dir():
            # the logs do not exist for this run
            continue

        parquet_file = logs_parquet_folder / "logs.parquet"
        if not parquet_file.exists():
            continue

        df = pd.read_parquet(parquet_file)
        frame_idx_values = df['frame_idx'].tolist()
        timestamp_values = df['timestamp'].tolist()

        frame_files_map = {int(p.stem.split("_")[1]): p for p in frames_path.glob("frame_*.jpg")}
        if not frame_files_map:
            print(f"Empty frames directory: {frames_path}... skipping")
            continue

        frame_processor = IBVSSplitterAnnotate()

        for frame_idx, timestamp in tqdm(
                zip(frame_idx_values, timestamp_values),
                desc=f"Processing {run_directory.name}",
                total=len(frame_idx_values)
        ):
            frame_file = frame_files_map.get(frame_idx)
            if frame_file is None:
                continue

            frame = cv2.imread(str(frame_file), cv2.IMREAD_UNCHANGED)
            frame_processor.process_frame(frame, frame_idx, timestamp)

        if len(frame_processor.log_parquet) != 0:
            pd.DataFrame(frame_processor.log_parquet).to_parquet(
                logs_parquet_folder / "logs-teacher-output.parquet", index=False
            )


def run_scene(scene_directory: Path):
    if scene_directory.is_dir():
        print(f"\nRunning teacher for: {scene_directory}")
        print("==========================================")
        process_all_frames_on_scene(scene_directory)


if __name__ == "__main__":
    paths_run = Path("/home/brittle/Desktop/work/space-time-vision-repos/nser-ibvs-drone/output")
    scene_dirs = [p for p in paths_run.iterdir() if p.is_dir()]

    s_time = time.time()
    for scene_dir in scene_dirs:
        run_scene(scene_dir)
    e_time = time.time()

    print(f"Total time: {e_time - s_time:.2f} seconds")
