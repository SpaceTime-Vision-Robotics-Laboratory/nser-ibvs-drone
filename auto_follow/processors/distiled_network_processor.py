import numpy as np
import torch
from collections import deque
from PIL import Image
import torchvision.transforms as T
import time
from pathlib import Path
import pandas as pd
import cv2

from auto_follow.detection.target_tracker import TargetTracker, CommandInfo
from auto_follow.detection.yolo_engine import YoloEngine
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor
from auto_follow.utils.path_manager import Paths
from auto_follow.distiled_network.distil_engine import StudentEngine


class DistiledNetworkProcessor(BaseVideoProcessor):
    """Basic video processor that only displays frames from the video stream."""
    def __init__(self, model_path: Path | str = Paths.SIM_STUDENT_NET_PATH, 
                 logs_parquet_path: str | Path | None = Paths.LOG_PARQUET_DIR,
                 **kwargs):
        super().__init__(**kwargs)
        # self.segmentation_model = 
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
                "err_uv",
            ])
    
    def _save_parquet_logs(self, parquet_row: dict, command_info: CommandInfo, logs: dict) -> None:
        parquet_row["x_cmd"] = command_info.x_cmd
        parquet_row["y_cmd"] = command_info.y_cmd
        parquet_row["z_cmd"] = command_info.z_cmd
        parquet_row["rot_cmd"] = command_info.rot_cmd
        parquet_row["err_uv"] = logs["err_uv"]

        self.log_parquet = pd.concat([self.log_parquet, pd.DataFrame([parquet_row])], ignore_index=True)
        self.log_parquet.to_parquet(self.parquet_path / "logs.parquet", index=False)
        

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        timestamp = time.perf_counter()

        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": self._frame_count,
        }
        
        command = self.student_engine.predict(frame)
        
        command = np.where(
            np.abs(command - np.floor(command)) > self.int_threshold, 
            np.ceil(command), 
            np.floor(command)
        )
        command = command.astype(int)
        
        drone_command = CommandInfo(
            x_cmd=command[0],
            y_cmd=command[1],
            z_cmd=0,
            rot_cmd=command[2],
            timestamp=time.time(),
            x_offset=0,
            y_offset=0,
            p_rot=0,
            d_rot=0,
            status="StudentNet"
        )
        
        self._save_parquet_logs(parquet_row, drone_command, {})
        
        self.perform_movement(drone_command)
        cv2.putText(frame, f"x_cmd: {drone_command.x_cmd}\n y_cmd: {drone_command.y_cmd}\n rot_cmd: {drone_command.rot_cmd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
