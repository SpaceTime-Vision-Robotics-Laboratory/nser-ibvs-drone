import os
from pathlib import Path

import numpy as np
import pandas as pd

from nser_ibvs_drone.detection.frame_visualizer import FrameVisualizer
from nser_ibvs_drone.detection.target_tracker import TargetTracker, CommandInfo
from nser_ibvs_drone.detection.yolo_engine import YoloEngine
from nser_ibvs_drone.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class SimpleYoloProcessor(BaseVideoProcessor):
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

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(results)

        command_info = self.target_tracker.calculate_movement(
            object_center=target_data.center, box_size=target_data.size, target_lost=target_data.is_lost
        )
        self.perform_movement(command_info)

        return self.visualizer.draw_frame(frame=frame, target_data=target_data, moved_up=False)[0]

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

        if self.detector_log_dir is not None:
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


if __name__ == '__main__':
    from drone_base.stream.base_streaming_controller import BaseStreamingController

    controller = BaseStreamingController(
        ip=DroneIp.SIMULATED,
        processor_class=SimpleYoloProcessor,
    )
