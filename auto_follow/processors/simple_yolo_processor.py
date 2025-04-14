from pathlib import Path

import numpy as np

from auto_follow.detection.frame_visualizer import FrameVisualizer
from auto_follow.detection.target_tracker import TargetTracker, CommandInfo
from auto_follow.detection.yolo_engine import YoloEngine
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor


class SimpleYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_YOLO_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = YoloEngine(model_path)
        self.target_tracker = TargetTracker(video_config=self.config)
        self.visualizer = FrameVisualizer(video_config=self.config)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.detect(frame)
        target_data = self.detector.find_best_target(results)

        command_info = self.target_tracker.calculate_movement(
            object_center=target_data.center, box_size=target_data.size, target_lost=target_data.is_lost
        )
        self.perform_movement(command_info)

        self.visualizer.display_frame(frame=frame, target_data=target_data, moved_up=False)

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


if __name__ == '__main__':
    from drone_base.stream.base_streaming_controller import BaseStreamingController

    controller = BaseStreamingController(
        ip=DroneIp.SIMULATED,
        processor_class=SimpleYoloProcessor,
    )
