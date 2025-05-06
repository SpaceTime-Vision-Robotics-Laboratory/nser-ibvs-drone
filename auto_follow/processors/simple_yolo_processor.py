import os
from pathlib import Path

import numpy as np
import pandas as pd

from auto_follow.detection.frame_visualizer import FrameVisualizer
from auto_follow.detection.target_tracker import TargetTracker, CommandInfo
from auto_follow.detection.yolo_engine import YoloEngine
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class SimpleYoloProcessor(BaseVideoProcessor):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_YOLO_SEG_PATH,
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
        results = self.detector.detect_stream(frame)
        target_data = self.detector.find_best_target(results)
        
        # --- Calculate 85% Box for Logic ---
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = self.config.frame_center_x
        frame_center_y = self.config.frame_center_y
        scale_factor = 0.85 ** 0.5
        box_width_85 = int(frame_width * scale_factor)
        box_height_85 = int(frame_height * scale_factor)
        x1_85 = int(frame_center_x - box_width_85 / 2)
        y1_85 = int(frame_center_y - box_height_85 / 2)
        x2_85 = int(frame_center_x + box_width_85 / 2)
        y2_85 = int(frame_center_y + box_height_85 / 2)
        # --- End Box Calculation ---

        self.visualizer.display_segmentation(frame=frame, target_data=target_data, moved_up=False)

        # --- Calculate Movement Command ---
        base_command_info = self.target_tracker.calculate_movement(
            target_data=target_data
        )

        final_command_info = base_command_info
        # If target is detected and outside the 85% box, prioritize centering
        if not target_data.is_lost and target_data.center is not None:
            target_x, target_y = target_data.center
            is_outside_box = not (x1_85 <= target_x <= x2_85 and y1_85 <= target_y <= y2_85)
            
            if is_outside_box:
                # Override forward/backward and sideways movement to prioritize rotation/altitude for centering
                final_command_info = CommandInfo(
                    timestamp=base_command_info.timestamp,
                    x_cmd=0,  # Disable sideways strafing
                    y_cmd=0,  # Disable forward/backward movement
                    z_cmd=base_command_info.z_cmd, # Keep altitude adjustment (if enabled in tracker)
                    rot_cmd=base_command_info.rot_cmd, # Keep rotation command
                    x_offset=base_command_info.x_offset,
                    y_offset=base_command_info.y_offset,
                    p_rot=base_command_info.p_rot,
                    d_rot=base_command_info.d_rot,
                    status="Centering" # Update status
                )
        # --- End Movement Calculation ---
        
        self.perform_movement(final_command_info)

        # self.visualizer.display_frame(frame=frame, target_data=target_data, moved_up=False)

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
                'angle_error': command_info.angle_error,
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
    
    import time
    controller.drone_commander.tilt_camera(pitch_deg=-90)
