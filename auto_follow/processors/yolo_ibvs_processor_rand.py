import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import time

from auto_follow.detection.frame_visualizer import FrameVisualizer
from auto_follow.detection.target_tracker import TargetTracker, CommandInfo
from auto_follow.detection.yolo_engine import YoloEngine
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.control.operations import PilotingCommand
from drone_base.stream.base_video_processor import BaseVideoProcessor
from drone_base.utils.readable_time import date_time_now_to_file_name


class YoloIbvsAngleProcessor(BaseVideoProcessor):
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

        # --- Draw Desired Points ---
        desired_center = np.array([frame_center_x, frame_center_y])
        desired_major_length = 100  # Desired length of major axis in pixels
        desired_minor_length = 50   # Desired length of minor axis in pixels
        desired_vertical_distance = 50  # Distance for vertical points
        
        # Calculate desired points (3 points + ellipse)
        desired_points = np.array([
            desired_center,  # Center point
            desired_center + np.array([0, -desired_vertical_distance]),  # Point above
            desired_center + np.array([0, desired_vertical_distance]),   # Point below
        ]).astype(np.int32)
        
        # Draw desired points and lines
        for i, point in enumerate(desired_points):
            # Draw points
            cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Red points
            
            # Draw lines between points
            if i == 0:  # Center point
                # Draw lines from center to vertical points
                cv2.line(frame, tuple(desired_points[0]), tuple(desired_points[1]), (0, 255, 0), 1)  # Green line
                cv2.line(frame, tuple(desired_points[0]), tuple(desired_points[2]), (0, 255, 0), 1)  # Green line
        
        # Draw desired ellipse
        cv2.ellipse(frame, 
                    center=tuple(desired_center),
                    axes=(int(desired_major_length/2), int(desired_minor_length/2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 255, 255),  # Yellow
                    thickness=1)
        # --- End Draw Desired Points ---

        self.visualizer.display_segmentation(frame=frame, target_data=target_data, moved_up=False)

        # --- Calculate Movement Command ---
        base_command_info = self.target_tracker.calculate_movement_segmentation(
            target_data=target_data
        )

        # Initialize logging data
        log_data = {
            'timestamp': time.time(),
            'frame_center_x': frame_center_x,
            'frame_center_y': frame_center_y,
            'target_lost': target_data.is_lost,
            'target_confidence': target_data.confidence if not target_data.is_lost else None,
            'target_center_x': target_data.center[0] if not target_data.is_lost else None,
            'target_center_y': target_data.center[1] if not target_data.is_lost else None,
            'target_angle': target_data.ellipse_angle if not target_data.is_lost else None,
            'target_major_axis': target_data.ellipse_axes[1] if not target_data.is_lost else None,
            'target_minor_axis': target_data.ellipse_axes[0] if not target_data.is_lost else None,
            'is_inside_box': False,  # Will be updated if target is found
            'x_cmd': 0,
            'y_cmd': 0,
            'z_cmd': 0,
            'rot_cmd': 0,
            'status': "Target Lost/Searching",
            'error_x': 0,
            'error_y': 0,
            'error_angle': 0,
            'error_major': 0,
            'error_minor': 0,
            'error_vertical': 0,
        }

        # --- Determine Final Command based on Target Position ---
        if target_data.is_lost or target_data.center is None:
            final_command_info = CommandInfo(
                timestamp=base_command_info.timestamp,
                x_cmd=base_command_info.x_cmd,
                y_cmd=base_command_info.y_cmd,
                z_cmd=base_command_info.z_cmd,
                rot_cmd=base_command_info.rot_cmd,
                x_offset=base_command_info.x_offset,
                y_offset=base_command_info.y_offset,
                p_rot=base_command_info.p_rot,
                d_rot=base_command_info.d_rot,
                angle_error=base_command_info.angle_error,
                status=base_command_info.status if base_command_info.status else "Target Lost/Searching"
            )
        else:
            target_x, target_y = target_data.center
            is_inside_box = (x1_85 <= target_x <= x2_85 and y1_85 <= target_y <= y2_85)
            log_data['is_inside_box'] = is_inside_box

            if is_inside_box and target_data.ellipse_keypoints is not None:
                # --- Enhanced IBVS Logic using 3 Points + Ellipse ---
                # Get the 5 keypoints: center, major1, major2, minor1, minor2
                center, major1, major2, minor1, minor2 = target_data.ellipse_keypoints
                
                # Calculate current vertical points
                current_vertical_distance = 50  # Same as desired distance
                current_points = np.array([
                    center,  # Center point
                    center + np.array([0, -current_vertical_distance]),  # Point above
                    center + np.array([0, current_vertical_distance]),   # Point below
                ])
                
                # Current feature vector (3 points + ellipse)
                current_features = np.array([
                    current_points[0],  # Center
                    current_points[1],  # Point above
                    current_points[2],  # Point below
                    major1,            # Major axis point 1
                    major2,            # Major axis point 2
                    minor1,            # Minor axis point 1
                    minor2,            # Minor axis point 2
                ]).flatten()
                
                # Desired feature vector
                desired_features = np.array([
                    desired_center,  # Center point
                    desired_center + np.array([0, -desired_vertical_distance]),  # Point above
                    desired_center + np.array([0, desired_vertical_distance]),   # Point below
                    desired_center + np.array([desired_major_length/2, 0]),     # Major axis point 1
                    desired_center - np.array([desired_major_length/2, 0]),     # Major axis point 2
                    desired_center + np.array([0, desired_minor_length/2]),     # Minor axis point 1
                    desired_center - np.array([0, desired_minor_length/2])      # Minor axis point 2
                ]).flatten()
                
                # Error vector
                error_vec = desired_features - current_features
                
                # Calculate specific errors for logging
                log_data['error_x'] = error_vec[0]  # Center x error
                log_data['error_y'] = error_vec[1]  # Center y error
                log_data['error_vertical'] = np.mean([error_vec[3], error_vec[5]])  # Average vertical points error
                log_data['error_major'] = np.mean([error_vec[6], error_vec[8]])  # Average major axis error
                log_data['error_minor'] = np.mean([error_vec[10], error_vec[12]])  # Average minor axis error
                log_data['error_angle'] = target_data.ellipse_angle if target_data.ellipse_angle is not None else 0
                
                # --- IBVS Parameters ---
                lambda_gain = 0.3  # Convergence gain
                
                # Interaction matrix for 7 points (3 vertical + 4 ellipse)
                L_eff = np.zeros((14, 4))  # 14 features (7 points × 2 coordinates) × 4 controls (x,y,z,rot)
                
                sin_p = np.sin(-45)
                cos_p = np.cos(-45)
                
                # Fill interaction matrix (simplified model)
                for i in range(7):
                    # For downward-facing camera (-90 degrees):
                    # x velocity affects y coordinates (inverted)
                    L_eff[2*i, 0] = cos_p
                    # y velocity affects x coordinates (inverted)
                    L_eff[2*i+1, 1] = sin_p
                    # z velocity affects both x and y (scaled by distance from center)
                    L_eff[2*i, 2] = sin_p * (current_features[2*i] - frame_center_x) / frame_width
                    L_eff[2*i+1, 2] = cos_p * (current_features[2*i+1] - frame_center_y) / frame_height
                    # rotation affects both x and y (adjusted for downward camera)
                    L_eff[2*i, 3] = -(current_features[2*i+1] - frame_center_y)
                    L_eff[2*i+1, 3] = (current_features[2*i] - frame_center_x)
                
                # Pseudo-inverse of interaction matrix
                L_eff_inv = np.linalg.pinv(L_eff)
                
                # IBVS Control Law: cmd = -lambda * L_inv * error
                command_vec = -lambda_gain * L_eff_inv @ error_vec
                print(f'Command Vector: {command_vec}')
                
                # Extract and clamp commands
                ibvs_x_cmd = int(np.clip(command_vec[0], -100, 100))  # Left/Right
                ibvs_y_cmd = int(np.clip(command_vec[1], -100, 100))  # Forward/Backward
                ibvs_z_cmd = 0  # Altitude
                ibvs_rot_cmd = int(np.clip(20*command_vec[3], -100, 100))  # Rotation
                
                # Update log data with commands
                log_data.update({
                    'x_cmd': ibvs_x_cmd,
                    'y_cmd': ibvs_y_cmd,
                    'z_cmd': ibvs_z_cmd,
                    'rot_cmd': ibvs_rot_cmd,
                    'status': "IBVS Tracking (3 Points + Ellipse)"
                })
                
                final_command_info = CommandInfo(
                    timestamp=base_command_info.timestamp,
                    x_cmd=ibvs_x_cmd,
                    y_cmd=ibvs_y_cmd,
                    z_cmd=ibvs_z_cmd,
                    rot_cmd=ibvs_rot_cmd,
                    x_offset=base_command_info.x_offset,
                    y_offset=base_command_info.y_offset,
                    p_rot=base_command_info.p_rot,
                    d_rot=base_command_info.d_rot,
                    angle_error=base_command_info.angle_error,
                    status="IBVS Tracking (3 Points + Ellipse)"
                )
                # --- End Enhanced IBVS Logic ---
            else:
                # --- Centering Logic (Target Outside Box) ---
                final_command_info = CommandInfo(
                    timestamp=base_command_info.timestamp,
                    x_cmd=0,                        # Disable sideways strafing
                    y_cmd=0,                        # Disable forward/backward movement
                    z_cmd=base_command_info.z_cmd,  # Keep altitude adjustment
                    rot_cmd=base_command_info.rot_cmd, # Keep rotation command
                    x_offset=base_command_info.x_offset,
                    y_offset=base_command_info.y_offset,
                    p_rot=base_command_info.p_rot,
                    d_rot=base_command_info.d_rot,
                    angle_error=base_command_info.angle_error,
                    status="Centering"
                )
                # Update log data for centering mode
                log_data.update({
                    'x_cmd': 0,
                    'y_cmd': 0,
                    'z_cmd': base_command_info.z_cmd,
                    'rot_cmd': base_command_info.rot_cmd,
                    'status': "Centering"
                })
                # --- End Centering Logic ---
        # --- End Movement Calculation ---

        # Save log data to CSV
        if self.detector_log_dir is not None:
            new_data = pd.DataFrame([log_data])
            new_data.to_csv(
                self.detector_log_dir, mode='a', header=not os.path.exists(self.detector_log_dir), index=False
            )

        self.perform_movement(final_command_info)

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
    
    import time
    controller.drone_commander.tilt_camera(pitch_deg=-90)
