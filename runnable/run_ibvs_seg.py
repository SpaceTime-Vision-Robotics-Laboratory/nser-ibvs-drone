from pathlib import Path
import sys
import time
import math
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

import numpy as np
import cv2
from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor
from ultralytics import YOLO

camera_fx = 465
camera_fy = 348
camera_cx = 320.0
camera_cy = 180.0

lambda_gain = 0.05

X1Y1 = (270, 65)
X2Y2 = (370, 305)
X1Y2 = (270, 305)
X2Y1 = (370, 65)

class IBVSController(BaseStreamingController):
    def __init__(self, lambda_gain:float = lambda_gain, **kwargs):
        # Pass lambda_gain to the base controller to be forwarded to the processor
        processor_kwargs = {'lambda_gain': lambda_gain}
        super().__init__(processor_kwargs=processor_kwargs, **kwargs)
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
        
        
    def initialize_position(self, takeoff_height=5.0, gimbal_angle=-90, back_distance=-0.0, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
            
        print("Taking off...")
        
        self.drone_commander.tilt_camera(pitch_deg=gimbal_angle, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        
        self.drone_commander.take_off()
        time.sleep(1)
        
        print(f"Moving to initial position...")
        self.drone_commander.move_by(forward=back_distance, right=0, down=-takeoff_height, rotation=0)
        time.sleep(1)
        
        print(f"Starting Program...")

class IBVSProcessor(BaseVideoProcessor):
    def __init__(self, lambda_gain: float = lambda_gain, model_path: str = "/home/sebnae/shared_drive/ws/drone_ws/auto-follow/models/best__yolo11n-seg__seg__27_04__sim.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = YOLO(model_path)
        self.lambda_gain = lambda_gain
        # State for defining target features
        self.target_features_defined = False
        self.stabilization_duration = 3.0 # Seconds
        self.stabilization_start_time = None
        self.locked_target_angle_rad = None # Store the target orientation angle
        # State for PCA results smoothing
        self.smoothed_pca_params = None # Stores (cx, cy, angle_rad, half_major, half_minor)
        self.smoothing_alpha = 0.1 # Smoothing factor (lower = more smoothing)
        # --- New parameters ---
        self.center_box_ratio = 0.75
        self.centering_threshold = 10.0 # Pixels

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # if self._frame_count % 10 == 0:
        #     return frame
        
        results = self.detector.predict(frame, stream=True, verbose=False)
        first_result = next(results, None)

        # Initialize variables for this frame
        pitch_error = 0.0
        angle_error = 0.0
        pitch_cmd = 0
        yaw_rate_cmd = 0
        roll_cmd = 0
        gaz_cmd = 0
        smoothed_params_calculated = False

        plotted_frame = frame.copy()
        height, width = frame.shape[:2]
        image_center_x, image_center_y = width // 2, height // 2

        # --- Parameters ---
        # target_separation = 30.0 # Fixed separation for target orientation points (No longer used directly)

        # --- Calculate 75% Bounding Box ---
        box_width = width * self.center_box_ratio
        box_height = height * self.center_box_ratio
        x_min = int(image_center_x - box_width / 2)
        y_min = int(image_center_y - box_height / 2)
        x_max = int(image_center_x + box_width / 2)
        y_max = int(image_center_y + box_height / 2)

        # --- Target Definition --- 
        # Target for pitch/roll is image_center_y/image_center_x
        # Target for yaw is self.locked_target_angle_rad (defined after stabilization)

        # --- Visualization Defaults --- 
        cv2.circle(plotted_frame, (image_center_x, image_center_y), 3, (255, 255, 255), -1) # Image Center (White)
        cv2.rectangle(plotted_frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2) # 75% Box (Cyan)
        # No target points to draw initially

        object_detected_this_frame = False
        is_within_box = False # Default status for logging
        centering_error_dist = 0.0 # Default status for logging

        # --- Detect Object Mask and Calculate Smoothed PCA Features --- 
        if first_result and first_result.masks and first_result.masks.xy:
            mask_contours = first_result.masks.xy
            if mask_contours and len(mask_contours) > 0:
                contour = np.array(mask_contours[0], dtype=np.int32)
                if len(contour) >= 5: # Need enough points for PCA
                    try:
                        # --- PCA Calculation ---
                        points = contour.astype(np.float32)
                        raw_cx, raw_cy = np.mean(points, axis=0)
                        centered_points = points - (raw_cx, raw_cy)
                        covariance_matrix = np.cov(centered_points, rowvar=False)
                        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                        sort_indices = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[sort_indices]
                        eigenvectors = eigenvectors[:, sort_indices]
                        raw_angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

                        # Use 2*sqrt(eigenvalue) as half axis length estimate
                        # Add checks for non-negative eigenvalues before sqrt
                        if eigenvalues[0] < 0 or eigenvalues[1] < 0:
                             raise ValueError("PCA Eigenvalues cannot be negative.")
                        raw_half_major = 2 * np.sqrt(eigenvalues[0])
                        raw_half_minor = 2 * np.sqrt(eigenvalues[1])

                        # Check for NaN/inf just in case
                        if not (np.isfinite(raw_half_major) and np.isfinite(raw_half_minor)):
                            raise ValueError("PCA axis lengths are not finite.")

                        # --- Apply Temporal Smoothing (EMA) to PCA results --- 
                        if self.smoothed_pca_params is None:
                            self.smoothed_pca_params = (raw_cx, raw_cy, raw_angle_rad, raw_half_major, raw_half_minor)
                        else:
                            prev_cx, prev_cy, prev_angle, prev_hmaj, prev_hmin = self.smoothed_pca_params
                            alpha = self.smoothing_alpha
                            sm_cx = alpha * raw_cx + (1 - alpha) * prev_cx
                            sm_cy = alpha * raw_cy + (1 - alpha) * prev_cy
                            sm_cos = alpha * math.cos(raw_angle_rad) + (1 - alpha) * math.cos(prev_angle)
                            sm_sin = alpha * math.sin(raw_angle_rad) + (1 - alpha) * math.sin(prev_angle)
                            sm_angle_rad = math.atan2(sm_sin, sm_cos)
                            sm_half_major = alpha * raw_half_major + (1 - alpha) * prev_hmaj
                            sm_half_minor = alpha * raw_half_minor + (1 - alpha) * prev_hmin
                            self.smoothed_pca_params = (sm_cx, sm_cy, sm_angle_rad, sm_half_major, sm_half_minor)

                        # --- Use SMOOTHED parameters ---
                        cx, cy, angle_rad, half_major, half_minor = self.smoothed_pca_params
                        smoothed_params_calculated = True # Flag that we have params for this frame
                        object_detected_this_frame = True # Redundant but clear

                        # --- Calculate current orientation points for drawing ---
                        cos_a = np.cos(angle_rad)
                        sin_a = np.sin(angle_rad)
                        current_center = np.array([cx, cy])
                        current_orient_p1 = current_center + half_major * np.array([cos_a, sin_a])
                        current_orient_p2 = current_center - half_major * np.array([cos_a, sin_a])
                        current_orient_p3 = current_center + half_minor * np.array([-sin_a, cos_a])
                        current_orient_p4 = current_center - half_minor * np.array([-sin_a, cos_a])

                        # --- Draw Raw Contour and SMOOTHED Features/Axes (Moved Inside Try) ---
                        cv2.drawContours(plotted_frame, [contour], -1, (0, 255, 0), 1) # Contour (Green)
                        cv2.circle(plotted_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1) # Smoothed Centroid (Red)
                        cv2.circle(plotted_frame, (int(current_orient_p1[0]), int(current_orient_p1[1])), 5, (255, 0, 0), -1) # Smoothed Major P1 (Blue)
                        cv2.circle(plotted_frame, (int(current_orient_p2[0]), int(current_orient_p2[1])), 5, (255, 0, 0), -1) # Smoothed Major P2 (Blue)
                        cv2.circle(plotted_frame, (int(current_orient_p3[0]), int(current_orient_p3[1])), 5, (0, 255, 0), -1) # Smoothed Minor P1 (Lime)
                        cv2.circle(plotted_frame, (int(current_orient_p4[0]), int(current_orient_p4[1])), 5, (0, 255, 0), -1) # Smoothed Minor P2 (Lime) 
                        cv2.line(plotted_frame, (int(current_orient_p2[0]), int(current_orient_p2[1])), (int(current_orient_p1[0]), int(current_orient_p1[1])), (0, 165, 255), 2) # Major Axis (Orange)
                        cv2.line(plotted_frame, (int(current_orient_p4[0]), int(current_orient_p4[1])), (int(current_orient_p3[0]), int(current_orient_p3[1])), (255, 0, 255), 2) # Minor Axis (Purple)

                    except (np.linalg.LinAlgError, ValueError, Exception) as e:
                         self.logger.warning(f"Error during PCA or feature definition: {e}")
                         object_detected_this_frame = False # Ensure flag is false on error
                         smoothed_params_calculated = False # Ensure flag is false on error

        # --- State Machine: Stabilization / Tracking --- 
        # Initialize commands for this frame
        pitch_cmd = 0
        roll_cmd = 0
        gaz_cmd = 0
        yaw_rate_cmd = 0
        pitch_error = 0.0 # For logging
        roll_error = 0.0 # For logging
        angle_error = 0.0 # For logging

        if not self.target_features_defined: # Using this flag name, but it now means 'target angle defined'
            # --- Stabilization Phase --- 
            if object_detected_this_frame and smoothed_params_calculated:
                # Start or continue timer
                if self.stabilization_start_time is None:
                    self.stabilization_start_time = time.time()
                    self.logger.info("Stabilization timer started.")
                
                elapsed_time = time.time() - self.stabilization_start_time
                self.logger.info(f"Stabilizing... Time: {elapsed_time:.2f}/{self.stabilization_duration:.1f}s")

                if elapsed_time >= self.stabilization_duration:
                    # --- Lock Target Angle using current smoothed angle --- 
                    _, _, sm_angle_rad, _, _ = self.smoothed_pca_params # Get angle from smoothed params
                    self.locked_target_angle_rad = sm_angle_rad
                    self.target_features_defined = True # Mark target angle as defined
                    self.logger.info(f"Target angle ({np.rad2deg(sm_angle_rad):.1f} deg) defined and locked after {elapsed_time:.2f}s!")
                
            else:
                # Object lost or PCA failed during stabilization
                if self.stabilization_start_time is not None:
                    self.logger.info("Object lost/PCA failed during stabilization, resetting timer.")
                    self.stabilization_start_time = None # Reset timer
                # No commands sent during stabilization
                pass # Commands already initialized to 0

        else:
            # --- Tracking Phase --- 
            if object_detected_this_frame and smoothed_params_calculated and self.locked_target_angle_rad is not None:
                # Get current smoothed parameters
                sm_cx, sm_cy, sm_angle_rad, _, _ = self.smoothed_pca_params
                
                # 1. Calculate Positional Errors (relative to image center)
                pitch_error = sm_cy - image_center_y # Vertical error
                roll_error = sm_cx - image_center_x # Horizontal error
                centering_error_dist = np.sqrt(pitch_error**2 + roll_error**2)
                
                # 2. Calculate Angle Error 
                angle_error = self.locked_target_angle_rad - sm_angle_rad
                # Normalize error to [-pi, pi]
                while angle_error > np.pi: angle_error -= 2 * np.pi
                while angle_error <= -np.pi: angle_error += 2 * np.pi

                # 3. Check if centroid is within the 75% box (for status)
                is_within_box = (x_min <= sm_cx <= x_max) and (y_min <= sm_cy <= y_max)

                # --- Direct Control Command Calculation --- 
                MAX_COMMAND = 20
                K_trans = 0.1 # Translational gain (for pitch/roll)
                K_rot_angle = 50 # Rotational gain (for yaw)

                # Always calculate potential yaw command
                potential_yaw_rate_cmd = int(np.clip(-angle_error * K_rot_angle, -MAX_COMMAND, MAX_COMMAND))
                yaw_rate_cmd = potential_yaw_rate_cmd

                # Only apply pitch/roll commands if outside the centering threshold
                if centering_error_dist > self.centering_threshold:
                    potential_pitch_cmd = int(np.clip(-pitch_error * K_trans, -MAX_COMMAND, MAX_COMMAND))
                    potential_roll_cmd = int(np.clip(roll_error * K_trans, -MAX_COMMAND, MAX_COMMAND))
                    pitch_cmd = potential_pitch_cmd
                    roll_cmd = potential_roll_cmd
                    control_status = "CENTERING"
                else:
                    # Inside threshold, don't command roll/pitch (set to 0)
                    pitch_cmd = 0
                    roll_cmd = 0
                    control_status = "TRACKING (Centered)"
                
            else:
                # Object lost or PCA failed during tracking
                self.logger.warning("Object lost/PCA failed during tracking. Sending zero commands.")
                # Keep commands at 0 (already initialized)
                control_status = "LOST"
                # Optionally reset target angle if lost for too long? 
                # self.target_features_defined = False 
                # self.stabilization_start_time = None
                # self.locked_target_angle_rad = None
                pass # Commands already initialized to 0

        # --- Apply Control Law (Commands already calculated/set) --- 

        # Log commands and errors
        status = "Stabilizing" if not self.target_features_defined else control_status
        stab_timer_info = f" ({(time.time() - self.stabilization_start_time):.1f}s)" if self.stabilization_start_time is not None and not self.target_features_defined else ""
        
        current_state_info = ""
        if self.target_features_defined and object_detected_this_frame:
             current_state_info = f"InBox:{is_within_box}, C_Err:{centering_error_dist:.1f}, P_Err:{pitch_error:.1f}, R_Err:{roll_error:.1f}, A_Err:{np.rad2deg(angle_error):.1f}"
        elif not self.target_features_defined and object_detected_this_frame:
             current_state_info = "Waiting for stabilization lock"
        
        print(f'Status:{status}{stab_timer_info}, {current_state_info} -> Cmd: R:{roll_cmd}, P:{pitch_cmd}, G:{gaz_cmd}, YR:{yaw_rate_cmd}')

        # Send piloting command
        self.drone_commander.piloting(x=roll_cmd, y=pitch_cmd, z=gaz_cmd, z_rot=yaw_rate_cmd, dt=0.1)

        return plotted_frame
    
    
    
if __name__ == "__main__":
    controller = IBVSController(
        ip=DroneIp.SIMULATED,
        processor_class=IBVSProcessor,
        speed=35
    )
    
    controller.initialize_position(takeoff_height=10)
    controller.run()
    