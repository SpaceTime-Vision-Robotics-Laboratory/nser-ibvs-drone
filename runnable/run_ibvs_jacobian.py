import argparse
import sys
from pathlib import Path
import ultralytics
import pandas as pd
import os
import time
import numpy as np
import cv2
from collections import deque  # Add deque for historical tracking

# Modify paths to include drone_base
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor

# Default Camera Intrinsics (Example - REPLACE with actual values for your drone camera)
# These might be found in camera documentation or estimated via calibration.
# Values are in pixels.
DEFAULT_FX = 465.60298  # Focal length in x
DEFAULT_FY = 465.60298  # Focal length in y
DEFAULT_CX = 320.0  # Principal point x-coordinate (assuming 640 width)
DEFAULT_CY = 180.0  # Principal point y-coordinate (assuming 720 height)


class IBVSJacobianController(BaseStreamingController):
    def __init__(self, **kwargs):
        # Extract processor-specific args and store temporarily
        processor_class = kwargs.get('processor_class')
        # Also extract init params needed for processor config (like Z calculation)
        processor_kwargs = {k: v for k, v in kwargs.items() if k not in ['ip', 'processor_class']}
        self._processor_kwargs = processor_kwargs

        # Call super init, it will create the processor using its default __init__
        super().__init__(ip=kwargs.get('ip'), processor_class=processor_class)

        # --- Configure processor after its initial __init__ ---
        if hasattr(self, 'frame_processor') and isinstance(self.frame_processor, IBVSJacobianProcessor):
            print(f"Configuring IBVSJacobianProcessor with provided arguments...")

            # Set basic attributes from kwargs
            self.frame_processor.target_width_ratio = self._processor_kwargs.get('target_width_ratio', 0.3)
            self.frame_processor.lambda_gain = self._processor_kwargs.get('lambda_gain', 0.5)
            self.frame_processor.fx = self._processor_kwargs.get('fx', DEFAULT_FX)
            self.frame_processor.fy = self._processor_kwargs.get('fy', DEFAULT_FY)
            self.frame_processor.cx = self._processor_kwargs.get('cx', DEFAULT_CX)
            self.frame_processor.cy = self._processor_kwargs.get('cy', DEFAULT_CY)
            # Store init height/gimbal needed for Z calculation
            self.frame_processor.init_height = self._processor_kwargs.get('init_height')
            self.frame_processor.init_gimbal = self._processor_kwargs.get('init_gimbal')
            self.frame_processor.override_depth_Z = self._processor_kwargs.get('assumed_depth') # User override
            
            # Add new parameters for improved tracking
            self.frame_processor.confidence_threshold = self._processor_kwargs.get('confidence_threshold', 0.5)
            self.frame_processor.feature_smoothing_alpha = self._processor_kwargs.get('feature_smoothing_alpha', 0.7)

            # Load the model
            model_path = self._processor_kwargs.get('model_path', 'models/yolov11n_best_car_simulator.pt')
            try:
                self.frame_processor.detector = ultralytics.YOLO(model_path)
                print(f"Successfully loaded YOLO model from: {model_path}")
            except Exception as e:
                print(f"Error loading YOLO model from {model_path}: {e}")
                raise # Critical error

            # Reset target features so they recalculate with potentially new intrinsics/settings
            self.frame_processor.target_features_s_star = None
            self.frame_processor.target_width_pixels = None
            self.frame_processor.target_height_pixels = None
            self.frame_processor.target_aspect_ratio = None
            self.frame_processor.Z = None # Ensure Z is recalculated
            
            # Initialize history queues
            self.frame_processor.depth_history = deque(maxlen=10)
            self.frame_processor.feature_history = deque(maxlen=5)
            self.frame_processor.last_valid_features = None

            # Clear the temporary storage
            del self._processor_kwargs
        else:
            print("Warning: Could not find/configure frame_processor of type IBVSJacobianProcessor.")
        # -----------------------------------------------------


    def initialize_position(self, takeoff_height=1.5, gimbal_angle=-45, back_distance=2.5, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        self.drone_commander.take_off()
        time.sleep(1)
        self.drone_commander.tilt_camera(pitch_deg=gimbal_angle, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(1)
        self.drone_commander.move_by(forward=-back_distance, right=-left_distance, down=-takeoff_height, rotation=0)
        time.sleep(1)
        if hasattr(self, 'frame_processor') and hasattr(self.frame_processor, 'frame_queue'):
             self.frame_processor.frame_queue.empty() # Clear queue after movement
        print("Initialization complete. Ready for tracking.")


class IBVSJacobianProcessor(BaseVideoProcessor):
    def __init__(self, **kwargs): # Use kwargs initially, attributes set by Controller
        super().__init__(**kwargs)
        print(f"Initializing IBVSJacobianProcessor (defaults set, awaiting config)...")

        # Attributes are set by the Controller after initialization
        self.detector = None
        self.target_width_ratio = 0.3
        self.lambda_gain = 0.5
        self.fx = DEFAULT_FX
        self.fy = DEFAULT_FY
        self.cx = DEFAULT_CX
        self.cy = DEFAULT_CY
        self.init_height = None # Set by controller
        self.init_gimbal = None # Set by controller
        self.override_depth_Z = None # Set by controller from --depth arg

        # Internal state
        self.Z = None # Estimated/calculated depth, calculated on first run
        self.target_aspect_ratio = None
        self.target_width_pixels = None
        self.target_height_pixels = None
        self.target_features_s_star = None # Target feature vector s* = [u1*, v1*, ..., u4*, v4*, uc*, vc*]
        self.last_command_time = time.time()
        self.log_file = 'ibvs_jacobian_log.csv'
        if os.path.exists(self.log_file):
            print(f"Removing existing log file: {self.log_file}")
            os.remove(self.log_file)

        # New for reference features
        self.ref_features_s = None
        self.reference_captured = False
        
        # New additions for improved tracking
        self.depth_history = deque(maxlen=10)  # Store recent depth estimates
        self.feature_history = deque(maxlen=5)  # Store recent feature positions
        self.last_valid_features = None        # Last known good features
        self.confidence_threshold = 0.5        # Minimum detection confidence
        self.feature_smoothing_alpha = 0.7     # Smoothing factor for feature positions (0-1)

    def _get_features_from_box(self, box_coords):
        x1, y1, x2, y2 = box_coords
        # Corners: Top-left, Top-right, Bottom-right, Bottom-left
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        # Center
        center = np.array([[ (x1 + x2) / 2, (y1 + y2) / 2 ]], dtype=np.float32)
        # Stack corners and center
        features = np.vstack((corners, center)) # Shape (5, 2)
        return features

    def _compute_interaction_matrix(self, features_points, Z):
        """ Computes the 2Nx6 interaction matrix (Jacobian) for N feature points (u, v). """
        if Z <= 0: # Depth must be positive
            print("Warning: Assumed depth Z must be positive.")
            return None

        num_points = features_points.shape[0] # Number of points (e.g., 4 corners + 1 center = 5)
        L_s = np.zeros((2 * num_points, 6)) # Will be 10x6

        for i in range(num_points):
            u_pix = features_points[i, 0] # Pixel coordinate u
            v_pix = features_points[i, 1] # Pixel coordinate v
            # u = u_pix - self.cx # Coordinate relative to principal point (Optional for Jacobian)
            # v = v_pix - self.cy # Coordinate relative to principal point (Optional for Jacobian)

            # Interaction matrix rows for point i
            L_point = np.array([
                [-self.fx / Z, 0,           u_pix / Z,      (u_pix * v_pix) / self.fx, -(self.fx**2 + u_pix**2) / self.fx, v_pix],
                [0,           -self.fy / Z, v_pix / Z, (self.fy**2 + v_pix**2) / self.fy, -(u_pix * v_pix) / self.fy,       -u_pix]
            ])
            L_s[2*i : 2*i+2, :] = L_point

        return L_s

    def _process_frame(self, frame: np.ndarray) -> list:
        if self.detector is None: return [frame, None]
        results = self.detector.predict(frame, stream=False, verbose=False)
        processed_results = results[0] if results else None
        has_boxes = processed_results and processed_results.boxes and hasattr(processed_results.boxes, 'conf') and len(processed_results.boxes.conf) > 0

        # Calculate target features on first valid detection
        if self.target_features_s_star is None and has_boxes:
            best_conf_idx = processed_results.boxes.conf.argmax()
            coords = processed_results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            current_w, current_h = x2 - x1, y2 - y1

            if current_w > 0 and current_h > 0:
                frame_height, frame_width = frame.shape[:2]
                # Update intrinsics cx, cy based on frame size if needed (optional)
                # self.cx = frame_width / 2
                # self.cy = frame_height / 2

                # Calculate target W/H/Aspect
                self.target_width_pixels = int(frame_width * self.target_width_ratio)
                self.target_aspect_ratio = current_h / current_w
                self.target_height_pixels = int(self.target_width_pixels * self.target_aspect_ratio)

                # Define target box centered in the image using potentially updated cx, cy
                tx_center = self.cx
                ty_center = self.cy
                tw_half = self.target_width_pixels / 2
                th_half = self.target_height_pixels / 2
                tx1 = tx_center - tw_half
                ty1 = ty_center - th_half
                tx2 = tx_center + tw_half
                ty2 = ty_center + th_half
                target_box = (tx1, ty1, tx2, ty2)

                # Calculate target corner and center features (u*, v*)
                target_features = self._get_features_from_box(target_box) # Gets 5x2 array
                self.target_features_s_star = target_features.flatten() # Shape (10,)

                print(f"Frame: {frame_width}x{frame_height}, Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")
                print(f"First Detection: W={current_w}, H={current_h}, Aspect={self.target_aspect_ratio:.2f}")
                print(f"Target Box Set: W={self.target_width_pixels}, H={self.target_height_pixels}")

        return [frame, processed_results]

    def _calculate_depth_Z(self):
        """ Estimates depth Z based on init params or uses override with filtering. """
        if self.override_depth_Z is not None:
            if self.override_depth_Z > 0:
                # Add to depth history
                self.depth_history.append(self.override_depth_Z)
                # Return median filtered depth for stability
                return np.median(list(self.depth_history))
            else:
                print(f"Warning: Invalid override depth Z ({self.override_depth_Z}). Calculating from init params.")

        if self.init_height is None or self.init_gimbal is None:
            print("Warning: init_height or init_gimbal not available for Z calculation. Using default Z=3.0m.")
            default_depth = 3.0
            self.depth_history.append(default_depth)
            return default_depth

        # Calculate Z using trigonometry: Z = height / sin(abs(angle))
        gimbal_rad = np.radians(self.init_gimbal)
        sin_gimbal = np.sin(abs(gimbal_rad))

        if abs(sin_gimbal) < 0.05: # Avoid division by zero/instability near 0 degrees
            print(f"Warning: Gimbal angle ({self.init_gimbal:.1f} deg) too close to zero for Z calculation. Using default Z=3.0m.")
            default_depth = 3.0
            self.depth_history.append(default_depth)
            return default_depth
        else:
            calculated_Z = self.init_height / sin_gimbal
            
            # Add outlier rejection (don't accept depths that change too rapidly)
            if self.depth_history:
                median_depth = np.median(list(self.depth_history))
                if abs(calculated_Z - median_depth) > 2.0:  # More than 2m difference
                    print(f"Depth outlier detected: {calculated_Z:.2f}m, using filtered: {median_depth:.2f}m")
                    calculated_Z = median_depth
            
            # Add to history and return median-filtered result
            self.depth_history.append(calculated_Z)
            return np.median(list(self.depth_history))

    def _get_smoothed_features(self, current_features):
        """Apply temporal smoothing to feature points to reduce jitter"""
        if current_features is None:
            return self.last_valid_features
            
        # Add to history
        self.feature_history.append(current_features)
        
        if self.last_valid_features is None:
            # First valid features, just store them
            self.last_valid_features = current_features
            return current_features
            
        # Apply exponential smoothing to reduce jitter
        alpha = self.feature_smoothing_alpha
        smoothed_features = alpha * current_features + (1-alpha) * self.last_valid_features
        self.last_valid_features = smoothed_features
        
        return smoothed_features

    def _generate_follow_command(self, frame_dimensions, detection_results):
        """ Calculates drone velocity commands using Jacobian-based IBVS. """
        current_time = time.time()
        dt = max(0.01, current_time - self.last_command_time) # Ensure dt > 0

        # --- Calculate/Update Depth Z ---
        # Re-calculate Z every frame using the filtered method
        # This allows Z to adapt if the drone's altitude changes significantly
        current_Z = self._calculate_depth_Z()
        if current_Z is not None:
            self.Z = current_Z
            # print(f"Updated Z: {self.Z:.2f}m") # Optional debug print
        elif self.Z is None: # Only if calculation fails AND Z was never set
             print("Error: Depth Z could not be calculated and was not previously set. Hovering.")
             self.drone_commander.piloting(0,0,0,0, dt=dt) # Hover command
             self.last_command_time = current_time
             return # Stop processing if no valid Z

        # ---------------------------------

        frame_width, frame_height = frame_dimensions
        current_features_s = None # Flattened 10 features
        current_features = None # 5x2 features
        best_box = None
        target_lost = True
        status = "Lost/Hover"
        has_boxes = False

        # Check if we have valid detection with sufficient confidence
        if detection_results and detection_results.boxes and hasattr(detection_results.boxes, 'conf') and len(detection_results.boxes.conf) > 0:
            # Find the most confident detection above threshold
            best_conf_idx = -1
            best_conf = 0

            for i, conf in enumerate(detection_results.boxes.conf):
                conf_val = conf.item() if hasattr(conf, 'item') else conf
                if conf_val > self.confidence_threshold and conf_val > best_conf:
                    best_conf = conf_val
                    best_conf_idx = i

            # Extract features from best detection if found
            if best_conf_idx >= 0:
                has_boxes = True
                target_lost = False
                coords = detection_results.boxes.xyxy[best_conf_idx].cpu().numpy()
                best_box = tuple(map(int, coords)) # (x1, y1, x2, y2)

                # Get current features
                current_features = self._get_features_from_box(best_box) # 5x2 float array

                # Apply temporal smoothing
                current_features = self._get_smoothed_features(current_features)
                if current_features is not None: # Check if smoothing returned valid features
                     current_features_s = current_features.flatten() # Shape (10,)
                else: # If smoothing failed (e.g., no history yet), revert to raw
                     current_features = self._get_features_from_box(best_box)
                     current_features_s = current_features.flatten()


        # If target lost but we have recent features, use them with reduced confidence
        # Check self.reference_captured to ensure we don't use old features before reference is set
        if target_lost and self.reference_captured and self.last_valid_features is not None:
            print("Target temporarily lost - using last valid features")
            status = "Using Last Features"
            current_features = self.last_valid_features
            current_features_s = current_features.flatten()
            target_lost = False  # Pretend we still have the target for control calculation

        camera_velocity_vc = np.zeros(6)
        x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0
        error_norm = 0.0

        # --- Capture Reference Features on First Good Detection ---
        if not self.reference_captured and not target_lost and current_features is not None:
            self.ref_features_s = current_features_s # Store the flattened 10 features
            self.reference_captured = True
            print(f"*** Reference Features Captured ***")
            # status = "Reference Captured" # Set status below
            # Hover briefly after capture? Optional.
            # camera_velocity_vc = np.zeros(6)
            # x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0
        # ------------------------------------------------------

        # --- Calculate Control Commands (if reference is captured) ---
        if self.reference_captured and not target_lost and current_features_s is not None and self.Z is not None:
            status = "Tracking (Ref)"
            # Calculate feature error: e = s_current - s_reference
            error_e = current_features_s - self.ref_features_s
            error_norm = np.linalg.norm(error_e)

            # Compute interaction matrix L_s for current features (5 points -> 10x6 matrix)
            # Use the current smoothed features for L_s calculation
            L_s = self._compute_interaction_matrix(current_features, self.Z)

            if L_s is not None:
                try:
                    # Use Moore-Penrose pseudo-inverse for robustness
                    damping_factor = 0.01 # Small damping for numerical stability
                    L_s_pinv = np.linalg.pinv(L_s, rcond=damping_factor)
                    # L_s_pinv = np.linalg.inv(L_s.T @ L_s + damping_factor * np.identity(6)) @ L_s.T # Alternative: Explicit damped inverse

                    # Calculate camera velocity command: vc = -lambda * L_s^+ * e
                    camera_velocity_vc = -self.lambda_gain * L_s_pinv @ error_e
                except np.linalg.LinAlgError:
                    print("Warning: Jacobian pseudo-inverse calculation failed (LinAlgError). Hovering.")
                    status = "Jacobian Singularity"
                    camera_velocity_vc = np.zeros(6) # Hover if pseudo-inverse fails
            else:
                print("Warning: Interaction matrix computation failed (likely invalid depth Z). Hovering.")
                status = "Invalid Jacobian"
                camera_velocity_vc = np.zeros(6) # Hover if Jacobian is invalid

            # --- Command Mapping (Corrected for Gimbal Angle) ---
            v_x, v_y, v_z, w_x, w_y, w_z = camera_velocity_vc

            # Get gimbal angle (use initial angle, assume it doesn't change significantly during tracking)
            # If the gimbal *can* change during operation, this needs to be updated dynamically
            gimbal_rad = np.radians(self.init_gimbal if self.init_gimbal is not None else 0.0) # Default to 0 if not set
            cos_g = np.cos(gimbal_rad)
            sin_g = np.sin(gimbal_rad)

            # Rotate camera velocities (v_x, v_z) into drone frame (y_drone, -z_drone)
            # Drone Y (Forward) = Cam X * cos(g) - Cam Z * sin(g)
            # Drone Z (Up) = - (Cam X * sin(g) + Cam Z * cos(g))
            drone_vy_raw = v_x * cos_g - v_z * sin_g
            drone_vz_raw = -(v_x * sin_g + v_z * cos_g)
            drone_vx_raw = v_y # Drone X (Right) = Cam Y (Right)
            drone_wz_raw = w_z # Drone Yaw = Cam Z rot

            # Scaling factors - **MAY NEED TUNING BASED ON REAL-WORLD TESTS**
            # These scales convert the desired velocity (m/s or rad/s) from the IBVS controller
            # into the drone's piloting command units (often arbitrary units or percentages).
            # The optimal values depend on drone dynamics, altitude, and controller gain (lambda).
            vel_scale_linear = 75.0  # Adjust for forward/lateral/vertical speed sensitivity
            vel_scale_angular = 90.0 # Adjust for yaw speed sensitivity
            max_raw_vel = 1.0 # Limit raw IBVS output before scaling

            # Clip raw velocities first
            drone_vx_raw, drone_vy_raw, drone_vz_raw, drone_wz_raw = np.clip(
                [drone_vx_raw, drone_vy_raw, drone_vz_raw, drone_wz_raw], -max_raw_vel, max_raw_vel
            )

            # Map rotated and scaled velocities to drone commands
            # Drone axes: X=Right, Y=Forward, Z=Up
            x_cmd = int(np.clip(vel_scale_linear * drone_vx_raw, -50, 50))    # Lateral (Drone X)
            y_cmd = int(np.clip(vel_scale_linear * drone_vy_raw, -50, 50))    # Forward (Drone Y)
            z_cmd = int(np.clip(vel_scale_linear * drone_vz_raw, -50, 50))    # Altitude (Drone Z)
            rot_cmd = int(np.clip(vel_scale_angular * drone_wz_raw, -70, 70)) # Yaw (Drone Yaw)
            # --- End Command Mapping ---

        elif self.reference_captured and target_lost:
             status = "Lost/Hovering"
             x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0 # Hover if lost after capture
        elif not self.reference_captured:
             status = "Awaiting Reference"
             x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0 # Hover while waiting
        else: # Should not happen unless Z is invalid from the start
            status = "Unknown State/Hover"
            x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0

        # --- Logging ---
        log_data = { 'timestamp': current_time, 'status': status, 'error_norm': error_norm,
                     'vc_vx': camera_velocity_vc[0], 'vc_vy': camera_velocity_vc[1], 'vc_vz': camera_velocity_vc[2],
                     'vc_wx': camera_velocity_vc[3], 'vc_wy': camera_velocity_vc[4], 'vc_wz': camera_velocity_vc[5],
                     'x_cmd': x_cmd, 'y_cmd': y_cmd, 'z_cmd': z_cmd, 'rot_cmd': rot_cmd, 'depth_Z': self.Z }
        try:
            # Ensure log file exists or headers are written
            log_exists = os.path.exists(self.log_file)
            pd.DataFrame([log_data]).to_csv(self.log_file, mode='a', header=not log_exists, index=False)
        except Exception as e: print(f"Log Error: {e}")

        # Print status less frequently to reduce console spam
        if current_time - getattr(self, '_last_print_time', 0) > 0.5:
             print(f"[{current_time:.1f}] St: {status}, Err: {error_norm:.2f}, Z: {self.Z:.2f}, Cmds: X={x_cmd}, Y={y_cmd}, Z={z_cmd}, R={rot_cmd}")
             self._last_print_time = current_time

        # Send commands
        self.drone_commander.piloting(x=x_cmd, y=y_cmd, z=z_cmd, z_rot=rot_cmd, dt=dt)
        self.last_command_time = current_time

    def _display_frame(self, frame_data: list) -> None:
        """ Visualizes the frame, detections, reference/current features, and error vectors. """
        original_frame, results = frame_data[0].copy(), frame_data[1]
        plotted_frame = original_frame.copy()
        frame_height, frame_width = plotted_frame.shape[:2]
        frame_dimensions = (frame_width, frame_height)
        target_lost, current_features = True, None # Store 5x2 array

        # --- Draw Reference Features ( Magenta ) ---
        ref_color = (255, 0, 255)
        ref_text_y = 20
        if self.ref_features_s is not None:
            ref_features = self.ref_features_s.reshape((-1, 2)).astype(int)
            ref_corners = ref_features[:4, :]
            ref_center = tuple(ref_features[4, :])
            cv2.polylines(plotted_frame, [ref_corners], isClosed=True, color=ref_color, thickness=1)
            cv2.circle(plotted_frame, ref_center, 3, ref_color, -1)
            # Display first corner and center coords
            ref_text = f"Ref(TL,C): ({ref_corners[0][0]},{ref_corners[0][1]}), ({ref_center[0]},{ref_center[1]})"
            cv2.putText(plotted_frame, ref_text, (10, ref_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ref_color, 1, cv2.LINE_AA)
            ref_text_y += 15 # Move next text line down
        else:
            cv2.putText(plotted_frame, "Ref: None", (10, ref_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ref_color, 1, cv2.LINE_AA)
            ref_text_y += 15

        # --- Extract and Draw Current Detection & Features ( Green ) ---
        box_color = (0, 255, 0) # Green
        curr_text_y = ref_text_y
        current_features_s = None # Keep track of flattened current features

        has_boxes = results and results.boxes and hasattr(results.boxes, 'conf') and len(results.boxes.conf) > 0
        if has_boxes:
            target_lost = False
            best_conf_idx = results.boxes.conf.argmax()
            coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, 2)
            # Get current features
            current_features = self._get_features_from_box((x1, y1, x2, y2)) # 5x2 float array
            current_features_s = current_features.flatten() # Store flattened version
            current_corners_int = current_features[:4, :].astype(int)
            current_center_int = tuple(current_features[4, :].astype(int))

            # Draw current corners/center
            for i in range(current_corners_int.shape[0]):
                cv2.circle(plotted_frame, tuple(current_corners_int[i]), 5, box_color, -1)
                # cv2.circle(plotted_frame, tuple(current_corners_int[i]), 7, (255,255,255), 1)
            cv2.circle(plotted_frame, current_center_int, 5, box_color, -1)
            cv2.drawMarker(plotted_frame, current_center_int, (255,255,255), cv2.MARKER_TILTED_CROSS, 10, 1)

            # Display current feature coords
            curr_text = f"Curr(TL,C): ({current_corners_int[0][0]},{current_corners_int[0][1]}), ({current_center_int[0]},{current_center_int[1]})"
            cv2.putText(plotted_frame, curr_text, (10, curr_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

            # --- Draw Error Arrows ( Red ) ---
            if self.ref_features_s is not None:
                 ref_features_int = self.ref_features_s.reshape((-1, 2)).astype(int)
                 current_features_int = current_features.astype(int)
                 for i in range(current_features_int.shape[0]): # Iterate through all 5 points
                    p_ref = tuple(ref_features_int[i])
                    p_curr = tuple(current_features_int[i])
                    cv2.arrowedLine(plotted_frame, p_ref, p_curr, (0, 0, 255), 1, tipLength=0.2) # Red error arrow
        else:
             cv2.putText(plotted_frame, "Curr: None", (10, curr_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)


        # --- Generate commands ---
        # Pass necessary info. Note: `_generate_follow_command` modifies processor state (e.g., reference_captured)
        # We pass the already extracted `current_features` and `current_features_s`
        # to avoid recalculating them inside _generate_follow_command.
        # This requires modifying _generate_follow_command signature slightly if we want to pass them.
        # For now, let _generate_follow_command re-extract them based on detection_results.
        self._generate_follow_command(frame_dimensions, results) # Pass results object

        # --- Display Status Text ---
        # Status text based on processor state
        if not self.reference_captured:
            status_text = "AWAITING REF"
            text_color = (255, 255, 0) # Cyan
        elif target_lost: # target_lost is determined during drawing based on current frame
            status_text = "LOST"
            text_color = (0, 0, 255) # Red
        else:
            status_text = "TRACKING (REF)"
            text_color = (0, 255, 0) # Green

        status_font_scale, status_font_thickness, status_font = 0.7, 2, cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(status_text, status_font, status_font_scale, status_font_thickness)
        padding = 5
        rect_y1 = frame_height - h - padding * 3
        text_y = frame_height - padding * 2
        cv2.rectangle(plotted_frame, (padding, rect_y1), (padding + w + padding*2, frame_height - padding), (0,0,0), -1)
        cv2.putText(plotted_frame, status_text, (padding*2, text_y), status_font, status_font_scale, text_color, status_font_thickness, cv2.LINE_AA)

        cv2.imshow("IBVS Jacobian View", plotted_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image-Based Visual Servoing using Jacobian and reference features.")
    # Drone/Connection Arguments
    parser.add_argument("--ip", type=str, default="wireless", choices=["wireless", "simulated", "cable"],
                        help="IP address selection for the drone.")
    parser.add_argument("--init", action='store_true',
                        help="Run initialization sequence (takeoff, position).")

    # Initialization Arguments
    parser.add_argument("--init_gimbal", type=float, default=-45.0, help="Initial gimbal pitch angle (degrees). Used for Z estimate.")
    parser.add_argument("--init_height", type=float, default=1.5, help="Initial takeoff/hover height (meters). Used for Z estimate.")
    parser.add_argument("--init_back", type=float, default=3.0, help="Initial distance behind the target (meters).")

    # IBVS/Processor Arguments
    parser.add_argument("--model", type=str, default="models/yolov11n_best_car_simulator.pt",
                        help="Path to YOLO detection model.")
    parser.add_argument("--target_width", type=float, default=0.25,
                        help="Target width ratio (used for reference scale visualization).") # Modified help text
    parser.add_argument("--lambda", type=float, default=0.4, dest='lambda_gain', help="IBVS gain lambda.")
    parser.add_argument("--depth", type=float, default=None, dest='assumed_depth',
                        help="Override automatically calculated depth Z (meters). If not set, Z is estimated from init height/gimbal.")
    parser.add_argument("--confidence", type=float, default=0.5, 
                      help="Confidence threshold for detection (0.0-1.0).")
    parser.add_argument("--smoothing", type=float, default=0.7,
                      help="Feature smoothing factor (0.0-1.0, higher = less smoothing).")

    # Camera Intrinsics Arguments
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Focal length fx (pixels).")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="Focal length fy (pixels).")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="Principal point cx (pixels).")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="Principal point cy (pixels).")

    args = parser.parse_args()

    # Select IP
    if args.ip == "wireless": ip_selected = DroneIp.WIRELESS
    elif args.ip == "simulated": ip_selected = DroneIp.SIMULATED
    elif args.ip == "cable": ip_selected = DroneIp.CABLE
    else: raise ValueError("Invalid IP address selected")

    # Prepare processor kwargs, including init params needed for Z
    processor_kwargs = {
        "model_path": args.model,
        "target_width_ratio": args.target_width,
        "lambda_gain": args.lambda_gain,
        "assumed_depth": args.assumed_depth, # Pass user override (can be None)
        "fx": args.fx,
        "fy": args.fy,
        "cx": args.cx,
        "cy": args.cy,
        "init_height": args.init_height, # Pass init value
        "init_gimbal": args.init_gimbal, # Pass init value
        "confidence_threshold": args.confidence, # Add new params
        "feature_smoothing_alpha": args.smoothing, # Add new params
    }

    # Create Controller
    controller = IBVSJacobianController(
        ip=ip_selected,
        processor_class=IBVSJacobianProcessor,
        **processor_kwargs
    )

    # Run Initialization if requested
    if args.init:
        # Use the same args for the actual init maneuver
        controller.initialize_position(
            takeoff_height=args.init_height,
            gimbal_angle=args.init_gimbal,
            back_distance=args.init_back,
        )

    # Start the main loop
    try:
        print("Starting controller run loop...")
        controller.run()
    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        # Ensure drone lands, resources are released etc.
        print("Cleaning up...")
        if hasattr(controller, 'drone_commander') and controller.drone_commander is not None:
             print("Landing drone...")
             controller.drone_commander.land()
             time.sleep(2)
        if hasattr(controller, 'stop'):
             print("Stopping controller threads...")
             controller.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete.") 