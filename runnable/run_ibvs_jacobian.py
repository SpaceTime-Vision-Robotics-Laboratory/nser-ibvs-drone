import argparse
import sys
from pathlib import Path
import ultralytics
import pandas as pd
import os
import time
import numpy as np
import cv2

# Modify paths to include drone_base
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor

# Default Camera Intrinsics (Example - REPLACE with actual values for your drone camera)
# These might be found in camera documentation or estimated via calibration.
# Values are in pixels.
DEFAULT_FX = 700.0  # Focal length in x
DEFAULT_FY = 700.0  # Focal length in y
DEFAULT_CX = 320.0  # Principal point x-coordinate (assuming 640 width)
DEFAULT_CY = 360.0  # Principal point y-coordinate (assuming 720 height)


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
        """ Estimates depth Z based on init params or uses override. """
        if self.override_depth_Z is not None:
            if self.override_depth_Z > 0:
                # print(f"Using overridden depth Z = {self.override_depth_Z:.2f}m")
                return self.override_depth_Z
            else:
                print(f"Warning: Invalid override depth Z ({self.override_depth_Z}). Calculating from init params.")

        if self.init_height is None or self.init_gimbal is None:
            print("Warning: init_height or init_gimbal not available for Z calculation. Using default Z=3.0m.")
            return 3.0

        # Calculate Z using trigonometry: Z = height / sin(abs(angle))
        gimbal_rad = np.radians(self.init_gimbal)
        sin_gimbal = np.sin(abs(gimbal_rad))

        if abs(sin_gimbal) < 0.05: # Avoid division by zero/instability near 0 degrees
            print(f"Warning: Gimbal angle ({self.init_gimbal:.1f} deg) too close to zero for Z calculation. Using default Z=3.0m.")
            return 3.0
        else:
            calculated_Z = self.init_height / sin_gimbal
            # print(f"Calculated depth Z = {calculated_Z:.2f}m from height={self.init_height:.2f}m, gimbal={self.init_gimbal:.1f}deg")
            return calculated_Z

    def _generate_follow_command(self, frame_dimensions, detection_results):
        """ Calculates drone velocity commands using Jacobian-based IBVS. """
        current_time = time.time()
        dt = max(0.01, current_time - self.last_command_time) # Ensure dt > 0

        # --- Calculate/Update Depth Z --- 
        # Do this calculation once, or maybe periodically if Z estimate could change
        if self.Z is None:
            self.Z = self._calculate_depth_Z()
            print(f"Estimated/Set Depth Z = {self.Z:.2f}m for Jacobian calculations.")
        # --------------------------------- 

        frame_width, frame_height = frame_dimensions
        current_features_s = None
        best_box = None
        target_lost = True
        status = "Lost/Hover"
        has_boxes = detection_results and detection_results.boxes and hasattr(detection_results.boxes, 'conf') and len(detection_results.boxes.conf) > 0
        if has_boxes:
            best_conf_idx = detection_results.boxes.conf.argmax()
            target_lost = False
            coords = detection_results.boxes.xyxy[best_conf_idx].cpu().numpy()
            best_box = tuple(map(int, coords))
            current_features = self._get_features_from_box(best_box)
            current_features_s = current_features.flatten()

        camera_velocity_vc = np.zeros(6)
        x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0
        error_norm = 0.0

        if not target_lost and current_features_s is not None and self.target_features_s_star is not None and self.Z is not None:
            status = "Tracking"
            error_e = current_features_s - self.target_features_s_star
            error_norm = np.linalg.norm(error_e)
            L_s = self._compute_interaction_matrix(current_features, self.Z)

            if L_s is not None:
                try:
                    damping_factor = 0.01
                    L_s_pinv = np.linalg.inv(L_s.T @ L_s + damping_factor * np.identity(6)) @ L_s.T
                    camera_velocity_vc = -self.lambda_gain * L_s_pinv @ error_e
                except np.linalg.LinAlgError:
                    status = "Jacobian Singularity"
                    camera_velocity_vc = np.zeros(6)
            else:
                status = "Invalid Depth Z"
                camera_velocity_vc = np.zeros(6)

            # --- Command Mapping (Reverted to Standard Convention) --- 
            v_x, v_y, v_z, w_x, w_y, w_z = camera_velocity_vc
            vel_scale_linear = 75.0
            vel_scale_angular = 90.0
            max_raw_vel = 1.0
            # Clip raw velocities first
            v_x, v_y, v_z, w_z = np.clip([v_x, v_y, v_z, w_z], -max_raw_vel, max_raw_vel)
            
            # Standard Mapping: Cam(X:fwd, Y:right, Z:down) -> Drone(x:right, y:fwd, z:up)
            x_cmd = int(np.clip(vel_scale_linear * (+v_x), -50, 50))    # Lateral (Camera Y -> Drone X)
            y_cmd = int(np.clip(vel_scale_linear * (v_z), -50, 50))    # Forward (Camera X -> Drone Y)
            # z_cmd = int(np.clip(vel_scale_linear * (-v_z), -50, 50))    # Altitude (Camera -Z -> Drone Z)
            rot_cmd = int(np.clip(vel_scale_angular * (+v_y), -70, 70)) # Yaw (Camera Z -> Drone Yaw)
            # --- End Command Mapping ---
        else:
            status = "Lost/No Target" if target_lost else "Initializing"
            x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0

        # --- Logging ---\
        log_data = { 'timestamp': current_time, 'status': status, 'error_norm': error_norm,
                     'vc_vx': camera_velocity_vc[0], 'vc_vy': camera_velocity_vc[1], 'vc_vz': camera_velocity_vc[2],
                     'vc_wx': camera_velocity_vc[3], 'vc_wy': camera_velocity_vc[4], 'vc_wz': camera_velocity_vc[5],
                     'x_cmd': x_cmd, 'y_cmd': y_cmd, 'z_cmd': z_cmd, 'rot_cmd': rot_cmd, 'depth_Z': self.Z }
        try:
            pd.DataFrame([log_data]).to_csv(self.log_file, mode='a', header=not os.path.exists(self.log_file), index=False)
        except Exception as e: print(f"Log Error: {e}")

        print(f"[{current_time:.2f}] Status: {status}, ErrNorm: {error_norm:.2f}, Z: {self.Z:.2f}")
        print(f"[{current_time:.2f}] Cmds: X={x_cmd}, Y={y_cmd}, Z={z_cmd}, Rot={rot_cmd}")
        self.drone_commander.piloting(x=x_cmd, y=y_cmd, z=z_cmd, z_rot=rot_cmd, dt=dt)
        self.last_command_time = current_time

    def _display_frame(self, frame_data: list) -> None:
        """ Visualizes the frame, detections, target features, and current features. """
        original_frame = frame_data[0].copy()
        results = frame_data[1] # Ultralytics results object
        plotted_frame = original_frame.copy()

        frame_height, frame_width = plotted_frame.shape[:2]
        frame_dimensions = (frame_width, frame_height)

        target_lost = True
        current_features = None # Store 5x2 array

        # Draw Target Features (s*) if defined
        target_color = (255, 0, 255) # Magenta
        if self.target_features_s_star is not None:
            # Reshape s* back into 5x2 features
            target_features = self.target_features_s_star.reshape((-1, 2)).astype(int)
            target_corners = target_features[:4, :] # First 4 are corners
            target_center = tuple(target_features[4, :]) # Last one is center
            # Draw the target rectangle (from corners)
            cv2.polylines(plotted_frame, [target_corners], isClosed=True, color=target_color, thickness=1)
            # Draw target corner circles
            # for i in range(target_corners.shape[0]):
            #     cv2.circle(plotted_frame, tuple(target_corners[i]), 3, target_color, -1)
            # Draw target center circle
            cv2.circle(plotted_frame, target_center, 5, target_color, -1)
            cv2.drawMarker(plotted_frame, target_center, (255,255,255), cv2.MARKER_CROSS, 10, 1)


        # Extract and Draw Current Best Detection and Features (s)
        box_color = (0, 255, 0) # Green
        # Check if results are valid and contain boxes
        has_boxes = results and results.boxes and hasattr(results.boxes, 'conf') and len(results.boxes.conf) > 0

        if has_boxes:
            target_lost = False
            best_conf_idx = results.boxes.conf.argmax()
            coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            # Draw bounding box
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, 2)
            # Get current corners + center
            current_features = self._get_features_from_box((x1, y1, x2, y2)) # 5x2 float array
            current_corners_int = current_features[:4, :].astype(int)
            current_center_int = tuple(current_features[4, :].astype(int))

            # Draw current corners
            for i in range(current_corners_int.shape[0]):
                cv2.circle(plotted_frame, tuple(current_corners_int[i]), 5, box_color, -1)
                cv2.circle(plotted_frame, tuple(current_corners_int[i]), 7, (255,255,255), 1) # White outline
            # Draw current center
            cv2.circle(plotted_frame, current_center_int, 5, box_color, -1)
            cv2.drawMarker(plotted_frame, current_center_int, (255,255,255), cv2.MARKER_TILTED_CROSS, 10, 1)

            # Draw lines connecting current features to target features (error visualization)
            if self.target_features_s_star is not None:
                 target_features_int = self.target_features_s_star.reshape((-1, 2)).astype(int)
                 current_features_int = current_features.astype(int)
                 for i in range(current_features_int.shape[0]): # Iterate through all 5 points
                    p1 = tuple(current_features_int[i])
                    p2 = tuple(target_features_int[i])
                    cv2.line(plotted_frame, p1, p2, (0, 255, 255), 1) # Yellow line

        # Generate commands (needs detection results)
        # Call _generate_follow_command here, it will handle the logic based on target_lost
        self._generate_follow_command(frame_dimensions, results) # Pass results object

        # Display Status Text (Bottom Left)
        # Status is determined within _generate_follow_command now, fetch it for display?
        # For simplicity, use target_lost status determined during drawing
        status_text = "TRACKING" if not target_lost else "NO TARGET"
        text_color = (0, 255, 0) if not target_lost else (0, 0, 255)
        status_font_scale = 0.7
        status_font_thickness = 2
        status_font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(status_text, status_font, status_font_scale, status_font_thickness)
        padding = 5
        rect_x1 = padding
        rect_y1 = frame_height - h - padding * 3
        rect_x2 = padding + w + padding * 2
        rect_y2 = frame_height - padding
        text_x = padding * 2
        text_y = frame_height - padding * 2
        cv2.rectangle(plotted_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(plotted_frame, status_text, (text_x, text_y), status_font, status_font_scale, text_color, status_font_thickness, cv2.LINE_AA)

        # Display the final frame
        cv2.imshow("IBVS Jacobian View", plotted_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image-Based Visual Servoing using Jacobian.")
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
    parser.add_argument("--target_width", type=float, default=0.25, help="Target width ratio.")
    parser.add_argument("--lambda", type=float, default=0.4, dest='lambda_gain', help="IBVS gain lambda.")
    parser.add_argument("--depth", type=float, default=None, dest='assumed_depth',
                        help="Override automatically calculated depth Z (meters). If not set, Z is estimated from init height/gimbal.")

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
             # Check connection before landing
             # if controller.drone.connection_state(): # Requires drone object access
             controller.drone_commander.land()
             time.sleep(2) # Allow time for land command
             # Optional: disconnect
             # controller.drone_commander.disconnect()

        if hasattr(controller, 'stop'):
             print("Stopping controller threads...")
             controller.stop() # Assuming base controller has a stop method
        cv2.destroyAllWindows()
        print("Cleanup complete.") 