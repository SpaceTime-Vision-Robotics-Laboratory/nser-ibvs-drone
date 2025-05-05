from pathlib import Path
import sys
import time
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

import numpy as np
import cv2
from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor
from ultralytics import YOLO

# --- Constants ---
# Camera intrinsics (Placeholder - Replace with actual values)
camera_fx = 465.0
camera_fy = 348.0
camera_cx = 320.0
camera_cy = 180.0
camera_matrix = np.array([[camera_fx, 0, camera_cx],
                          [0, camera_fy, camera_cy],
                          [0, 0, 1]])
dist_coeffs = np.zeros(4) # Assuming no distortion for simplicity

# Control Parameters
lambda_gain = 0.5 # PBVS gain
Z_desired = 4.0  # Desired distance from the object (meters)

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# --- Controller ---
class PBVSController(BaseStreamingController):
    def __init__(self, lambda_gain:float = lambda_gain, **kwargs):
        processor_kwargs = {'lambda_gain': lambda_gain, 'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs, 'Z_desired': Z_desired}
        super().__init__(processor_kwargs=processor_kwargs, **kwargs)
        # Camera params (fx, fy, cx, cy) are now primarily needed by the processor/pose estimator

    def initialize_position(self, takeoff_height=5.0, gimbal_angle=-45, back_distance=-2.0, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        print("Taking off...")
        self.drone_commander.tilt_camera(pitch_deg=gimbal_angle, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        self.drone_commander.take_off()
        time.sleep(1)

        print(f"Moving to initial position (backward: {-back_distance}m, height: {takeoff_height}m)...")
        self.drone_commander.move_by(forward=back_distance, right=left_distance, down=-takeoff_height, rotation=0)
        time.sleep(1)

        print(f"Starting Pose-Based Visual Servoing Program...")

# --- Processor ---
class PBVSProcessor(BaseVideoProcessor):
    def __init__(self, lambda_gain: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, Z_desired: float,
                 model_path: str = "/home/sebnae/shared_drive/ws/drone_ws/auto-follow/models/best__yolo11n-seg__seg__27_04__sim.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = YOLO(model_path)
        self.lambda_gain = lambda_gain
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.t_star = np.array([0, 0, Z_desired]) # Target position: Centered, Z_desired distance
        self.theta_u_star = np.array([0, 0, 0])   # Target orientation: Zero rotation
        self.assumed_size = 1.0 # <<< IMPORTANT ASSUMPTION: Real-world size (e.g., meters) of the object face represented by minAreaRect. Affects distance estimation.

    # !!! Placeholder Function: Replace with actual pose estimation logic !!!
    # Updated implementation using minAreaRect and solvePnP with assumptions
    def estimate_pose_from_segmentation(self, segmentation_result, frame) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not (segmentation_result and segmentation_result.masks and segmentation_result.masks.xy):
            self.logger.info("No segmentation mask found.")
            return None, None

        mask_contours = segmentation_result.masks.xy
        if not mask_contours or len(mask_contours) == 0:
            self.logger.info("Empty mask contours.")
            return None, None

        # --- Find the largest contour ---
        # Assuming the largest contour is the target
        contours = [np.array(c, dtype=np.int32) for c in mask_contours]
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 100: # Filter out tiny contours
             self.logger.info("Largest contour area too small.")
             return None, None

        # --- Get Minimum Area Rotated Rectangle --- (Requires >= 4 points)
        if len(largest_contour) < 4:
             self.logger.info("Not enough points in contour for minAreaRect.")
             return None, None

        try:
            rect = cv2.minAreaRect(largest_contour)
            box_points_2d = cv2.boxPoints(rect) # Order: bottom-left, top-left, top-right, bottom-right (typically)
        except cv2.error as e:
             self.logger.warning(f"Error getting minAreaRect or boxPoints: {e}")
             return None, None

        # --- Define 3D Object Points (Based on Assumed Size) ---
        # Assume the box_points_2d correspond to a planar face (e.g., square) in the object's frame.
        # We define the 3D points in the object's coordinate system (e.g., Z=0 plane).
        half_size = self.assumed_size / 2.0
        # IMPORTANT: The order MUST match the order returned by cv2.boxPoints for solvePnP.
        # Let's define a standard object frame: Origin at center, +X right, +Y up, +Z out of plane
        # Assuming boxPoints order is BL, TL, TR, BR corresponding to object frame coords:
        # (-half_size, -half_size, 0), (-half_size, half_size, 0), (half_size, half_size, 0), (half_size, -half_size, 0)
        obj_points_3d = np.array([
            [-half_size, -half_size, 0.0], # Corresponds to box_points_2d[0]
            [-half_size,  half_size, 0.0], # Corresponds to box_points_2d[1]
            [ half_size,  half_size, 0.0], # Corresponds to box_points_2d[2]
            [ half_size, -half_size, 0.0]  # Corresponds to box_points_2d[3]
        ], dtype=np.float32)

        image_points = box_points_2d.astype(np.float32) # Ensure float32 for solvePnP

        # --- Use solvePnP --- (Requires >= 4 points)
        if len(image_points) < 4:
             return None, None # Should not happen if minAreaRect succeeded

        try:
            # Use iterative PnP method like SOLVEPNP_ITERATIVE or SOLVEPNP_IPPE
            # SOLVEPNP_IPPE can be good for planar objects, might return multiple solutions.
            # Let's start with the default iterative method.
            success, rvec, tvec = cv2.solvePnP(
                obj_points_3d, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                t = tvec.flatten()       # Translation vector [tx, ty, tz]
                theta_u = rvec.flatten() # Rotation vector (axis-angle) [thetau_x, thetau_y, thetau_z]
                # Optional: Check reprojection error if needed
                return t, theta_u
            else:
                self.logger.warning("solvePnP failed to find a solution.")
                return None, None

        except cv2.error as e:
            self.logger.error(f"cv2.solvePnP error: {e}", exc_info=True)
            return None, None
        except Exception as e:
             self.logger.error(f"Unexpected error during pose estimation: {e}", exc_info=True)
             return None, None

    def compute_control_law(self, error_vector:np.ndarray, interaction_matrix:np.ndarray) -> np.ndarray:
        # u = -lambda * L+ * e
        L = interaction_matrix # Shape (6, 6) for pose
        lambda_gain = self.lambda_gain

        try:
            if L.size == 0 or np.linalg.matrix_rank(L) < 6: # Check for rank deficiency
                 self.logger.warning(f"Interaction matrix is empty or rank deficient (rank {np.linalg.matrix_rank(L)}).")
                 return np.zeros(6)

            # Use Moore-Penrose pseudo-inverse (pinv) or standard inverse if rank is full
            L_pseudo_inv = np.linalg.pinv(L) # More robust
            # L_inv = np.linalg.inv(L) # Faster if always full rank

            u = -lambda_gain * L_pseudo_inv @ error_vector # Shape (6,)
            return u
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"Singular matrix encountered during pseudo-inverse calculation: {e}")
            return np.zeros(6) # Return zero velocity if calculation fails

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=True, verbose=False)
        first_result = next(results, None)

        # Initialize
        control_law_camera_frame = np.zeros(6)
        error_vector_display = np.zeros(6) # For logging
        plotted_frame = frame.copy()
        axis_length = 0.1 # Length of axis lines for visualization (in meters)

        # --- Estimate Pose ---
        t_current, theta_u_current = self.estimate_pose_from_segmentation(first_result, frame)

        if t_current is not None and theta_u_current is not None:
            # --- Calculate Pose Error ---
            error_t = t_current - self.t_star
            # Simple orientation error (for small angles); more robust methods exist
            error_theta_u = theta_u_current - self.theta_u_star
            error_vector = np.concatenate((error_t, error_theta_u)) # Shape (6,)
            error_vector_display = error_vector # Store for logging

            # --- Calculate Interaction Matrix (L_pose) ---
            # L_pose = [ -I_3x3   skew(t_current) ]
            #          [  0_3x3      -I_3x3      ]
            L_pose = np.zeros((6, 6))
            L_pose[0:3, 0:3] = -np.identity(3)
            L_pose[0:3, 3:6] = skew_symmetric(t_current)
            L_pose[3:6, 3:6] = -np.identity(3)

            # --- Compute Control Law ---
            control_law_camera_frame = self.compute_control_law(error_vector, L_pose)

            # --- Visualization ---
            # Project 3D axes onto the image
            try:
                # Convert axis-angle (theta_u) back to rotation vector (rvec) for projectPoints
                rvec = theta_u_current.reshape(3, 1)
                tvec = t_current.reshape(3, 1)
                axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
                imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                imgpts = imgpts.reshape(-1, 2).astype(int)

                # Draw axes lines
                origin = tuple(imgpts[0])
                cv2.line(plotted_frame, origin, tuple(imgpts[1]), (0,0,255), 2) # X (Red)
                cv2.line(plotted_frame, origin, tuple(imgpts[2]), (0,255,0), 2) # Y (Green)
                cv2.line(plotted_frame, origin, tuple(imgpts[3]), (255,0,0), 2) # Z (Blue)
            except (cv2.error, ValueError) as e:
                self.logger.warning(f"Error projecting axes for visualization: {e}")

        else:
            self.logger.info("Pose not estimated for this frame.")


        # --- Apply Control Law ---
        vx, vy, vz, wx, wy, wz = control_law_camera_frame

        MAX_COMMAND = 30 # Limit command values
        # --- Mapping (adjust signs and axes based on drone/camera frames) ---
        # Direct mapping of camera velocities to drone commands
        K_trans = 1  # Translational gain (adjust based on error units and desired speed)
        K_rot = 1    # Rotational gain (adjust based on error units and desired speed)

        # Tentative Mapping (needs verification based on coordinate systems)
        roll_cmd = int(np.clip(vx * K_trans, -MAX_COMMAND, MAX_COMMAND))  # Camera X -> Drone Roll ?
        pitch_cmd = int(np.clip(vy * K_trans, -MAX_COMMAND, MAX_COMMAND)) # Camera Y -> Drone Pitch ? (often inverted)
        gaz_cmd = int(np.clip(-vz * K_trans, -MAX_COMMAND, MAX_COMMAND))   # Camera Z -> Drone Gaz ? (often inverted)
        yaw_rate_cmd = int(np.clip(-wz * K_rot, -MAX_COMMAND, MAX_COMMAND)) # Camera Z rot -> Drone Yaw ?

        # Log commands and errors
        err_t_str = f"[{error_vector_display[0]:.2f} {error_vector_display[1]:.2f} {error_vector_display[2]:.2f}]"
        err_r_str = f"[{error_vector_display[3]:.2f} {error_vector_display[4]:.2f} {error_vector_display[5]:.2f}]"
        print(f'Err T:{err_t_str}, R:{err_r_str} -> Cmd: R:{roll_cmd}, P:{pitch_cmd}, G:{gaz_cmd}, YR:{yaw_rate_cmd}')

        # Send piloting command
        self.drone_commander.piloting(x=roll_cmd, y=pitch_cmd, z=0, z_rot=yaw_rate_cmd, dt=0.1)

        return plotted_frame

# --- Main Execution ---
if __name__ == "__main__":
    controller = PBVSController(
        ip=DroneIp.SIMULATED, # Or DroneIp.REAL
        processor_class=PBVSProcessor,
        speed=35 # Manual control speed (if keyboard input is used)
        # Add log_path and results_path if needed
        # log_path="./logs",
        # results_path="./results"
    )

    try:
        controller.initialize_position(takeoff_height=4.0, gimbal_angle=-45, back_distance=-6.0) # Start further back for PBVS
        controller.run() # run() handles start() and the main loop
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping controller...")
        controller.stop() # Ensure cleanup happens 