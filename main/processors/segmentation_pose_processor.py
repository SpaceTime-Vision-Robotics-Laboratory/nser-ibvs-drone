import cv2
import numpy as np
import time # Added for PID
from drone_base.main.stream.base_video_processor import BaseVideoProcessor # Import base class

# Simple PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 1e-3 # Avoid division by zero

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return output

# Inherit from BaseVideoProcessor
class SegmentationPoseProcessor(BaseVideoProcessor):
    def __init__(self, frame_width=960, frame_height=720, target_height_ratio=0.3, **kwargs):
        # Call the parent constructor FIRST
        super().__init__(**kwargs)

        self.logger.info("Initializing SegmentationPoseProcessor...") # Use logger from base class
        self.frame_width = self.config.width # Use width from config passed by base class
        self.frame_height = self.config.height # Use height from config passed by base class

        # --- Placeholder for Model Loading ---
        self.logger.info("TODO: Load actual segmentation model here.") # Use logger
        self.segmentation_model = None # Replace with actual model loading
        # ---

        # --- Pose Estimation / Control Setup ---
        # Target position (center of the frame)
        self.target_x = self.frame_width / 2
        self.target_y = self.frame_height / 2
        # Target size (relative height in the frame as proxy for distance)
        self.target_bbox_height = self.frame_height * target_height_ratio

        # PID controllers for smooth movement
        # Gains need tuning based on drone response and environment
        self.pid_x = PIDController(Kp=0.3, Ki=0.01, Kd=0.1, setpoint=self.target_x) # Yaw/Rotation control
        self.pid_y = PIDController(Kp=0.2, Ki=0.01, Kd=0.05, setpoint=self.target_y) # Up/Down control
        self.pid_z = PIDController(Kp=0.4, Ki=0.02, Kd=0.1, setpoint=self.target_bbox_height) # Forward/Backward control

        self.last_detection = None # Store last known position if target is lost
        self.lost_counter = 0 # Count frames since target was lost

    def _placeholder_segmentation(self, frame: np.ndarray):
        """ Placeholder for actual segmentation. Returns dummy data. """
        # In reality, run self.segmentation_model.predict(frame)
        # This should return a list of detections, each with class, confidence, mask, bbox
        # Example dummy output: Find brightest region and call it a 'car'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100: # Minimum size
                x, y, w, h = cv2.boundingRect(largest_contour)
                detections.append({
                    'class': 'car', # Assume it's a car
                    'confidence': 0.9,
                    'bbox': [x, y, x + w, y + h], # x1, y1, x2, y2
                    'mask': None # Mask data not used in this simple version
                })
        return detections

    # This is the core method required by BaseVideoProcessor
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame:
        1. Perform segmentation.
        2. Find the target.
        3. Estimate pose (centroid, bbox height).
        4. Calculate and send control commands.
        5. Annotate and return the frame for display.
        Args:
            frame: The input video frame (NumPy array).
        Returns:
            The annotated video frame (NumPy array).
        """
        # --- 1. Perform Segmentation (Placeholder) ---
        if self.segmentation_model:
            detections = self.segmentation_model.predict(frame) # Actual model
        else:
            detections = self._placeholder_segmentation(frame) # Placeholder

        # --- 2. Find Target (Simplified: Use first detection if any) ---
        target = detections[0] if detections else None
        detection_info = None # Store info if target found

        # --- 3. Estimate Pose ---
        if target:
            self.lost_counter = 0 # Reset lost counter
            x1, y1, x2, y2 = target['bbox']
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            bbox_height = y2 - y1
            detection_info = {'bbox': [x1, y1, x2, y2], 'center': (bbox_center_x, bbox_center_y), 'height': bbox_height}
            self.last_detection = detection_info # Update last known detection

            # --- 4. Calculate & Send Control Commands ---
            control_commands = self.calculate_control(bbox_center_x, bbox_center_y, bbox_height)
            self._send_commands(control_commands)
        else:
            # Target lost
            self.lost_counter += 1
            self.last_detection = None
            if self.lost_counter > 30: # Log if lost for a while
                self.logger.warning("Target lost, hovering.")
            # Send zero commands (hover)
            self._send_commands({'forward': 0.0, 'right': 0.0, 'down': 0.0, 'rotation': 0.0})

        # --- 5. Annotate and Return Frame ---
        annotated_frame = self.get_annotated_frame(frame, detection_info) # Use the raw frame
        return annotated_frame

    def calculate_control(self, current_x, current_y, current_height):
        """
        Calculate drone movement commands based on the estimated pose.
        Args:
            current_x: Current horizontal center of the target.
            current_y: Current vertical center of the target.
            current_height: Current bounding box height of the target.
        Returns:
            A dictionary with control commands.
        """
        # Update PIDs
        # Note: Signs might need inversion depending on drone's move_by convention
        rotation_speed = -self.pid_x.update(current_x)
        down_speed = self.pid_y.update(current_y)
        forward_speed = self.pid_z.update(current_height)

        # Clamp outputs to reasonable limits (e.g., -1 to 1, scaled later by speed factor)
        # These limits need tuning!
        max_rot_speed = 1.0
        max_vert_speed = 0.5
        max_fwd_speed = 0.8

        rotation_speed = np.clip(rotation_speed, -max_rot_speed, max_rot_speed)
        down_speed = np.clip(down_speed, -max_vert_speed, max_vert_speed)
        forward_speed = np.clip(forward_speed, -max_fwd_speed, max_fwd_speed)

        commands = {
            'forward': forward_speed,
            'right': 0.0, # Using rotation for horizontal control
            'down': down_speed,
            'rotation': rotation_speed
        }
        # print(f"Cmds: F:{forward_speed:.2f}, D:{down_speed:.2f}, R:{rotation_speed:.2f} | Target H:{self.target_bbox_height:.1f}, Curr H:{current_height:.1f}")
        return commands

    def _send_commands(self, control_commands):
        """ Sends commands to the drone commander. """
        # NOTE: Assumes BaseVideoProcessor provides self.drone_commander
        # Scaling might be needed based on speed settings
        if self.drone_commander:
            self.drone_commander.move_by_velocity(
                forward=control_commands['forward'],
                right=control_commands['right'],
                down=control_commands['down'],
                rotation=control_commands['rotation']
            )
        else:
             self.logger.warning("Drone commander not available in processor.")

    def get_annotated_frame(self, frame, detection_info):
        """
        Optional: Annotate the frame with segmentation/pose visualization.
        Args:
            frame: The original frame.
            detection_info: The detection info dict or None.
        Returns:
            The annotated frame.
        """
        annotated_frame = frame.copy()
        # detection_info = results # Results from process_frame is (frame, detection_info) # Old comment

        if detection_info:
            x1, y1, x2, y2 = map(int, detection_info['bbox'])
            center_x, center_y = map(int, detection_info['center'])
            height = detection_info['height']

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw centroid
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            # Display height info
            cv2.putText(annotated_frame, f"H: {height:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw target crosshairs
        tx, ty = int(self.target_x), int(self.target_y)
        cv2.line(annotated_frame, (tx - 20, ty), (tx + 20, ty), (255, 0, 0), 1)
        cv2.line(annotated_frame, (tx, ty - 20), (tx, ty + 20), (255, 0, 0), 1)

        return annotated_frame

    def stop(self):
        """
        Clean up resources when stopping.
        """
        self.logger.info("Stopping SegmentationPoseProcessor...")
        # TODO: Release any resources (models, etc.) if loaded
        if self.segmentation_model:
            # Add cleanup for your specific model if needed
            pass
        super().stop() # Call parent stop method
        self.logger.info("SegmentationPoseProcessor stopped.") 