from pathlib import Path
import sys
import time
import argparse
import ultralytics
import numpy as np
import cv2
import pandas as pd
import os
from collections import deque
import math # Add math import for angle calculations
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

# Helper function for angle normalization and difference
def normalize_angle(angle_deg):
    """Normalize angle to be within [-180, 180) degrees."""
    return (angle_deg + 180) % 360 - 180

def shortest_angle_diff(angle1_deg, angle2_deg):
    """Calculate the shortest difference between two angles in degrees."""
    diff = normalize_angle(angle1_deg - angle2_deg)
    return diff

class RealController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation_window_size = 240
        self.motion_vectors_deque = deque(maxlen=self.orientation_window_size)
        self.min_movement_threshold = 2.0 # Pixels
        # --- Bounding Box Angle Estimation ---
        self.smoothed_box_angle = 0.0 # Smoothed angle from minAreaRect (degrees)
        self.angle_smoothing_factor = 0.3 # Alpha for EMA on box angle
        self.mask_warning_printed = False

        # --- Control Gains ---

        self.kp_rot = 20 # Proportional gain for rotation (yaw) based on x_offset
        self.kd_rot = 5  # Derivative gain for rotation (damping) - Currently Unused
        self.kp_alt = 0   # Proportional gain for altitude (z) - Disabled for vertical centering
        self.kp_fwd = -45 # Proportional gain for forward (y) based on y_offset
        self.kp_angle = 20 # Proportional gain for rotation (yaw) based on estimated angle
        self.kp_pitch = 20 # Proportional gain for gimbal pitch speed based on y_offset
        self.max_pitch_speed = 15 # Max gimbal pitch speed (deg/s)

        # --- Thresholds ---
        self.offset_threshold = 0.1 # Deadband for x/y offset corrections

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time() # Initialize last time for dt calc

    def initialize_position(self):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-45, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)
        self.drone_commander.move_by(forward=-2, right=0, down=-2, rotation=0)
        self.frame_processor.frame_queue.empty()
        print("Initialization complete. Ready for tracking.")
        

class CarYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "/home/sebnae/shared_drive/ws/drone_ws/auto-follow/models/yolov11n_best_car_simulator.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = ultralytics.YOLO(model_path)
        self.moved_up = False

        # --- Orientation Estimation ---
        self.previous_center = None
        self.smoothed_orientation_vector = np.array([1.0, 0.0]) # Initial assumption (e.g., pointing right)
        self.orientation_window_size = 240
        self.motion_vectors_deque = deque(maxlen=self.orientation_window_size)
        self.min_movement_threshold = 2.0 # Pixels

        # --- Bounding Box Angle Estimation ---
        self.smoothed_box_angle = 0.0 # Smoothed angle from minAreaRect (degrees)
        self.angle_smoothing_factor = 0.3 # Alpha for EMA on box angle
        self.mask_warning_printed = False

        # --- Control Gains ---

        self.kp_rot = 20 # Proportional gain for rotation (yaw) based on x_offset
        self.kd_rot = 5  # Derivative gain for rotation (damping) - Currently Unused
        self.kp_alt = 0   # Proportional gain for altitude (z) - Disabled for vertical centering
        self.kp_fwd = -45 # Proportional gain for forward (y) based on y_offset
        self.kp_angle = 20 # Proportional gain for rotation (yaw) based on estimated angle
        self.kp_pitch = 20 # Proportional gain for gimbal pitch speed based on y_offset
        self.max_pitch_speed = 15 # Max gimbal pitch speed (deg/s)

        # --- Thresholds ---
        self.offset_threshold = 0.1 # Deadband for x/y offset corrections

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time() # Initialize last time for dt calc

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # frame = cv2.resize(frame, (640, 640))
        # Request masks (parameter might differ based on ultralytics version)
        results = self.detector.predict(frame, stream=False, verbose=False, retina_masks=True)
        # Return the full results object for the first image
        return [frame, results[0] if results else None]

    def _generate_follow_command(self, frame_center, object_center, frame_dimensions,
                                 box_size=None, target_lost=False):
        current_time = time.time()
        # Calculate dt, handle potential first run or zero dt
        # dt = current_time - self.last_command_time # dt no longer needed for yaw
        # if dt <= 0.001: # Avoid division by zero or excessively large derivatives
        #     dt = 0.15 # Assume a nominal dt if issue occurs
        # self.last_command_time = current_time

        frame_center_x, frame_center_y = frame_center
        frame_width, frame_height = frame_dimensions

        # --- Orientation Estimation Logic ---
        # Note: This updates self.smoothed_orientation_vector
        if not target_lost and object_center is not None:
            new_vector_added = False # Flag to check if deque was updated
            current_center = np.array(object_center)
            if self.previous_center is not None:
                motion_vector = current_center - self.previous_center
                motion_magnitude = np.linalg.norm(motion_vector)

                if motion_magnitude > self.min_movement_threshold:
                    # Normalize the motion vector
                    direction_vector = motion_vector / motion_magnitude

                    # Add normalized vector to deque for SMA
                    self.motion_vectors_deque.append(direction_vector)
                    new_vector_added = True

            # Calculate SMA orientation vector from deque if it's not empty
            if self.motion_vectors_deque:
                sum_vector = np.sum(list(self.motion_vectors_deque), axis=0)
                sum_norm = np.linalg.norm(sum_vector)
                if sum_norm > 1e-6: # Avoid division by zero/near-zero
                    self.smoothed_orientation_vector = sum_vector / sum_norm
                # Else: Keep the previous smoothed vector if sum is negligible

            # Update previous center for the next frame
            self.previous_center = current_center
        else:
            # Reset previous center if target is lost
            self.previous_center = None

        # Initialize commands to zero (hover)
        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0
        # Initialize state variables for logging/debugging
        x_offset, y_offset = 0.0, 0.0
        angle_rot_term = 0 # Initialize angle-based rotation term
        proportional_rot_term = 0
        pitch_speed = 0 # Initialize gimbal pitch speed command
        # Calculate state and commands ONLY if target is NOT lost
        
        if not target_lost and object_center is not None and box_size is not None:
            object_center_x, object_center_y = object_center
            x_offset = (object_center_x - frame_center_x) / (frame_width / 2) if frame_width > 0 else 0
            y_offset = (object_center_y - frame_center_y) / (frame_height / 2) if frame_height > 0 else 0

            # --- PD Control for Rotation ---
            # --- Yaw Control (Centering + Angle Alignment) ---
            proportional_rot_term = 0
            if abs(x_offset) > self.offset_threshold:
                proportional_rot_term = self.kp_rot * x_offset

            # Calculate Angle-Based Term for Rotation
            # Use the horizontal component (x) of the smoothed orientation vector
            angle_rot_term = self.kp_angle * self.smoothed_orientation_vector[0]

            # Combine P and D terms for rotation command
            z_rot = int(proportional_rot_term + angle_rot_term)

            # --- Gimbal Pitch Control for Vertical Centering ---
            if abs(y_offset) > self.offset_threshold:
                # Negative sign: positive y_offset (target below center) needs negative pitch (down)
                pitch_speed = -int(self.kp_pitch * y_offset)
                pitch_speed = max(-self.max_pitch_speed, min(self.max_pitch_speed, pitch_speed))
            # Ensure z_movement remains 0 as kp_alt is 0
            z_movement = 0

            # --- P Control for Forward/Backward ---
            if abs(y_offset) > self.offset_threshold:
                y_movement = int(self.kp_fwd * y_offset)

            # No longer need previous_x_offset for yaw control

        else:
            # Target Lost or Not Detected: Reset previous offset
            self.previous_center = None # Reset orientation tracking on loss
            if not self.moved_up:
                # self.drone_commander.move_by(forward=0, right=0, down=-0.5, rotation=0)
                print("Moved up")
                self.moved_up = True
                return
            else:
                # self.drone_commander.move_by(forward=0, right=0, down=0, rotation=0)
                print("Moved nothing")
                return

        # Clamp commands
        # Reduce the maximum command values to limit speed
        max_speed_command = 30 # Example: limit to 50% of max, adjust as needed
        max_rot_command = 50   # Example: limit rotation rate

        x_movement = max(-max_speed_command, min(max_speed_command, x_movement)) # Note: x_movement is not calculated here, usually 0
        y_movement = max(-max_speed_command, min(max_speed_command, y_movement))
        z_movement = max(-max_speed_command, min(max_speed_command, z_movement))
        z_rot = max(-max_rot_command, min(max_rot_command, z_rot))

        # Print the calculated commands instead of sending them
        status = "Tracking" if not target_lost else "Lost/Hover"
        # Record data to CSV
        current_time = time.time()
        new_data = pd.DataFrame([{
            'timestamp': current_time,
            'x_cmd': x_movement,
            'y_cmd': y_movement, 
            'z_cmd': z_movement,
            'rot_cmd': z_rot,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'p_rot': proportional_rot_term,
            'd_rot': angle_rot_term,
            'angle_rot': angle_rot_term,
            'pitch_speed': pitch_speed,
            'status': status
        }])
        new_data.to_csv('yolo_camera_log.csv', mode='a', header=not os.path.exists('yolo_camera_log.csv'), index=False)

        print(f"[{current_time:.2f}] Calculated Cmds ({status}): X={x_movement}, Y={y_movement}, Z={z_movement}, Rot={z_rot}")
        print(f"[{current_time:.2f}] Offsets: x={x_offset:.2f}, y={y_offset:.2f}, P_rot={proportional_rot_term:.1f}, Angle_rot={angle_rot_term:.1f}, PitchSpd={pitch_speed}")

        if not target_lost:
            self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.15)
            # Send gimbal pitch speed command (adjust method name if needed)
            try:
                self.drone_commander.set_gimbal_pitch_speed(pitch_speed)
            except AttributeError:
                print("[WARN] drone_commander does not have set_gimbal_pitch_speed method.")

            if self.moved_up == True:
                # self.drone_commander.move_by(forward=0, right=0, down=0.5, rotation=0)
                print("Moved down")
                self.moved_up = False
                return
            

    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        results = frame_data[1] # This should be the results object from predict

        if results is None:
             cv2.imshow("Drone View", original_frame) # Show original if no results
             cv2.waitKey(1)
             return

        plotted_frame = original_frame.copy() # Work on a copy for drawing
        overlay = plotted_frame.copy() # Create an overlay for semi-transparent drawing

        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        frame_center = (frame_center_x, frame_center_y)
        frame_dimensions = (frame_width, frame_height)

        best_target = {'conf': -1, 'center': None, 'size': None, 'box': None}
        target_lost_this_frame = True # Assume lost initially
        status_text = "LOST - HOVERING" # Default status text
        text_color = (0, 0, 255) # Red

        # Extract bounding boxes and find the best target
        current_raw_box_angle = None # Initialize for this frame
        if results.boxes and results.boxes.conf is not None and len(results.boxes.conf) > 0:
            best_conf_idx = results.boxes.conf.argmax()
            best_conf = results.boxes.conf[best_conf_idx].item() # Get as float
            # You might want a confidence threshold here too
            # if best_conf > SOME_THRESHOLD:
            target_lost_this_frame = False
            coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            best_target = {
                'conf': best_conf,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'size': (x2 - x1, y2 - y1),
                'box': (x1, y1, x2, y2)
            }

            # --- Orientation Estimation from Bounding Box Shape (using masks) ---
            if results.masks and len(results.masks) > best_conf_idx:
                mask_xy = results.masks.xy[best_conf_idx] # Get mask polygon coordinates
                contour = np.array(mask_xy, dtype=np.int32)

                if contour.size >= 6: # Need at least 3 points for minAreaRect
                    rect = cv2.minAreaRect(contour)
                    box_points = cv2.boxPoints(rect)
                    box_points_int = np.int0(box_points)

                    # Draw rotated rectangle (Magenta)
                    cv2.drawContours(plotted_frame, [box_points_int], 0, (255, 0, 255), 2)

                    # Estimate heading angle from the rectangle
                    (w, h) = rect[1]
                    raw_angle = rect[2] # Angle is between -90 and 0

                    if w < h: # If height is the longer side
                        heading_angle_raw = raw_angle + 90.0
                    else: # Width is the longer side
                        heading_angle_raw = raw_angle

                    # Normalize angle to [-180, 180)
                    heading_angle_raw = normalize_angle(heading_angle_raw)
                    current_raw_box_angle = heading_angle_raw # Store for display

                    # Smooth the angle using EMA (handling wrap-around)
                    angle_diff = shortest_angle_diff(heading_angle_raw, self.smoothed_box_angle)
                    self.smoothed_box_angle += self.angle_smoothing_factor * angle_diff
                    self.smoothed_box_angle = normalize_angle(self.smoothed_box_angle)

            elif not self.mask_warning_printed:
                 print("[WARN] YOLO model did not return masks. Cannot estimate orientation from shape.")
                 self.mask_warning_printed = True

        # --- Motion-based Orientation Estimation Logic (SMA) ---
        # This part calculates self.smoothed_orientation_vector for CONTROL
        if not target_lost_this_frame:
            current_center = np.array(best_target['center'])
            if self.previous_center is not None:
                motion_vector = current_center - self.previous_center
                motion_magnitude = np.linalg.norm(motion_vector)

                if motion_magnitude > self.min_movement_threshold:
                    # Normalize the motion vector
                    direction_vector = motion_vector / motion_magnitude

                    # Add normalized vector to deque for SMA
                    self.motion_vectors_deque.append(direction_vector)

            # Update previous center for the next frame
            self.previous_center = current_center
        else:
            # Reset previous center if target is lost
            self.previous_center = None


        # --- Command Generation ---
        # Note: Ensure _generate_follow_command handles the target_lost state correctly based on your logic
        # The previous state 'self.moved_up' affects command generation
        self._generate_follow_command(frame_center, best_target['center'], frame_dimensions,
                                      best_target['size'], target_lost=target_lost_this_frame)

        # --- Drawing Logic ---
        # Draw frame center marker regardless of target status
        cv2.circle(plotted_frame, frame_center, 7, (0, 0, 255), -1) # Slightly larger red dot
        cv2.circle(plotted_frame, frame_center, 9, (255, 255, 255), 1) # White outline

        if not target_lost_this_frame:
            # --- Target Found Drawing ---
            x1, y1, x2, y2 = best_target['box']
            best_center = best_target['center']
            confidence = best_target['conf']

            # Bounding Box Style
            box_color = (0, 255, 0) # Green
            box_thickness = 3
            fill_alpha = 0.15 # Transparency level for fill

            # Draw semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, fill_alpha, plotted_frame, 1 - fill_alpha, 0, plotted_frame)

            # Draw bounding box outline
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw line from frame center to target center
            cv2.line(plotted_frame, frame_center, best_center, box_color, 2)

            # Draw target center marker
            cv2.circle(plotted_frame, best_center, 5, box_color, -1) # Solid dot
            cv2.circle(plotted_frame, best_center, 7, (255, 255, 255), 1) # White outline

            # --- Draw Motion-Based Orientation Line (Blue) ---
            # Uses self.smoothed_orientation_vector from SMA calculation
            motion_line_length = 40
            motion_end_x = int(best_center[0] + motion_line_length * self.smoothed_orientation_vector[0])
            motion_end_y = int(best_center[1] + motion_line_length * self.smoothed_orientation_vector[1])
            cv2.line(plotted_frame, best_center, (motion_end_x, motion_end_y), (255, 0, 0), 2) # Blue line

            # --- Draw Shape-Based Orientation Line (Magenta) ---
            # Uses self.smoothed_box_angle
            shape_line_length = 50 # Make slightly longer for visibility
            angle_rad_shape = math.radians(self.smoothed_box_angle)
            shape_end_x = int(best_center[0] + shape_line_length * math.cos(angle_rad_shape))
            # Note: Screen Y is inverted, angle increases clockwise, use -sin
            shape_end_y = int(best_center[1] + shape_line_length * math.sin(angle_rad_shape))
            cv2.line(plotted_frame, best_center, (shape_end_x, shape_end_y), (255, 0, 255), 2) # Magenta line

            # --- Draw Orientation Line ---
            # if self.smoothed_orientation_vector is not None:
            #     orientation_line_length = 40 # Length of the orientation line in pixels
            #     current_angle_deg = None # Initialize for display scope
            #     # Calculate the angle from the smoothed vector
            #     angle_rad = np.arctan2(self.smoothed_orientation_vector[1], self.smoothed_orientation_vector[0])
            #     current_angle_deg = np.degrees(angle_rad) # For potential display or logging
            #     # Calculate the endpoint of the orientation line
            #     end_x = int(best_center[0] + orientation_line_length * self.smoothed_orientation_vector[0])
            #     end_y = int(best_center[1] + orientation_line_length * self.smoothed_orientation_vector[1])
            #     # Draw the orientation line (e.g., Blue color)
            #     cv2.line(plotted_frame, best_center, (end_x, end_y), (255, 0, 0), 2)


            # Confidence Text
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 5 # Position above or below box

            # Optional: Add background rect for confidence text
            # cv2.rectangle(plotted_frame, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), (0,0,0), -1)
            cv2.putText(plotted_frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)

            # Update status text for display
            status_text = "TRACKING"
            text_color = (0, 255, 0) # Green

            # Add logic for "Moved Down" status if applicable from _generate_follow_command
            if self.moved_up: # Check if the drone *was* moved up (implying it should move down now)
                 # This might need adjustment based on exactly when self.moved_up is False'd
                 status_text = "MOVING DOWN"
                 text_color = (255, 165, 0) # Blue/Cyan

        else:
            # --- Target Lost Drawing ---
            # Determine status based on whether the drone *has* moved up
            if self.moved_up:
                status_text = "LOST - SEARCHING UP"
                text_color = (0, 165, 255) # Orange
            else:
                # This case might mean lost before the first move up, or just generic lost
                status_text = "NO TARGET DETECTED"
                text_color = (0, 0, 255) # Red


        # --- Display Status Text ---
        status_font_scale = 0.8
        status_font_thickness = 2
        status_font = cv2.FONT_HERSHEY_TRIPLEX # A slightly fancier font
        (w, h), _ = cv2.getTextSize(status_text, status_font, status_font_scale, status_font_thickness)
        padding = 5
        # Position at top-left
        rect_x1, rect_y1 = padding, padding
        rect_x2, rect_y2 = padding + w + padding*2, padding + h + padding*2
        text_x, text_y = padding*2, padding + h + padding

        # Draw background rectangle for status text
        cv2.rectangle(plotted_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) # Black background
        # Draw the status text
        cv2.putText(plotted_frame, status_text, (text_x, text_y), status_font, status_font_scale, text_color, status_font_thickness, cv2.LINE_AA)

        # --- Display Estimated Angles ---
        angle_motion_deg = math.degrees(math.atan2(self.smoothed_orientation_vector[1], self.smoothed_orientation_vector[0]))
        angle_text_motion = f"Motion Angle: {normalize_angle(angle_motion_deg):.1f}"
        angle_text_box = f"Box Angle: {self.smoothed_box_angle:.1f}"
        # Position text
        text_y_motion = rect_y2 + h + padding
        text_y_box = text_y_motion + h + padding
        cv2.putText(plotted_frame, angle_text_motion, (padding * 2, text_y_motion), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plotted_frame, angle_text_box, (padding * 2, text_y_box), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the final frame
        cv2.imshow("Drone View", plotted_frame)
        cv2.waitKey(1)
                    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default="wireless", help="IP address of the drone. Use 'wireless' for physical drone or 'simulated' for simulator", choices=["wireless", "simulated", "cable"])
    args.add_argument("--speed", type=int, default=10)
    args.add_argument("--init", type=bool, default=False)
    args = args.parse_args()

    ip_selected = None
    if args.ip == "wireless":
        ip_selected = DroneIp.WIRELESS
    elif args.ip == "simulated":
        ip_selected = DroneIp.SIMULATED
    elif args.ip == "cable":
        ip_selected = DroneIp.CABLE

    if ip_selected is None:
        raise ValueError("Invalid IP address selected")

    controller = RealController(
        ip=ip_selected,
        processor_class=CarYoloProcessor,
        speed=args.speed,
    )

    if args.init:
        controller.initialize_position()

    controller.run()
    