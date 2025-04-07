from pathlib import Path
import sys
import time
import argparse
import ultralytics
import numpy as np
import cv2
import pandas as pd
import os
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

class RealController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self):
        # self.drone_commander.take_off()
        # time.sleep(5)
        # self.drone_commander.move_by(forward=0, right=0, down=-2.0, rotation=0)
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-45, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)
        self.frame_processor.frame_queue.empty()
        print("Initialization complete. Ready for tracking.")


class CarYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/car_segmentation.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = ultralytics.YOLO(model_path)

        # --- Control Gains ---
        # self.kp_rot = 75 # Proportional gain for rotation (yaw) based on x_offset
        # self.kd_rot = 15  # Derivative gain for rotation (damping)
        # self.kp_alt = 0   # Proportional gain for altitude (z) based on y_offset (DISABLED)
        # self.kp_fwd = -95 # Proportional gain for forward (y) based on y_offset

        self.kp_rot = 35 # Proportional gain for rotation (yaw) based on x_offset
        self.kd_rot = 25  # Derivative gain for rotation (damping)
        self.kp_alt = 0   # Proportional gain for altitude (z) based on y_offset (DISABLED)
        self.kp_fwd = -25 # Proportional gain for forward (y) based on y_offset

        # --- Thresholds ---
        self.offset_threshold = 0.1 # Deadband for x/y offset corrections

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time() # Initialize last time for dt calc

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]

    def _generate_follow_command(self, frame_center, object_center, frame_dimensions,
                                 box_size=None, target_lost=False):
        current_time = time.time()
        # Calculate dt, handle potential first run or zero dt
        dt = current_time - self.last_command_time
        if dt <= 0.001: # Avoid division by zero or excessively large derivatives
            dt = 0.15 # Assume a nominal dt if issue occurs
        self.last_command_time = current_time

        frame_center_x, frame_center_y = frame_center
        frame_width, frame_height = frame_dimensions

        # Initialize commands to zero (hover)
        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0
        # Initialize state variables for logging/debugging
        x_offset, y_offset = 0.0, 0.0
        derivative_rot_term = 0 # Initialize derivative term
        proportional_rot_term = 0
        # Calculate state and commands ONLY if target is NOT lost

        if not target_lost and object_center is not None and box_size is not None:
            object_center_x, object_center_y = object_center
            x_offset = (object_center_x - frame_center_x) / (frame_width / 2) if frame_width > 0 else 0
            y_offset = (object_center_y - frame_center_y) / (frame_height / 2) if frame_height > 0 else 0

            # --- PD Control for Rotation ---
            proportional_rot_term = 0
            if abs(x_offset) > self.offset_threshold:
                proportional_rot_term = self.kp_rot * x_offset

            # Calculate Derivative Term for Rotation
            delta_x_offset = x_offset - self.previous_x_offset
            derivative_rot_term = self.kd_rot * (delta_x_offset / dt)

            # Combine P and D terms for rotation command
            z_rot = int(proportional_rot_term + derivative_rot_term)

            # --- P Control for Altitude (Disabled) ---
            if abs(y_offset) > self.offset_threshold:
                z_movement = -int(self.kp_alt * y_offset) # Logic remains, gain is 0

            # --- P Control for Forward/Backward ---
            if abs(y_offset) > self.offset_threshold:
                y_movement = int(self.kp_fwd * y_offset)

            # Update previous offset for next calculation *after* using it
            self.previous_x_offset = x_offset

        else:
            # Target Lost or Not Detected: Reset previous offset
            self.previous_x_offset = 0.0

        # Clamp commands
        x_movement = max(-100, min(100, x_movement)) # Note: x_movement is not calculated here, usually 0
        y_movement = max(-100, min(100, y_movement))
        z_movement = max(-100, min(100, z_movement))
        z_rot = max(-100, min(100, z_rot))

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
            'd_rot': derivative_rot_term,
            'status': status
        }])
        new_data.to_csv('yolo_camera_log.csv', mode='a', header=not os.path.exists('yolo_camera_log.csv'), index=False)

        print(f"[{current_time:.2f}] Calculated Cmds ({status}): X={x_movement}, Y={y_movement}, Z={z_movement}, Rot={z_rot}")
        print(f"[{current_time:.2f}] Offsets: x={x_offset:.2f}, y={y_offset:.2f}, P_rot={proportional_rot_term:.1f}, D_rot={derivative_rot_term:.1f}")

    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        results = frame_data[1]
        plotted_frame = original_frame.copy() # Work on a copy for drawing

        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        frame_center = (frame_center_x, frame_center_y)
        frame_dimensions = (frame_width, frame_height)

        best_target = {'conf': -1, 'center': None, 'size': None, 'box': None}

        # Extract bounding boxes and find the best target (highest confidence)
        if results.boxes:
            for i, box_tensor in enumerate(results.boxes.xyxy):
                conf = results.boxes.conf[i].cpu().numpy() if hasattr(results.boxes, 'conf') else 0.5 # Get confidence
                if conf > best_target['conf']:
                    coords = box_tensor.cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    best_target = {
                        'conf': conf,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'size': (x2 - x1, y2 - y1),
                        'box': (x1, y1, x2, y2) # Store box coords
                    }

        target_lost_this_frame = (best_target['center'] is None)

        # Generate commands based on the best target found (or target lost state)
        self._generate_follow_command(frame_center, best_target['center'], frame_dimensions,
                                      best_target['size'], target_lost=target_lost_this_frame)

        # Draw bounding box and center line if target exists
        if not target_lost_this_frame:
            x1, y1, x2, y2 = best_target['box']
            best_center = best_target['center']
            # Draw the rectangle directly onto the plotted frame
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box, thickness 2
            # Draw line from frame center to target center
            cv2.line(plotted_frame, frame_center, best_center, (0, 255, 0), 2)
            # Draw frame center
            cv2.circle(plotted_frame, frame_center, 5, (0, 0, 255), -1) # Red dot for frame center
        else:
            # Optionally indicate no target found on screen
            status_text = "NO TARGET DETECTED"
            cv2.putText(plotted_frame, status_text, (int(frame_width * 0.1), int(frame_height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        # Display the frame with drawings
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
        speed=args.speed
    )

    if args.init:
        controller.initialize_position()

    controller.run()
