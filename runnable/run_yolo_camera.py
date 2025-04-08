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
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-75, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)
        if hasattr(self.frame_processor, 'reset_state'):
             self.frame_processor.reset_state()
        print("Initialization complete. Ready for tracking.")

    def initialize_position_simulated(self, takeoff_height=2.0, gimbal_angle=-45, back_distance=2.0, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        print("Taking off...")
        self.drone_commander.take_off()
        print(f"Ascending to {takeoff_height}m...")
        self.drone_commander.move_by(forward=0, right=0, down=-takeoff_height, rotation=0)
        print(f"Moving backward {back_distance}m...")
        self.drone_commander.move_by(forward=-back_distance, right=0, down=0, rotation=0)
        print(f"Moving left {left_distance}m...")
        self.drone_commander.move_by(forward=0, right=-left_distance, down=0, rotation=0)
        print(f"Tilting gimbal to {gimbal_angle} degrees...")
        self.drone_commander.tilt_camera(
            pitch_deg=gimbal_angle,
            control_mode=GimbalType.MODE_POSITION,
            reference_type=GimbalType.REF_ABSOLUTE
            )
        if hasattr(self.frame_processor, 'reset_state'):
             self.frame_processor.reset_state()
        print("Initialization complete. Ready for tracking.")


class CarYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/yolo11n.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = ultralytics.YOLO(model_path)

        # --- Vertical Movement Parameters ---
        self.vertical_pilot_speed = 60      # Speed (%) for piloting up/down
        # *** Tune this duration based on speed to approximate desired distance (e.g., 0.5m) ***
        self.vertical_move_duration = 1.5   # Seconds for vertical move

        # --- Frame Count Thresholds ---
        self.lost_frame_threshold = 30      # Frames to wait before ascending
        self.centered_descend_threshold = 90 # Frames to wait while centered before descending

        # --- Centering Threshold ---
        self.centering_offset_threshold = 0.08 # Horizontal offset for "centered"

        # --- Control Gains ---
        self.kp_rot = 20
        self.kd_rot = 5
        self.kp_fwd = -25

        # --- Horizontal Offset Threshold ---
        self.offset_threshold = 0.1 # Deadband for starting horizontal corrections

        # --- State Variables ---
        self.moved_up = False # Represents the drone's *current* vertical state (low or high)
        self.was_ascending = False # Helper flag to know which move just finished
        self.lost_frame_counter = 0
        self.found_and_centered_counter = 0
        self.is_moving_vertically = False
        self.vertical_move_start_time = None
        self.previous_x_offset = 0.0
        self.last_command_time = time.time()

        self.reset_state() # Initialize

    def reset_state(self):
        """Resets state flags and counters."""
        self.moved_up = False
        self.lost_frame_counter = 0
        self.found_and_centered_counter = 0
        self.is_moving_vertically = False
        self.vertical_move_start_time = None
        self.previous_x_offset = 0.0
        self.was_ascending = False # Reset helper flag
        print("Processor state reset.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]

    def _generate_follow_command(self, frame_center, object_center, frame_dimensions,
                                 box_size=None, target_lost=False):
        current_time = time.time()
        dt = current_time - self.last_command_time
        if dt <= 0.001: dt = 0.15
        self.last_command_time = current_time

        frame_center_x, frame_center_y = frame_center
        frame_width, frame_height = frame_dimensions

        x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0
        x_offset, y_offset = 0.0, 0.0
        proportional_rot_term = 0.0
        derivative_rot_term = 0.0
        status = "Unknown"
        is_centered = False # Default this frame

        # --- Step 1: Handle Active Timed Vertical Movement ---
        if self.is_moving_vertically:
            elapsed_time = current_time - self.vertical_move_start_time
            if elapsed_time < self.vertical_move_duration:
                # Continue the move based on the direction set when the move started
                z_cmd = self.vertical_pilot_speed if self.was_ascending else -self.vertical_pilot_speed
                status = f"Ascending ({elapsed_time:.1f}s)" if self.was_ascending else f"Descending ({elapsed_time:.1f}s)"
                x_cmd, y_cmd, rot_cmd = 0, 0, 0 # Hover horizontally
                self.previous_x_offset = 0.0
            else:
                # Finish the move
                print(f"[{current_time:.2f}] Timed vertical move finished.")
                z_cmd = 0
                self.is_moving_vertically = False
                self.vertical_move_start_time = None

                # Update moved_up state *after* finishing the move
                if self.was_ascending:
                    self.moved_up = True  # Ascent finished, we are now high
                    print(f"[{current_time:.2f}] State Update: moved_up = True")
                else: # was descending
                    self.moved_up = False # Descent finished, we are now low
                    print(f"[{current_time:.2f}] State Update: moved_up = False")

                self.was_ascending = False # Reset helper flag

                status = "Searching High" if self.moved_up else "Tracking Low"


        # --- Step 2: If NOT moving vertically, determine actions based on target ---
        elif target_lost:
            self.lost_frame_counter += 1
            self.found_and_centered_counter = 0 # Reset centering counter
            self.previous_x_offset = 0.0
            x_cmd, y_cmd, rot_cmd = 0, 0, 0 # Hover horizontally

            # Check condition to START ascending
            if not self.moved_up and self.lost_frame_counter > self.lost_frame_threshold:
                print(f"[{current_time:.2f}] Lost threshold ({self.lost_frame_threshold}) exceeded. Initiating ascent.")
                # Don't set moved_up=True yet, only when move finishes
                self.is_moving_vertically = True
                self.was_ascending = True # Set helper flag
                self.vertical_move_start_time = current_time
                z_cmd = -self.vertical_pilot_speed # Start command
                status = f"Ascending (Started)"
                self.lost_frame_counter = 0 # Reset counter after triggering
            else:
                # Lost, but either not long enough or already high -> Hover/Search
                z_cmd = 0
                status = "Searching High" if self.moved_up else f"Lost (Waiting {self.lost_frame_counter}/{self.lost_frame_threshold})"

        else: # Target Found
            self.lost_frame_counter = 0 # Reset lost counter

            if object_center is not None: # Ensure we have coordinates
                obj_x, obj_y = object_center
                x_offset = (obj_x - frame_center_x) / (frame_width / 2) if frame_width > 0 else 0
                y_offset = (obj_y - frame_center_y) / (frame_height / 2) if frame_height > 0 else 0
                is_centered = abs(x_offset) < self.centering_offset_threshold

                # Always calculate horizontal commands when target is found
                if abs(x_offset) > self.offset_threshold:
                    proportional_rot_term = self.kp_rot * x_offset
                delta_x_offset = x_offset - self.previous_x_offset
                derivative_rot_term = self.kd_rot * (delta_x_offset / dt)
                rot_cmd = int(proportional_rot_term + derivative_rot_term)

                if abs(y_offset) > self.offset_threshold:
                    y_cmd = int(self.kp_fwd * y_offset)

                # --- Decide vertical action based on altitude ---
                if self.moved_up: # Drone is High
                    z_cmd = 0 # Default to hover vertically while high
                    if is_centered:
                        self.found_and_centered_counter += 1
                        status = f"Centered High (Wait {self.found_and_centered_counter}/{self.centered_descend_threshold})"
                        # Check condition to START descending
                        if self.found_and_centered_counter > self.centered_descend_threshold:
                            print(f"[{current_time:.2f}] Centered threshold ({self.centered_descend_threshold}) exceeded. Initiating descent.")
                            self.is_moving_vertically = True
                            self.was_ascending = False # Set helper flag
                            self.vertical_move_start_time = current_time
                            z_cmd = self.vertical_pilot_speed # Start descent command
                            # moved_up flag remains True until descent *finishes*
                            status = f"Descending (Started)"
                            x_cmd, y_cmd, rot_cmd = 0, 0, 0 # Hover horizontally during descent
                            self.found_and_centered_counter = 0 # Reset counter
                    else:
                        # High, but not centered -> Keep centering
                        self.found_and_centered_counter = 0 # Reset counter
                        status = "Centering High"
                        # Horizontal commands already calculated above

                    self.previous_x_offset = x_offset # Update PD state while centering/waiting high

                else: # Drone is Low -> Normal Tracking
                    self.found_and_centered_counter = 0 # Reset counter
                    status = "Tracking Low"
                    z_cmd = 0 # Maintain altitude
                    # Horizontal commands already calculated above
                    self.previous_x_offset = x_offset # Update PD state

            else: # Target found frame, but no object center? Hover.
                 status = "Tracking Low - No Box" if not self.moved_up else "Searching High - No Box"
                 x_cmd, y_cmd, z_cmd, rot_cmd = 0, 0, 0, 0
                 self.previous_x_offset = 0.0
                 self.found_and_centered_counter = 0


        # --- Step 3: Clamp and Log ---
        max_lin_speed = 50
        max_rot_speed = 70
        max_vert_speed = 100
        x_cmd = max(-max_lin_speed, min(max_lin_speed, x_cmd))
        y_cmd = max(-max_lin_speed, min(max_lin_speed, y_cmd))
        z_cmd = max(-max_vert_speed, min(max_vert_speed, z_cmd))
        rot_cmd = max(-max_rot_speed, min(max_rot_speed, rot_cmd))

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'yolo_camera_centered_log.csv'
        new_data = pd.DataFrame([{
            'timestamp': current_time, 'status': status,
            'target_lost': target_lost, 'moved_up': self.moved_up,
            'is_moving_vert': self.is_moving_vertically,
            'lost_frames': self.lost_frame_counter,
            'centered_frames': self.found_and_centered_counter,
            'is_centered': is_centered,
            'x_offset': x_offset, 'y_offset': y_offset,
            'p_rot': proportional_rot_term, 'd_rot': derivative_rot_term,
            'x_cmd': x_cmd, 'y_cmd': y_cmd, 'z_cmd': z_cmd, 'rot_cmd': rot_cmd
        }])
        new_data.to_csv(log_file, mode='a', header=not log_file.exists(), index=False)

        # --- Step 4: Print Debug Info ---
        print(f"[{current_time:.2f}] Status: {status} | MovedUp: {self.moved_up} | MovingVert: {self.is_moving_vertically} | Lost: {self.lost_frame_counter} | Centered: {self.found_and_centered_counter}")
        print(f"[{current_time:.2f}] Piloting Cmds: X={x_cmd}, Y={y_cmd}, Z={z_cmd}, Rot={rot_cmd}")

        # --- Step 5: Send Command ---
        self.drone_commander.piloting(x=x_cmd, y=y_cmd, z=z_cmd, z_rot=rot_cmd, dt=0.15)

        return status

    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        results = frame_data[1]
        plotted_frame = original_frame.copy()
        overlay = plotted_frame.copy()

        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        frame_center = (frame_center_x, frame_center_y)

        best_target = {'conf': -1, 'center': None, 'size': None, 'box': None}
        target_lost_this_frame = True

        confidence_threshold = 0.3
        if results.boxes and results.boxes.conf is not None and len(results.boxes.conf) > 0:
            best_conf_idx = results.boxes.conf.argmax()
            best_conf = results.boxes.conf[best_conf_idx].item()
            if best_conf >= confidence_threshold:
                target_lost_this_frame = False
                coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                best_target = {
                    'conf': best_conf, 'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'size': (x2 - x1, y2 - y1), 'box': (x1, y1, x2, y2)
                }

        # --- Generate Command & Get Status ---
        current_status = self._generate_follow_command(
            frame_center, best_target['center'], (frame_width, frame_height),
            best_target['size'], target_lost=target_lost_this_frame
        )

        # --- Drawing Logic ---
        cv2.circle(plotted_frame, frame_center, 7, (0, 0, 255), -1)
        cv2.circle(plotted_frame, frame_center, 9, (255, 255, 255), 1)

        text_color = (255, 255, 255) # Default white

        if not target_lost_this_frame and best_target['center'] is not None:
            x1, y1, x2, y2 = best_target['box']
            best_center = best_target['center']
            confidence = best_target['conf']
            box_color = (0, 255, 0)
            box_thickness = 3; fill_alpha = 0.15
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, fill_alpha, plotted_frame, 1 - fill_alpha, 0, plotted_frame)
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            cv2.line(plotted_frame, frame_center, best_center, box_color, 2)
            cv2.circle(plotted_frame, best_center, 5, box_color, -1)
            cv2.circle(plotted_frame, best_center, 7, (255, 255, 255), 1)
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; font_thickness = 1
            text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            text_x = x1; text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 5
            cv2.putText(plotted_frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)

        # --- Display Status Text ---
        if "Ascending" in current_status or "Descending" in current_status:
             text_color = (255, 0, 0) # Blue
        elif "Tracking Low" in current_status and "No Box" not in current_status:
             text_color = (0, 255, 0) # Green
        elif "Centering High" in current_status:
             text_color = (0, 255, 255) # Yellow
        elif "Centered High" in current_status:
             text_color = (255, 255, 0) # Cyan
        elif "Searching High" in current_status:
             text_color = (0, 165, 255) # Orange
        elif "Lost" in current_status:
             text_color = (0, 0, 255) # Red

        status_font_scale = 0.7
        status_font_thickness = 2
        status_font = cv2.FONT_HERSHEY_TRIPLEX
        # Add counters to display text
        display_text = f"{current_status} (L:{self.lost_frame_counter}, C:{self.found_and_centered_counter})"
        (w, h), _ = cv2.getTextSize(display_text, status_font, status_font_scale, status_font_thickness)
        padding = 5
        rect_x1, rect_y1 = padding, padding
        rect_x2, rect_y2 = padding + w + padding*2, padding + h + padding*2
        text_x, text_y = padding*2, padding + h + padding
        cv2.rectangle(plotted_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(plotted_frame, display_text, (text_x, text_y), status_font, status_font_scale, text_color, status_font_thickness, cv2.LINE_AA)

        cv2.imshow("Drone View", plotted_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default="simulated", help="IP address of the drone. Use 'simulated' for simulator", choices=["wireless", "simulated", "cable"])
    args.add_argument("--init", type=bool, default=True)

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
        processor_class=CarYoloProcessor
    )

    if args.init:
        controller.initialize_position_simulated(takeoff_height=2.5, gimbal_angle=-45, back_distance=3.0, left_distance=0.0)

    controller.run()
