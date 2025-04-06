import numpy as np
import ultralytics
import threading
import queue
import time
import cv2
import pandas as pd
from pathlib import Path

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

class YOLOProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/yolo11n.pt", **kwargs):
        self.detector = ultralytics.YOLO(model_path)

        # Initialize tracking data DataFrame (simplified columns)
        self.tracking_data = pd.DataFrame(columns=[
            'timestamp',
            'frame_center_x', 'frame_center_y',
            'object_center_x', 'object_center_y',
            'x_offset', 'y_offset',
            'box_width', 'box_height', 'size_ratio', 'target_ratio',
            'size_error', 'aspect_ratio',
            'x_cmd', 'y_cmd', 'z_cmd', 'rot_cmd',
            # Add Gains for logging if desired later
        ])

        # Tracking data file path
        self.data_path = Path("tracking_data")
        self.data_path.mkdir(exist_ok=True)
        self.data_file = self.data_path / f"tracking_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        # --- Control Gains ---
        self.kp_rot = 75 # Proportional gain for rotation (yaw) based on x_offset
        self.kd_rot = 15  # Derivative gain for rotation (damping) (NEEDS TUNING)
        self.kp_alt = 0   # Proportional gain for altitude (z) based on y_offset (DISABLED)
        self.kp_fwd = -95 # Proportional gain for forward (y) based on y_offset

        self.target_ratio = 0.07 # Target ratio (Not used for control, but kept for logging/info)

        # --- Thresholds ---
        self.offset_threshold = 0.1 # Deadband for x/y offset corrections

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time() # Initialize last time for dt calc

        super().__init__(**kwargs)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Runs YOLO detection on the frame."""
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]

    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        results = frame_data[1]
        plotted_frame = original_frame.copy()
        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        frame_center = (frame_center_x, frame_center_y)
        frame_dimensions = (frame_width, frame_height)
        cv2.circle(plotted_frame, frame_center, 5, (0, 255, 0), -1)

        best_target_in_frame = {'conf': -1, 'center': None, 'size': None, 'cls': None}
        vehicle_classes = {2, 5, 7}
        class_names = {2: "Car", 5: "Bus", 7: "Truck"}

        if results.boxes is not None and len(results.boxes) > 0:
            for i, cls_tensor in enumerate(results.boxes.cls):
                cls = int(cls_tensor.cpu().numpy())

                if cls in vehicle_classes:
                    box = results.boxes.xyxy[i].cpu().numpy()
                    conf = results.boxes.conf[i].cpu().numpy() if hasattr(results.boxes, 'conf') else 0.5

                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255) if cls == 2 else (255, 0, 0) if cls == 5 else (0, 255, 255)
                    label = f"{class_names.get(cls, 'Veh')} {conf:.2f}"
                    cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(plotted_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if conf > best_target_in_frame['conf']:
                        best_target_in_frame = {
                            'conf': conf,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'size': (x2 - x1, y2 - y1),
                            'cls': cls
                        }

        target_lost_this_frame = (best_target_in_frame['center'] is None)

        self._generate_follow_command(frame_center, best_target_in_frame['center'], frame_dimensions,
                                      best_target_in_frame['size'], best_target_in_frame['cls'],
                                      target_lost=target_lost_this_frame)

        if not target_lost_this_frame:
            best_center = best_target_in_frame['center']
            best_size = best_target_in_frame['size']
            cv2.line(plotted_frame, frame_center, best_center, (0, 255, 0), 2)

            box_width, box_height = best_size
            if frame_width > 0 and frame_height > 0:
                size_ratio = (box_width * box_height) / (frame_width * frame_height)
            else:
                size_ratio = 0
            cv2.putText(plotted_frame, f"Size: {size_ratio:.1%} (Tgt: {self.target_ratio:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(plotted_frame, f"Offset X: {(best_center[0]-frame_center_x)/frame_width*2:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(plotted_frame, f"Offset Y: {(best_center[1]-frame_center_y)/frame_height*2:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            status_text = "NO TARGET DETECTED"
            cv2.putText(plotted_frame, status_text, (int(frame_width * 0.1), int(frame_height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        cv2.putText(original_frame, "Original Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(plotted_frame, "Detection", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Ensure frames have the same width for vertical concatenation
        if original_frame.shape[1] == plotted_frame.shape[1]:
            combined_frame = cv2.vconcat([original_frame, plotted_frame]) # Use vconcat for vertical stacking
        else:
            # Fallback if shapes mismatch (though unlikely here)
            self.logger.warning("Frame widths mismatch, showing detection only.")
            combined_frame = plotted_frame
        cv2.imshow("Drone View", combined_frame)
        cv2.waitKey(1)

    def _run_processing_loop(self):
        self.logger.info("Starting frame processing loop.")
        self.last_command_time = time.time()
        self.previous_x_offset = 0.0 # Initialize previous offset

        with self._frame_display_context():
            while self._running.is_set() and threading.main_thread().is_alive():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    with self._lock:
                        self._frame_count += 1
                        if self.is_frame_saved:
                            self.frame_saver.add_frame(frame=np.array(frame), timestamp=time.perf_counter())
                        self._display_frame(self._process_frame(frame))
                except queue.Empty:
                    continue
                except Exception:
                    self.logger.error("Unable to process frame...", exc_info=True)
                    if not self._running.is_set():
                        break

        if not self.tracking_data.empty:
            self.tracking_data.to_csv(self.data_file, index=False)
            self.logger.info(f"Tracking data saved to {self.data_file}")
        self.logger.info("Frame processing loop terminated.")

    def _generate_follow_command(self, frame_center, object_center, frame_dimensions,
                                 box_size=None, object_class=None, target_lost=False):
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
        # Initialize state variables for logging
        x_offset, y_offset, size_error, aspect_ratio = 0.0, 0.0, 0.0, 1.0
        box_width, box_height, size_ratio = 0, 0, 0.0
        derivative_rot_term = 0 # Initialize derivative term

        # Calculate state and commands ONLY if target is NOT lost
        if not target_lost and object_center is not None and box_size is not None:
            object_center_x, object_center_y = object_center
            x_offset = (object_center_x - frame_center_x) / (frame_width / 2) if frame_width > 0 else 0
            y_offset = (object_center_y - frame_center_y) / (frame_height / 2) if frame_height > 0 else 0

            # --- Log Box Info ---
            box_width, box_height = box_size
            if frame_width > 0 and frame_height > 0:
                size_ratio = (box_width * box_height) / (frame_width * frame_height)
            if box_height > 0:
                aspect_ratio = box_width / box_height
            if self.target_ratio > 0:
                size_error = (self.target_ratio - size_ratio) / self.target_ratio
            else:
                size_error = 0

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

            self.logger.debug(f"Tracking Class: {object_class}, Off(x,y): ({x_offset:.2f},{y_offset:.2f}), P_rot:{proportional_rot_term:.1f}, D_rot:{derivative_rot_term:.1f}")

        else:
            # Target Lost or Not Detected: Hover
            self.logger.info("No target detected - Hovering")
            # Reset previous offset when target is lost to prevent large derivative jump on re-acquisition
            self.previous_x_offset = 0.0
            # All commands remain 0

        # Update previous offset for next calculation *after* using it
        # Only update if we had a valid offset this frame
        if not target_lost and object_center is not None:
             self.previous_x_offset = x_offset

        # Clamp commands
        x_movement = max(-100, min(100, x_movement))
        y_movement = max(-100, min(100, y_movement))
        z_movement = max(-100, min(100, z_movement))
        z_rot = max(-100, min(100, z_rot))

        status = f"Track Class {object_class}" if not target_lost else "Lost/Hover"
        self.logger.info(f"Cmds ({status}): X={x_movement}, Y={y_movement}, Z={z_movement}, Rot={z_rot}")

        # Record data
        new_data = pd.DataFrame([{
            'timestamp': current_time,
            'frame_center_x': frame_center_x, 'frame_center_y': frame_center_y,
            'object_center_x': object_center[0] if not target_lost else frame_center_x,
            'object_center_y': object_center[1] if not target_lost else frame_center_y,
            'x_offset': x_offset, 'y_offset': y_offset,
            'box_width': box_width, 'box_height': box_height,
            'size_ratio': size_ratio, 'target_ratio': self.target_ratio,
            'size_error': size_error, 'aspect_ratio': aspect_ratio,
            'x_cmd': x_movement, 'y_cmd': y_movement, 'z_cmd': z_movement, 'rot_cmd': z_rot,
            'object_class': object_class,
            # Add kp_rot, kd_rot to log if needed
        }])
        if 'aspect_ratio' not in self.tracking_data.columns:
             self.tracking_data['aspect_ratio'] = np.nan
        self.tracking_data = pd.concat([self.tracking_data, new_data], ignore_index=True)

        # Send command
        self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.15)

        # Save data periodically
        if len(self.tracking_data) % 50 == 0:
             self.tracking_data.to_csv(self.data_file, index=False)
