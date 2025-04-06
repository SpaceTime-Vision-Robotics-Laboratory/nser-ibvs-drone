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
        ])

        # Tracking data file path
        self.data_path = Path("tracking_data")
        self.data_path.mkdir(exist_ok=True)
        self.data_file = self.data_path / f"tracking_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        # Control gains (adjustable) - These are now the primary tuning parameters
        self.kp_rot = 30
        self.kp_alt = 30
        self.kp_fwd = 70
        self.target_ratio = 0.7

        # --- Thresholds ---
        self.offset_threshold = 0.05
        self.size_error_threshold = 0.05

        super().__init__(**kwargs)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Runs YOLO detection on the frame."""
        results = self.detector.track(frame, stream=False, verbose=True)
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

        view_mode_text = f"Mode: {self.view_mode.upper()}"
        cv2.putText(plotted_frame, view_mode_text, (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(original_frame, "Original Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(plotted_frame, "Detection", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if original_frame.shape[0] == plotted_frame.shape[0]:
            combined_frame = cv2.hconcat([original_frame, plotted_frame])
        else:
            combined_frame = plotted_frame
        cv2.imshow("Drone View", combined_frame)
        cv2.waitKey(1)

    def _run_processing_loop(self):
        self.logger.info("Starting frame processing loop.")
        self.view_mode = 'follow'
        self.last_command_time = time.time()

        with self._frame_display_context():
            while self._running.is_set() and threading.main_thread().is_alive():
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('v'):
                        self.view_mode = 'overhead' if self.view_mode == 'follow' else 'follow'
                        self.logger.info(f"Switched info display mode to {self.view_mode}")

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
        frame_center_x, frame_center_y = frame_center
        frame_width, frame_height = frame_dimensions

        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0
        x_offset, y_offset, size_error, aspect_ratio = 0.0, 0.0, 0.0, 1.0
        box_width, box_height, size_ratio = 0, 0, 0.0

        if not target_lost and object_center is not None and box_size is not None:
            object_center_x, object_center_y = object_center
            x_offset = (object_center_x - frame_center_x) / (frame_width / 2) if frame_width > 0 else 0
            y_offset = (object_center_y - frame_center_y) / (frame_height / 2) if frame_height > 0 else 0

            box_width, box_height = box_size
            if frame_width > 0 and frame_height > 0:
                size_ratio = (box_width * box_height) / (frame_width * frame_height)
            if box_height > 0:
                aspect_ratio = box_width / box_height

            if self.target_ratio > 0:
                size_error = (self.target_ratio - size_ratio) / self.target_ratio
            else:
                size_error = 0

            if abs(x_offset) > self.offset_threshold:
                z_rot = int(self.kp_rot * x_offset)
            if abs(y_offset) > self.offset_threshold:
                z_movement = -int(self.kp_alt * y_offset)
            if abs(size_error) > self.size_error_threshold:
                y_movement = int(self.kp_fwd * size_error)

            self.logger.debug(f"Tracking Class: {object_class}, Off(x,y): ({x_offset:.2f},{y_offset:.2f}), SzErr:{size_error:.2f}")

        else:
            self.logger.info("No target detected - Hovering")

        x_movement = max(-100, min(100, x_movement))
        y_movement = max(-100, min(100, y_movement))
        z_movement = max(-100, min(100, z_movement))
        z_rot = max(-100, min(100, z_rot))

        status = f"Track Class {object_class}" if not target_lost else "Lost/Hover"
        self.logger.info(f"Cmds ({status}): X={x_movement}, Y={y_movement}, Z={z_movement}, Rot={z_rot}")

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
        }])
        if 'aspect_ratio' not in self.tracking_data.columns:
             self.tracking_data['aspect_ratio'] = np.nan
        self.tracking_data = pd.concat([self.tracking_data, new_data], ignore_index=True)

        self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.1)

        if len(self.tracking_data) % 50 == 0:
             self.tracking_data.to_csv(self.data_file, index=False)
