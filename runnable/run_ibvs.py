import argparse
import sys
from pathlib import Path
import ultralytics
import pandas as pd
import os

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController
from drone_base.main.stream.base_video_processor import BaseVideoProcessor
import time
import numpy as np
import cv2


class IBVSController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self, takeoff_height=2.0, gimbal_angle=-35, back_distance=2.0, left_distance=0.0):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()
        self.drone_commander.take_off()
        time.sleep(1)
        self.drone_commander.tilt_camera(pitch_deg=gimbal_angle, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(1)
        self.drone_commander.move_by(forward=-back_distance, right=-left_distance, down=-takeoff_height, rotation=0)
        time.sleep(1)
        self.frame_processor.frame_queue.empty()
        print("Initialization complete. Ready for tracking.")
        
class IBVSProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/yolov11n_best_car_simulator.pt", target_width_ratio: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        print(f"Initializing IBVSProcessor with: model='{model_path}', target_width_ratio={target_width_ratio}")
        self.detector = ultralytics.YOLO(model_path)
        self.target_width_ratio = target_width_ratio

        self.target_aspect_ratio = None
        self.target_width_pixels = None
        self.target_height_pixels = None
        self.target_area_pixels = None

        self.kp_rot = 25
        self.kp_fwd_area = -35
        self.kp_lat = 15
        self.kp_alt = 20

        self.offset_threshold = 0.05
        self.area_threshold_ratio = 0.10

        self.last_command_time = time.time()
        self.log_file = 'ibvs_log.csv'
        if os.path.exists(self.log_file):
            print(f"Removing existing log file: {self.log_file}")
            os.remove(self.log_file)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        processed_results = results[0]

        if self.target_area_pixels is None and processed_results.boxes and len(processed_results.boxes.conf) > 0:
            best_conf_idx = processed_results.boxes.conf.argmax()
            coords = processed_results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            current_w = x2 - x1
            current_h = y2 - y1

            if current_w > 0 and current_h > 0:
                frame_height, frame_width = frame.shape[:2]
                self.target_width_pixels = int(frame_width * self.target_width_ratio)
                self.target_aspect_ratio = current_h / current_w
                self.target_height_pixels = int(self.target_width_pixels * self.target_aspect_ratio)
                self.target_area_pixels = self.target_width_pixels * self.target_height_pixels
                print(f"Frame Dims: {frame_width}x{frame_height}")
                print(f"First Detection: W={current_w}, H={current_h}")
                print(f"Target Features Set: W={self.target_width_pixels}, H={self.target_height_pixels}, Area={self.target_area_pixels}, Aspect={self.target_aspect_ratio:.2f}")

        return [frame, processed_results]
    
    def _generate_follow_command(self, frame_center, object_center, frame_dimensions,
                                 box_coords=None, target_lost=False):
        current_time = time.time()
        dt = current_time - self.last_command_time
        if dt <= 0.001: dt = 0.15
        self.last_command_time = current_time

        frame_center_x, frame_center_y = frame_center
        frame_width, frame_height = frame_dimensions

        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0
        x_offset_norm, y_offset_norm = 0.0, 0.0
        area_error_ratio = 0.0
        current_area = 0
        status = "Lost/Hover"

        if not target_lost and object_center is not None and box_coords is not None and self.target_area_pixels is not None:
            status = "Tracking"
            x1, y1, x2, y2 = box_coords
            current_cx = (x1 + x2) / 2
            current_cy = (y1 + y2) / 2
            current_w = x2 - x1
            current_h = y2 - y1

            if current_w > 0 and current_h > 0:
                current_area = current_w * current_h

                target_cx = frame_center_x
                target_cy = frame_center_y

                x_offset_pixels = current_cx - target_cx
                y_offset_pixels = current_cy - target_cy
                area_error = current_area - self.target_area_pixels

                x_offset_norm = x_offset_pixels / (frame_width / 2) if frame_width > 0 else 0
                y_offset_norm = y_offset_pixels / (frame_height / 2) if frame_height > 0 else 0
                area_error_ratio = area_error / self.target_area_pixels if self.target_area_pixels > 0 else 0

                if abs(x_offset_norm) > self.offset_threshold:
                    z_rot = int(self.kp_rot * x_offset_norm)
                    x_movement = int(self.kp_lat * x_offset_norm)

                if abs(y_offset_norm) > self.offset_threshold:
                    z_movement = -int(self.kp_alt * y_offset_norm)

                if abs(area_error_ratio) > self.area_threshold_ratio:
                    y_movement = int(self.kp_fwd_area * area_error_ratio)
            else:
                status = "Invalid Box Size"

        max_lin_speed_command = 40
        max_rot_command = 60

        x_movement = max(-max_lin_speed_command, min(max_lin_speed_command, x_movement))
        y_movement = max(-max_lin_speed_command, min(max_lin_speed_command, y_movement))
        z_movement = max(-max_lin_speed_command, min(max_lin_speed_command, z_movement))
        z_rot = max(-max_rot_command, min(max_rot_command, z_rot))

        log_data = {
            'timestamp': current_time,
            'status': status,
            'x_cmd': x_movement,
            'y_cmd': y_movement,
            'z_cmd': z_movement,
            'rot_cmd': z_rot,
            'x_offset_norm': x_offset_norm,
            'y_offset_norm': y_offset_norm,
            'current_area': current_area,
            'target_area': self.target_area_pixels if self.target_area_pixels else 0,
            'area_error_ratio': area_error_ratio,
        }
        try:
            new_data = pd.DataFrame([log_data])
            new_data.to_csv(self.log_file, mode='a', header=not os.path.exists(self.log_file), index=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

        print(f"[{current_time:.2f}] Status: {status}")
        print(f"[{current_time:.2f}] Cmds : X={x_movement}, Y={y_movement}, Z={z_movement}, Rot={z_rot}")
        print(f"[{current_time:.2f}] Errors: x_off={x_offset_norm:.2f}, y_off={y_offset_norm:.2f}, area_err={area_error_ratio:.2f}")

        if status == "Tracking" or status == "Lost/Hover":
            self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.15)


    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        results = frame_data[1]
        plotted_frame = original_frame.copy()
        overlay = plotted_frame.copy()

        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        frame_center = (frame_center_x, frame_center_y)
        frame_dimensions = (frame_width, frame_height)

        target_color = (255, 0, 255)
        target_box_coords = None
        if self.target_width_pixels and self.target_height_pixels:
            tw_half = self.target_width_pixels // 2
            th_half = self.target_height_pixels // 2
            tx1 = frame_center_x - tw_half
            ty1 = frame_center_y - th_half
            tx2 = frame_center_x + tw_half
            ty2 = frame_center_y + th_half
            target_box_coords = (tx1, ty1, tx2, ty2)

        best_target = {'conf': -1, 'center': None, 'box': None}
        target_lost_this_frame = True
        status_text = "LOST - HOVERING"
        text_color = (0, 0, 255)

        if results.boxes and results.boxes.conf is not None and len(results.boxes.conf) > 0:
            best_conf_idx = results.boxes.conf.argmax()
            best_conf = results.boxes.conf[best_conf_idx].item()
            target_lost_this_frame = False
            coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
            x1_det, y1_det, x2_det, y2_det = map(int, coords)
            best_target = {
                'conf': best_conf,
                'center': ((x1_det + x2_det) // 2, (y1_det + y2_det) // 2),
                'box': (x1_det, y1_det, x2_det, y2_det)
            }

        self._generate_follow_command(frame_center, best_target['center'], frame_dimensions,
                                      best_target['box'], target_lost=target_lost_this_frame)

        cv2.circle(plotted_frame, frame_center, 7, (0, 0, 255), -1)
        cv2.circle(plotted_frame, frame_center, 9, (255, 255, 255), 1)

        if target_box_coords:
            tx1, ty1, tx2, ty2 = target_box_coords
            cv2.rectangle(plotted_frame, (tx1, ty1), (tx2, ty2), target_color, 1)
            target_info_text = f"Target W:{self.target_width_pixels} H:{self.target_height_pixels}"
            cv2.putText(plotted_frame, target_info_text, (tx1, ty1 - 5 if ty1 > 10 else ty1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1, cv2.LINE_AA)

        if not target_lost_this_frame:
            x1, y1, x2, y2 = best_target['box']
            best_center = best_target['center']
            confidence = best_target['conf']
            current_w = x2 - x1
            current_h = y2 - y1

            box_color = (0, 255, 0)
            box_thickness = 2
            fill_alpha = 0.1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, fill_alpha, plotted_frame, 1 - fill_alpha, 0, plotted_frame)
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            cv2.line(plotted_frame, frame_center, best_center, box_color, 1)
            cv2.circle(plotted_frame, best_center, 5, box_color, -1)
            cv2.circle(plotted_frame, best_center, 7, (255, 255, 255), 1)

            info_text = f"Conf: {confidence:.2f} W:{current_w} H:{current_h}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (w_text, h_text), _ = cv2.getTextSize(info_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 5 if y1 > 15 else y1 + h_text + 5
            cv2.putText(plotted_frame, info_text, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)

            status_text = "TRACKING"
            text_color = (0, 255, 0)
        else:
            status_text = "NO TARGET DETECTED"
            text_color = (0, 0, 255)

        status_font_scale = 0.7
        status_font_thickness = 2
        status_font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(status_text, status_font, status_font_scale, status_font_thickness)
        padding = 5
        rect_x1, rect_y1 = padding, frame_height - h - padding*3
        rect_x2, rect_y2 = padding + w + padding*2, frame_height - padding
        text_x, text_y = padding*2, frame_height - padding*2

        cv2.rectangle(plotted_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(plotted_frame, status_text, (text_x, text_y), status_font, status_font_scale, text_color, status_font_thickness, cv2.LINE_AA)

        cv2.imshow("IBVS Multi-Feature View", plotted_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default="wireless", help="IP address of the drone ('wireless', 'simulated', 'cable')", choices=["wireless", "simulated", "cable"])
    args.add_argument("--init", action='store_true', help="Run initialization sequence (takeoff, position)")
    args.add_argument("--model", type=str, default="models/yolon_car_detector.pt", help="Path to YOLO detection model")
    args.add_argument("--target_width", type=float, default=0.3, help="Target bounding box width as a ratio of frame width (e.g., 0.3 for 30%)")
    args.add_argument("--init_gimbal", type=float, default=-25.0, help="Initial gimbal pitch angle (degrees)")
    args.add_argument("--init_height", type=float, default=1.5, help="Initial takeoff/hover height (meters)")
    args.add_argument("--init_back", type=float, default=2.5, help="Initial distance behind the target (meters)")

    args = args.parse_args()

    if args.ip == "wireless":
        ip_selected = DroneIp.WIRELESS
    elif args.ip == "simulated":
        ip_selected = DroneIp.SIMULATED
    elif args.ip == "cable":
        ip_selected = DroneIp.CABLE
    else:
        raise ValueError("Invalid IP address selected")


    controller = IBVSController(
        ip=ip_selected,
        processor_class=IBVSProcessor,
    )

    if hasattr(controller, 'frame_processor') and isinstance(controller.frame_processor, IBVSProcessor):
        print(f"Configuring IBVSProcessor with: model='{args.model}', target_width={args.target_width}")
        try:
            controller.frame_processor.detector = ultralytics.YOLO(args.model)
            print(f"Successfully loaded YOLO model from: {args.model}")
        except Exception as e:
            print(f"Error loading YOLO model from {args.model}: {e}")

        controller.frame_processor.target_width_ratio = args.target_width
        controller.frame_processor.target_width_pixels = None 
    else:
        print("Warning: Could not find frame_processor of type IBVSProcessor to configure.")

    if args.init:
        controller.initialize_position(
            takeoff_height=args.init_height,
            gimbal_angle=args.init_gimbal,
            back_distance=args.init_back,
            left_distance=-1.0
        )

    controller.run()