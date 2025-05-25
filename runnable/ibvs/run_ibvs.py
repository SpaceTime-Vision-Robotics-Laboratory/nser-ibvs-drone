from pathlib import Path
import sys
import time
import argparse
import ultralytics
import numpy as np
import cv2
import pandas as pd
import os

import pickle

from pathlib import Path
import json

from utils import e2h, plot_bbox_keypoints, check_stability_bbox_simple, compute_distance

# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

## -------------------------------------------------------------

MODEL_PATH = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/models/yolo11n-seg_car_sim_simple.pt"

## -------------------------------------------------------------

class ImageBasedVisualServo():
    def __init__(self, K, goal_points):
        self.K = K
        # print(f"{self.K=}")
        self.Kinv = np.linalg.inv(self.K)
        # print(f"{self.Kinv=}")

        self.lambda_factor = 0.1
        diagonal = [self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor * 2]
        self.lambda_factor = np.diag(diagonal)

        self.goal_points = goal_points
        self.goal_points_normalized = self.compute_normalized_image_plane_coordinates(self.goal_points)

        self.current_points = None
        self.current_points_normalized = None

        self.jcond_values = []
        self.err_values = []
    
    def compute_normalized_image_plane_coordinates(self, points):
        points_normalized = []
        
        for p in points:
            p_array = np.array([[p[0]], [p[1]]])

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]
            points_normalized.append((x, y))
        
        points_normalized = np.array(points_normalized)
        points_normalized = np.hstack(points_normalized)

        return points_normalized
    
    def set_current_points(self, current_points):
        self.current_points = current_points
        self.current_points_normalized = self.compute_normalized_image_plane_coordinates(self.current_points)
    
    def compute_interaction_matrix(self):
        ## TODO set a proper value for Z
        Z = 12

        J_list = []
        for i in range(0, len(self.current_points_normalized), 2):
            x = self.current_points_normalized[i]
            y = self.current_points_normalized[i+1]

            J_point = self.K[:2,:2] @ np.array([
                [-1/Z,     0,    x/Z,  x*y,     -(1+x*x),  y],
                [    0, -1/Z,    y/Z,  1+y*y,   -x*y,     -x]
            ])

            J_list.append(J_point)
        
        J = np.vstack(J_list)
        # print(f"Interaction matrix shape: {J.shape}")
        # print(f"Interaction matrix : {J}")
        # print("\n ----------------------------------- \n")

        return J

    def compute_velocities(self, plot=False):
        J = self.compute_interaction_matrix()
        
        jcond = np.linalg.cond(J)
        self.jcond_values.append(jcond)
        print(f"J cond: {jcond}")

        J_pinv = np.linalg.pinv(J)

        err = self.goal_points_normalized - self.current_points_normalized
        self.err_values.append(err)

        print("--------------------------------------------")
        print(f"err: {err}")
        print(f"Current: {self.current_points_normalized} \n Goal: {self.goal_points_normalized}")
        print("--------------------------------------------")

        if (isinstance(self.lambda_factor, np.ndarray)):
            vel = self.lambda_factor @ J_pinv @ err
        else:
            vel = self.lambda_factor * J_pinv @ err

        return vel


class IBVSController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-90, control_mode=GimbalType.MODE_POSITION, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)

        self.drone_commander.move_by(forward=0, right=5, down=-10, rotation=0)

        self.frame_processor.frame_queue.empty()
        
        print("Initialization complete. Ready for tracking.")

class IBVSYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path=MODEL_PATH, **kwargs):
        super().__init__(**kwargs)

        self.detector = ultralytics.YOLO(model_path)

        ## Goal position

        goal_frame_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/images/frame_drone_sim_10m_center.png"
        self.goal_frame_points_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/data/frame_drone_sim_10m_center.json"
        with open(self.goal_frame_points_path, "r") as f:
            goal_points = json.load(f)
        
        self.goal_points_bbox = goal_points["bbox_points"]
        self.goal_points_bbox = self.goal_points_bbox[:4]

        path_to_file = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/camera-parameters/sim-anafi-4k/intrinsic_matrix_half_size.pkl"
        with open(path_to_file, 'rb') as f:
            K = pickle.load(f)
        
        self.ibvs = ImageBasedVisualServo(K, self.goal_points_bbox)

        self.max_linear_speed = 2 # m/s
        self.max_height_linear_speed = 1 # m/s
        self.max_angular_speed = np.deg2rad(60) # rad/s
    
    def velocity_to_command(self, velocities):
        roll = 100 * velocities[0] / self.max_linear_speed
        pitch = -100 * velocities[1] / self.max_linear_speed
        gaz = 100 * velocities[2] / self.max_height_linear_speed
        yaw = 100 * velocities[5] / self.max_angular_speed

        roll *= 1000
        pitch *= 1000
        yaw *= 1000

        print(f"{roll=} | {pitch=} | {gaz=} | {yaw=}")

        self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=0, dt=0.15)
        # self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=int(yaw), dt=0.15)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]
    
    def _display_frame(self, frame_data: list) -> None:
        plotted_frame = frame_data[0].copy()

        frame_h, frame_w = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        
        frame_center = (frame_center_x, frame_center_y)
        cv2.circle(plotted_frame, frame_center, 7, (0, 165, 255), -1)

        predictions = frame_data[1]
        if predictions.boxes and predictions.boxes.conf is not None and len(predictions.boxes.conf) > 0:
            best_idx = predictions.boxes.conf.argmax()
            # print(f"Best idx {best_idx} / {len(predictions.boxes.conf)}")
            
            best_conf = predictions.boxes.conf[best_idx]
            if (best_conf >= 0.8):
                ## Segmentation
                ## ------------------------------------------------
                # xy_seg = predictions.masks.xy[best_idx]
                # xy_seg = [list(xy) for xy in xy_seg]
                # xy_seg = np.array(xy_seg).astype(np.int32)
                
                
                # seg_color = (0, 200, 0)
                # cv2.fillPoly(plotted_frame, pts=[xy_seg], color=seg_color)
                ## ------------------------------------------------

                ## Bounding box
                ## ------------------------------------------------
                xyxy = predictions.boxes.xyxy[best_idx].cpu().numpy()
                xl, yl, xr, yr = map(int, xyxy)

                xc = (xl + xr) // 2
                yc = (yl + yr) // 2
                cv2.circle(plotted_frame, (xc, yc), 5, (255, 255, 0), -1)

                points_bbox = [(xl, yl), (xr, yl), (xr, yr), (xl, yr)]
                ## ------------------------------------------------

                ## STABILITY
                ## ------------------------------------------------

                if (not check_stability_bbox_simple(points_bbox)):
                    cv2.imshow("Drone View", plotted_frame)
                    cv2.waitKey(1)

                    return

                ## ------------------------------------------------

                ## Plotting
                ## ------------------------------------------------
                box_color = (0, 255, 0)
                cv2.rectangle(plotted_frame, (xl, yl), (xr, yr), box_color, 2)
                
                plot_bbox_keypoints(plotted_frame, self.goal_points_bbox)
                plot_bbox_keypoints(plotted_frame, points_bbox)
                ## ------------------------------------------------

                ## confidence
                # TODO have a plotter class with methods for each thing
                # (bbox, ellipse, points) to be plotted
                ## ------------------ 
                conf_text = f"Conf: {best_conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
                text_x = xl
                text_y = yl - 10 if yl > 20 else yl + text_size[1] + 5
                cv2.putText(plotted_frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)
                ## ------------------

                self.ibvs.set_current_points(points_bbox)
                vel = self.ibvs.compute_velocities()

                distance_from_goal = compute_distance(frame_center, (xc, yc))
                print(f"Object center {(xc, yc)} | Frame center {frame_center} | {distance_from_goal=}")
                if (distance_from_goal <= 8):
                    vel = np.zeros(6)

                ## ------------------------------------------------
                ## Drone
                ## ------------------------------------------------
                self.velocity_to_command(vel)

        # Display the final frame
        cv2.imshow("Drone View", plotted_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    ip = DroneIp.SIMULATED
    speed = 40

    controller = IBVSController(
        ip=ip,
        processor_class=IBVSYoloProcessor,
        speed=speed
    )

    if (ip == DroneIp.SIMULATED):
        controller.initialize_position()
        
    controller.run()

    print(f"STOP")
