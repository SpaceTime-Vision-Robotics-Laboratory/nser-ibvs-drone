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

import supervision as sv

## Constants
## --------------------------------

# in mm
PIXEL_WIDTH = 0.001078
PIXEL_HEIGHT = 0.001069

FOCAL_LENGTH = 465.60298

Px = FOCAL_LENGTH / PIXEL_WIDTH
Py = FOCAL_LENGTH / PIXEL_HEIGHT

U0 = 320
V0 = 180

Z = 10

## --------------------------------

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

        # self.drone_commander.move_by(forward=0, right=0, down=-5, rotation=0)
        for _ in range(8):
            self.drone_commander.piloting(x=0, y=0, z=100, z_rot=0, dt=1)
            time.sleep(2)

        self.frame_processor.frame_queue.empty()
        
        print("Initialization complete. Ready for tracking.")

class IBVSYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path="/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/models/best__yolo11n-obb_sim_01.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = ultralytics.YOLO(model_path)

        self.offset_threshold = 2 # Deadband for x/y offset corrections

        self.kp_rot = 20 # Proportional gain for rotation (yaw) based on x_offset
        self.kp_alt = 0   # Proportional gain for altitude (z) based on y_offset (DISABLED)
        
        self.kp_fwd_x = -0.01 # Proportional gain for forward (y) based on y_offset
        self.kd_fwd_x = -0.01  # Derivative gain for rotation (damping)
        
        self.kp_fwd_y = -0.01 # Proportional gain for forward (y) based on y_offset
        self.kd_fwd_y = -0.01 # Proportional gain for forward (y) based on y_offset

        self.previous_x_offset = 0.0
        self.previous_y_offset = 0.0

        self.last_command_time = time.time() # Initialize last time for dt calc

        ## IBVS

        self.lambda_factor = 0.5

    def find_distance(self, p1, p2):
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)

        return np.linalg.norm(p1_arr - p2_arr)
    
    def compute_oriented_ellipse(self, frame, xy_seg):
        object_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        # print(car_frame.shape)

        poly = cv2.fillPoly(object_frame, pts=[xy_seg], color=255)
        # print(poly)
        cv2.imwrite("car.png", object_frame)

        contours, hierarchy = cv2.findContours(object_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = contours[0]

        ellipse = cv2.fitEllipse(cont)
        return ellipse
    
    def compute_ellipse_axis_keypoints(self, ellipse):
        (xc_ellipse,yc_ellipse), (MA,ma), angle = ellipse

        kp_center = np.array([xc_ellipse, yc_ellipse])

        MA, ma = MA / 2, ma / 2
        angle_rad = np.radians(angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        M_dx, M_dy = MA * cos_a, MA * sin_a
        m_dx, m_dy = ma * -sin_a, ma * cos_a

        kp_MA1 = kp_center + np.array([M_dx, M_dy])
        kp_MA2 = kp_center - np.array([M_dx, M_dy])
        kp_m1 = kp_center + np.array([m_dx, m_dy])
        kp_m2 = kp_center - np.array([m_dx, m_dy])

        return kp_MA1.astype(np.int32), \
               kp_MA2.astype(np.int32), \
               kp_m1.astype(np.int32), \
               kp_m2.astype(np.int32)
        

    def convert_pixel_to_image_plane_coordinates(self, u, v):
        x = (u - U0) / Px
        y = (v - V0) / Py

        return (x, y)
    
    def compute_interaction_matrix(self, points):
        ## TODO set a proper value for Z
        Z = 10

        J_list = []
        for u, v in points:
            x, y = self.convert_pixel_to_image_plane_coordinates(u, v)
            J_point = np.array([
                [-1/Z,     0,    x/Z,  x*y,     -(1+x*x),  y],
                [    0, -1/Z,    y/Z,  1+y*y,   -x*y,     -x]
            ])

            J_list.append(J_point)
        
        J = np.vstack(J_list)
        # print(f"Interaction matrix shape: {J.shape}")
        # print(f"Interaction matrix : {J}")
        # print("\n ----------------------------------- \n")

        return J

    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]
    
    def _generate_ibvs_command(self, velocities, object_center, target_lost=False):
        # dt = 0.15
        current_time = time.time()
        # Calculate dt, handle potential first run or zero dt
        dt = current_time - self.last_command_time
        if dt <= 0.001: # Avoid division by zero or excessively large derivatives
            dt = 0.15 # Assume a nominal dt if issue occurs
        self.last_command_time = current_time

        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0

        derivative_x_term = 0 # Initialize derivative term
        proportional_x_term = 0

        max_speed_command = 30
        max_rot_command = 50

        if not target_lost and object_center is not None:
            xc, yc = object_center

            # z_rot = int(proportional_rot_term + derivative_rot_term)
            z_rot = max(-max_rot_command, min(max_rot_command, z_rot)) ## yaw - rotate towards the target

            ## ----------------------------------------------
            ## X Movement
            ## ----------------------------------------------

            delta_x = velocities[0] - self.previous_x_offset
            derivative_x_term = self.kd_fwd_x * (delta_x / dt)
            self.previous_x_offset = velocities[0]

            proportional_x_term = self.kp_fwd_x * velocities[0]

            # x_movement = int(proportional_rot_term)
            x_movement = int(proportional_x_term + derivative_x_term)
            x_movement = max(-max_speed_command, min(max_speed_command, x_movement)) ## TODO check - no roll normally
            print(f"x_movement: {proportional_x_term} + {derivative_x_term} | {x_movement}")

            ## ----------------------------------------------

            # # --- P Control for Altitude (Disabled) ---
            # if abs(y_offset) > self.offset_threshold:
            #     z_movement = -int(self.kp_alt * y_offset) # Logic remains, gain is 0

            ## ----------------------------------------------
            ## Y Movement
            ## ----------------------------------------------

            delta_y = velocities[1] - self.previous_y_offset
            derivative_y_term = self.kd_fwd_y * (delta_y / dt)
            self.previous_y_offset = velocities[1]

            proportional_y_term = self.kp_fwd_y * velocities[1]

            # --- P Control for Forward/Backward ---
            y_movement = int(proportional_y_term + derivative_y_term)
            y_movement = max(-max_speed_command, min(max_speed_command, y_movement)) ## pitch - front/back
            print(f"y_movement: {proportional_y_term} + {derivative_y_term} | {y_movement}")

            ## ----------------------------------------------


        z_movement = max(-max_speed_command, min(max_speed_command, z_movement)) ## up/down for drone (gaz) - _no_

        if not target_lost:
            self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.15)
    
    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()
        plotted_frame = original_frame.copy()

        frame_h, frame_w = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        
        frame_center = (frame_center_x, frame_center_y)

        frame_dimensions = (frame_w, frame_h)

        ## used for display
        ## ----------------
        padding = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (255, 0, 0)

        status_text = '' # text shown at the bottom-left of the screen
        ## ----------------
        
        target_lost = True
        object_center = None

        ## Goal

        goal_w = 361 - 275
        goal_h = 264 - 36

        goal_xl = frame_center_x - (goal_w // 2)
        goal_yl = frame_center_y - (goal_h // 2)

        goal_xr = frame_center_x + (goal_w // 2)
        goal_yr = frame_center_y + (goal_h // 2)

        # goal_points = [(goal_xl, goal_yl), (goal_xl, goal_yr), (goal_xr, goal_yl), (goal_xr, goal_yr)]
        goal_points = [(goal_xl, goal_yr), (goal_xr, goal_yr), (goal_xr, goal_yl), (goal_xl, goal_yl)]
        
        goal_points_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in goal_points])
        goal_points_xy = np.hstack(goal_points_xy)

        # cv2.rectangle(plotted_frame, (goal_xl, goal_yl), (goal_xr, goal_yr), (255, 0, 0), 1)

        ## -------------------------

        vel = np.zeros(6)

        predictions = frame_data[1].obb
        if len(predictions.conf) > 0:
            best_idx = predictions.conf.argmax()
            # print(f"Best idx {best_idx} / {len(predictions.boxes.conf)}")
            
            best_conf = predictions.conf[best_idx]
            if (best_conf >= 0.7):
                target_lost = False

                xyxy = predictions.xyxy[best_idx].cpu().numpy()
                xl, yl, xr, yr = map(int, xyxy)
                # points = [(xl, yl), (xl, yr), (xr, yl), (xr, yr)]

                points = predictions.xyxyxyxy[best_idx].cpu().numpy()
                points = [(p[0], p[1]) for p in points]

                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,255)]
                print(f"Points: {points}")
                for p, color in zip(points, colors):
                    p_int = tuple(map(int, p))
                    cv2.circle(plotted_frame, p_int, 5, color, -1)

                ## Jacobian

                J = self.compute_interaction_matrix(points)
                J_pinv = np.linalg.pinv(J)

                points_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in points])
                points_xy = np.hstack(points_xy)

                e = goal_points_xy - points_xy
                # print(f"Current: {points_xy} | Goal: {goal_points_xy}")

                vel = self.lambda_factor * J_pinv.dot(e)
                print(f"Velocities: {vel}")

                ## ----------------------------------------

                # print(f"{(xl, yl), (xr, yr)}")

                box_color = (0, 255, 0)
                box_thickness = 3
                # cv2.rectangle(plotted_frame, (xl, yl), (xr, yr), box_color, box_thickness)

                box = predictions.xyxyxyxy[best_idx].cpu().numpy()
                box = np.array(box).astype(int)
                cv2.drawContours(plotted_frame, [box], 0, (36,255,12), 3)

                # rect = predictions.xywhr[best_idx].cpu().numpy()
                # ((center_x, center_y), (dim_x, dim_y), angle)
                # rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
                # box = cv2.boxPoints(rect)
                # print(f"Box: {box} | {np.int0(box)}")
                # cv2.drawContours(plotted_frame, [np.int0(box)], 0, (36,255,12), 3)

                ## --------------------------------------
                # detections = sv.Detections.from_ultralytics(frame_data[1])
                # print(f"Detections {detections} | {box}")

                # oriented_box_annotator = sv.OrientedBoxAnnotator()
                # annotated_frame = oriented_box_annotator.annotate(
                #     scene=original_frame,
                #     detections=detections
                # )

                # cv2.imwrite("check_obb.png", annotated_frame)
                ## --------------------------------------
                
                xc = (xl + xr) // 2
                yc = (yl + yr) // 2
                object_center = (xc, yc)

                distance_from_goal = self.find_distance(frame_center, object_center)
                print(f"distance from goal: {distance_from_goal}")
                if (distance_from_goal < self.offset_threshold):
                    vel = np.zeros(6)

                best_center = (xc, yc)
                cv2.circle(plotted_frame, best_center, 5, (255, 0, 0), -1)

                cv2.line(plotted_frame, frame_center, best_center, (0, 165, 255), 2)

                ## confidence
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

                status_text = f"Object found: {(xc, yc)}"
                text_color = (0, 255, 0)
            else:
                status_text = f"Object not found"
                text_color = (0, 0, 255)
        
        cv2.circle(plotted_frame, frame_center, 7, (0, 0, 255), -1)

        (w, h), _ = cv2.getTextSize(status_text, font, font_scale, font_thickness)
        text_x, text_y = 2*padding, frame_h - (h + 2*padding)
        cv2.putText(plotted_frame, status_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.rectangle(plotted_frame, (goal_xl, goal_yl), (goal_xr, goal_yr), (255, 0, 0), 1)

        # Display the final frame
        cv2.imshow("Drone View", plotted_frame)
        cv2.waitKey(1)

        # self._generate_ibvs_command(vel, object_center, target_lost)



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
