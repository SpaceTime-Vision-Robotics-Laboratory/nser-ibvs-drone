from pathlib import Path
import sys
import time
import argparse
import ultralytics
import numpy as np
import cv2
import pandas as pd
import os

from pathlib import Path
import json

# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

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

        self.drone_commander.move_by(forward=0, right=5, down=-10, rotation=0)

        self.frame_processor.frame_queue.empty()
        
        print("Initialization complete. Ready for tracking.")

class IBVSYoloProcessor(BaseVideoProcessor):
    def __init__(self, model_path="/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/models/yolo11n-seg_car_sim_simple.pt", **kwargs):
        super().__init__(**kwargs)
        self.detector = ultralytics.YOLO(model_path)

        self.offset_threshold = 8 # Deadband for x/y offset corrections

        self.kp_rot = 100 # Proportional gain for rotation (yaw) based on x_offset
        self.kp_alt = 0   # Proportional gain for altitude (z) based on y_offset (DISABLED)
        
        self.kp_fwd_x = -0.05 # Proportional gain for roll (x) based on x_offset
        self.kd_fwd_x = -0.001  # Derivative gain for roll (x)
        
        self.kp_fwd_y = -0.05 # Proportional gain for forward (y) based on y_offset
        self.kd_fwd_y = -0.001  # Derivative gain for pitch (y)

        self.previous_x_offset = 0.0
        self.previous_y_offset = 0.0

        self.last_command_time = time.time() # Initialize last time for dt calc

        ## Goal frame
        ## -------------------------------------

        goal_frame_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/images/frame_drone_sim_10m_center.png"
        
        self.goal_frame_points_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/data/frame_drone_sim_10m_center.json"
        with open(self.goal_frame_points_path, "r") as f:
            goal_points = json.load(f)
        
        ## bbox
        ## ---------------------------------
        self.goal_points_bbox = goal_points["bbox_points"]
        self.goal_points_bbox = self.goal_points_bbox[:4]
        
        goal_points_bbox_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in self.goal_points_bbox])
        self.goal_points_bbox_xy = np.hstack(goal_points_bbox_xy)
        ## ---------------------------------

        ## ellipse
        ## ---------------------------------
        self.goal_points_ellipse = goal_points["ellipse_points"]
        self.goal_points_ellipse = self.goal_points_ellipse[:4]
        
        goal_points_ellipse_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in self.goal_points_ellipse])
        self.goal_points_ellipse_xy = np.hstack(goal_points_ellipse_xy)
        ## ---------------------------------

        self.goal_colors = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255), (51,87,255)]

        ## -------------------------------------

        self.frame_mod = 8
        self.frame_count = 0

        ## IBVS

        self.lambda_factor = 0.5

        # with open("missed.txt", "w") as f:
        #     f.write("Missed objects frames\n")

        with open("rpy.txt", "w") as f:
            f.write("roll-pitch-yaw\n")

        with open("xy.txt", "w") as f:
            f.write("xy\n")
        
        with open("cmd.txt", "w") as f:
            f.write("commands\n")

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

        rect = cv2.minAreaRect(cont)
        box = np.int0(cv2.boxPoints(rect))

        ellipse = cv2.fitEllipse(cont)
        return ellipse, box
    
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

        return kp_MA1, \
               kp_MA2, \
               kp_m1, \
               kp_m2
    
    def plot_bbox_keypoints(self, plotted_frame, keypoints):
        kp_M1, kp_M2, kp_m1, kp_m2 = keypoints
        # kp_M1, kp_M2, kp_m1 = keypoints

        cv2.circle(plotted_frame, kp_m1, 5, (255, 0, 0), -1)
        cv2.circle(plotted_frame, kp_m2, 5, (0,255,0), -1)

        cv2.circle(plotted_frame, kp_M1, 5, (0,0,255), -1)
        cv2.circle(plotted_frame, kp_M2, 5, (255, 255, 255), -1)

    def plot_ellipse_keypoints(self, plotted_frame, keypoints):
        kp_M1, kp_M2, kp_m1, kp_m2 = keypoints

        cv2.line(plotted_frame, tuple(kp_m1.astype(np.int32)), tuple(kp_m2.astype(np.int32)), (255, 255, 0), 1)
        cv2.circle(plotted_frame, tuple(kp_m1.astype(np.int32)), 5, (255, 0, 0), -1)
        cv2.circle(plotted_frame, tuple(kp_m2.astype(np.int32)), 5, (0,255,0), -1)

        cv2.line(plotted_frame, tuple(kp_M1.astype(np.int32)), tuple(kp_M2.astype(np.int32)), (255, 255, 0), 1)
        cv2.circle(plotted_frame, tuple(kp_M1.astype(np.int32)), 5, (0,0,255), -1)
        cv2.circle(plotted_frame, tuple(kp_M2.astype(np.int32)), 5, (255, 255, 255), -1)
        

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
    
    def compute_velocities_ibvs(self, points, goal_points_xy):
        J = self.compute_interaction_matrix(points)
        J_pinv = np.linalg.pinv(J)

        points_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in points])
        points_xy = np.hstack(points_xy)

        e = goal_points_xy - points_xy
        # print(f"Current: {points_xy} | Goal: {self.goal_points_xy}")

        vel = self.lambda_factor * J_pinv.dot(e)
        return vel
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]
    
    def _generate_ibvs_command(self, velocities, object_center, target_lost=False):
        dt = 0.15
        
        current_time = time.time()
        # Calculate dt, handle potential first run or zero dt
        dt = current_time - self.last_command_time
        self.last_command_time = current_time
        if dt <= 0.001: # Avoid division by zero or excessively large derivatives
            dt = 0.15 # Assume a nominal dt if issue occurs
        print(f"dt: {dt}")

        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0

        derivative_x_term = 0
        proportional_x_term = 0

        derivative_y_term = 0
        proportional_y_term = 0

        max_speed_command = 30
        max_rot_command = 50

        if not target_lost and object_center is not None:
            # z_rot = int(proportional_rot_term + derivative_rot_term)
            ## TODO vel[2] or vel[5] but with a larger factor
            if (abs(velocities[2]) > 5):
                z_rot = int(-self.kp_rot * velocities[5])
            else:
                z_rot = int(self.kp_rot * velocities[5])
            z_rot = max(-max_rot_command, min(max_rot_command, z_rot)) ## yaw - rotate towards the target
            print(f"z_rot: {z_rot}")

            ## ----------------------------------------------
            ## X Movement
            ## ----------------------------------------------

            delta_x = velocities[0] - self.previous_x_offset
            derivative_x_term = self.kd_fwd_x * (delta_x / dt)
            self.previous_x_offset = velocities[0]

            proportional_x_term = self.kp_fwd_x * velocities[0]

            x_movement = int(proportional_x_term + derivative_x_term)
            # x_movement = int(proportional_x_term)
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
            # y_movement = int(proportional_y_term)
            y_movement = max(-max_speed_command, min(max_speed_command, y_movement)) ## pitch - front/back
            print(f"y_movement: {proportional_y_term} + {derivative_y_term} | {y_movement}")

            ## ----------------------------------------------
        # else:
        #     self.previous_x_offset = 0.0
        #     self.previous_y_offset = 0.0


        z_movement = max(-max_speed_command, min(max_speed_command, z_movement)) ## up/down for drone (gaz) - _no_

        with open("cmd.txt", "a") as f:
            f.write(f"velocities: {velocities}\n")
            f.write(f"z_rot: {z_rot}\n")
            f.write(f"delta_x: {delta_x} / dt: {dt}\n")
            f.write(f"x_movement: {proportional_x_term} + {derivative_x_term} | {x_movement}\n")
            f.write(f"delta_y: {delta_y} / dt: {dt}\n")
            f.write(f"y_movement: {proportional_y_term} + {derivative_y_term} | {y_movement}\n\n")

        if not target_lost:
            self.drone_commander.piloting(x=x_movement, y=y_movement, z=z_movement, z_rot=z_rot, dt=0.15)
    
    def _display_frame(self, frame_data: list) -> None:
        original_frame = frame_data[0].copy()

        plotted_frame = original_frame.copy()
        # cv2.rectangle(plotted_frame, self.goal_points_bbox[0], self.goal_points_bbox[3], (255, 0, 0), 1)
        # for point in self.goal_points_ellipse:
        #     # print(point)
        #     cv2.circle(plotted_frame, tuple(point), 10, (51,87,255), -1)

        frame_h, frame_w = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        
        frame_center = (frame_center_x, frame_center_y)

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

        vel = np.zeros(6)
        predictions = frame_data[1]
        if predictions.boxes and predictions.boxes.conf is not None and len(predictions.boxes.conf) > 0:
            best_idx = predictions.boxes.conf.argmax()
            # print(f"Best idx {best_idx} / {len(predictions.boxes.conf)}")
            
            best_conf = predictions.boxes.conf[best_idx]
            if (best_conf >= 0.7):
                target_lost = False

                ## Segmentation
                ## ------------------------------------------------
                xy_seg = predictions.masks.xy[best_idx]
                xy_seg = [list(xy) for xy in xy_seg]
                xy_seg = np.array(xy_seg).astype(np.int32)
                
                
                seg_color = (0, 200, 0)
                cv2.fillPoly(plotted_frame, pts=[xy_seg], color=seg_color)
                ## ------------------------------------------------

                ## Bounding box
                ## ------------------------------------------------
                xyxy = predictions.boxes.xyxy[best_idx].cpu().numpy()
                xl, yl, xr, yr = map(int, xyxy)

                box_color = (0, 255, 0)
                box_thickness = 3
                cv2.rectangle(plotted_frame, (xl, yl), (xr, yr), box_color, box_thickness)

                points_bbox = [(xl, yl), (xl, yr), (xr, yl), (xr, yr)]
                # points_bbox = points_bbox[:3]
                self.plot_bbox_keypoints(plotted_frame, tuple(points_bbox))

                self.plot_bbox_keypoints(plotted_frame, tuple(self.goal_points_bbox))
                ## ------------------------------------------------
                
                xc = (xl + xr) // 2
                yc = (yl + yr) // 2
                object_center = (xc, yc)
                cv2.circle(plotted_frame, object_center, 5, (255, 0, 0), -1)
                cv2.line(plotted_frame, frame_center, object_center, (0, 165, 255), 2)

                with open("xy.txt", "a") as f:
                    f.write(f"{object_center[0]} {object_center[1]}\n")

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

                ## ellipse
                ## ------------------

                ellipse, box = self.compute_oriented_ellipse(original_frame, xy_seg)

                (xc_ellipse,yc_ellipse), (MA,ma), angle = ellipse
                cv2.ellipse(
                    img=plotted_frame,
                    center=(int(xc_ellipse),int(yc_ellipse)),
                    axes=(int(MA/2), int(ma/2)),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=(0,0,255)
                )

                kp_M1, kp_M2, kp_m1, kp_m2 = self.compute_ellipse_axis_keypoints(ellipse)
                # self.plot_ellipse_keypoints(plotted_frame, (kp_M1, kp_M2, kp_m1, kp_m2))
                points_ellipse = [tuple(kp_m1), tuple(kp_m2), tuple(kp_M1), tuple(kp_M2)]

                ## ------------------

                ## velocity
                ## ------------------
                
                distance_from_goal = self.find_distance(frame_center, object_center)
                print(f"Object center {object_center} | Frame center {frame_center} | distance from goal: {distance_from_goal}")

                vel = self.compute_velocities_ibvs(points_bbox, self.goal_points_bbox_xy)
                print(f"Velocities bbox: {vel}")

                vel_ellipse = self.compute_velocities_ibvs(points_ellipse, self.goal_points_ellipse_xy)
                print(f"Velocities ellipse: {vel_ellipse}")

                vel[2] = vel_ellipse[2]
                vel[5] = vel_ellipse[5]

                if (distance_from_goal <= self.offset_threshold):
                    vel = np.zeros(6)
                
                # if (abs(vel[0]) > 100 or abs(vel[1]) > 100):
                #     cv2.imwrite("check_frame.png", plotted_frame)
                #     with open("check_vel.txt", "w") as f:
                #         f.write(f"{vel}\n")

                with open("rpy.txt", "a") as f:
                    f.write(f"{vel[0]} {vel[1]} {vel[5]}\n")

                ## ------------------
        else:
            with open("missed.txt", "a") as f:
                f.write("1")

        
        cv2.circle(plotted_frame, frame_center, 7, (0, 0, 255), -1)

        # Display the final frame
        cv2.imshow("Drone View", plotted_frame)
        cv2.waitKey(1)

        # self.frame_count += 1
        # if (self.frame_count % self.frame_mod == 0):
        #     self._generate_ibvs_command(vel, object_center, target_lost)
        #     self.frame_count = 0
        self._generate_ibvs_command(vel, object_center, target_lost)


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
