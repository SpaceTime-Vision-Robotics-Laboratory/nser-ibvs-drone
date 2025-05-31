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

import matplotlib.pyplot as plt

from utils import e2h, plot_bbox_keypoints, compute_distance
from utils import check_stability_bbox_oriented

# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

## -------------------------------------------------------------

MODEL_PATH = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/runnable/models/yolo11n-seg_car_sim_simple.pt"

## -------------------------------------------------------------

class ImageBasedVisualServo():
    def __init__(self, K, goal_points):
        self.K = K
        # print(f"{self.K=}")
        self.Kinv = np.linalg.inv(self.K)
        # print(f"{self.Kinv=}")

        self.lambda_factor = 0.2
        
        # diagonal = [self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor]
        diagonal = [self.lambda_factor, self.lambda_factor, self.lambda_factor]
        
        self.lambda_factor = np.diag(diagonal)

        self.goal_points = goal_points
        self.goal_points_flat = np.hstack(self.goal_points)
        # self.goal_points_normalized = self.compute_normalized_image_plane_coordinates(self.goal_points)

        self.current_points = None
        self.current_points_normalized = None

        self.jcond_values = []
        self.err_values = []
        self.err_uv_values = []

        ## TODO Z should not be fixed at 45deg
        ## if depth is scalar is a problem to other angles than 90 degrees
        self.Z = 11.5
    
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
        self.current_points_flat = np.hstack(self.current_points)
        # self.current_points_normalized = self.compute_normalized_image_plane_coordinates(self.current_points)
    
    ## TODO this needs to be rewritten for other than 90deg tilt
    def compute_depths(self, pixels):
        """
        Estimate per-point depth based on image geometry.

        Args:
            pixels: Nx2 array of (u, v) pixel coordinates.
            K: 3x3 camera intrinsic matrix.
            altitude: Height of the drone in meters (h).

        Returns:
            depths: Array of Z_i depth estimates for each pixel.
        """
        depths = []

        for u, v in pixels:
            pixel_homog = np.array([u, v, 1.0])
            norm_coords = self.Kinv @ pixel_homog  # Gives [x_n, y_n, 1]
            x_n, y_n = norm_coords[0], norm_coords[1]
            Z_i = self.Z * np.sqrt(x_n**2 + y_n**2 + 1)
            depths.append(Z_i)

        return depths

    def compute_interaction_matrix(self):
        # J = np.empty((0, 6))
        J = np.empty((0, 3))

        depths = self.compute_depths(self.current_points)

        for depth, p in zip(depths, self.current_points):
            p_array = np.array([[p[0]], [p[1]]])

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]

            # J_point = self.K[:2,:2] @ np.array([
            #     [-1/depth,     0,    x/depth,  x*y,     -(1+x*x),  y],
            #     [    0, -1/depth,    y/depth,  1+y*y,   -x*y,     -x]
            # ])

            J_point = self.K[:2,:2] @ np.array([
                [-1/depth,     0,      y],
                [    0, -1/depth,     -x]
            ])

            J = np.vstack([J, J_point])
        
        # print(f"Interaction matrix shape: {J.shape}")
        print(f"{J=}")
        # print("\n ----------------------------------- \n")

        return J
    
    import numpy as np

    # def damped_pseudo_inverse(self, J, damping=0.01):
    #     """
    #     Computes the damped least-squares pseudo-inverse of a matrix J.

    #     Args:
    #         J (np.ndarray): Jacobian matrix of shape (m, n)
    #         damping (float): Damping factor (mu), small positive scalar

    #     Returns:
    #         np.ndarray: Regularized pseudo-inverse of J
    #     """
    #     JT = J.T
    #     JJ_T = J @ JT
    #     mu_squared = damping**2
    #     identity = np.eye(JJ_T.shape[0])
    #     regularized_inv = np.linalg.inv(JJ_T + mu_squared * identity)
    #     J_damped_pinv = JT @ regularized_inv
    #     return J_damped_pinv


    def compute_velocities(self, plot=False):
        J = self.compute_interaction_matrix()
        
        jcond = np.linalg.cond(J)
        self.jcond_values.append(jcond)
        print("--------------------------------------------")
        print(f"J cond: {jcond}")

        J_pinv = np.linalg.pinv(J)
        # J_pinv = self.damped_pseudo_inverse(J, damping=0.01)

        # err = self.goal_points_normalized - self.current_points_normalized
        # self.err_values.append(np.linalg.norm(err))

        err_uv = self.goal_points_flat - self.current_points_flat
        self.err_uv_values.append(np.linalg.norm(err_uv))

        # print("--------------------------------------------")
        # print(f"err: {err}")
        # print(f"err_uv: {err_uv}")
        print(f"Current: {self.current_points} \n Goal: {self.goal_points}\n")
        print(f"Current flat: {self.current_points_flat} \n Goal flat: {self.goal_points_flat}")
        print("--------------------------------------------")

        if (isinstance(self.lambda_factor, np.ndarray)):
            vel = self.lambda_factor @ J_pinv @ err_uv
        else:
            vel = self.lambda_factor * J_pinv @ err_uv
        
        print(f"{err_uv=}")
        print(f"{vel=}")

        return vel
    
    def plot_values(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
                    
        # Plot X values
        ax1.plot(range(len(self.jcond_values)), self.jcond_values, 'b-o')
        ax1.set_title('J_cond')
        ax1.set_ylabel('value')
        ax1.grid(True)
        
        # Plot Y values
        ax2.plot(range(len(self.err_uv_values)), self.err_uv_values, 'r-o')
        ax2.set_title('error')
        ax2.set_xlabel('idx')
        ax2.set_ylabel('err norm')
        ax2.grid(True)

        # ax3.plot(range(len(self.err_uv_values)), self.err_uv_values, 'g-o')
        # ax3.set_title('error')
        # ax3.set_xlabel('idx')
        # ax3.set_ylabel('err norm')
        # ax3.grid(True)

        plt.savefig("_check_errs.jpg", bbox_inches='tight')


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
        
        self.goal_points_bbox = goal_points["bbox_oriented_points"]
        self.goal_points_bbox = self.goal_points_bbox[:4]

        path_to_file = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/camera-parameters/sim-anafi-4k/intrinsic_matrix_half_size.pkl"
        with open(path_to_file, 'rb') as f:
            K = pickle.load(f)
            print(f"{K=}")
        
        self.ibvs = ImageBasedVisualServo(K, self.goal_points_bbox)

        self.max_linear_speed = 2 # m/s
        self.max_height_linear_speed = 1 # m/s
        self.max_angular_speed = np.deg2rad(60) # rad/s
    
    def compute_bbox_oriented(self, frame, xy_seg):
        object_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        # print(car_frame.shape)

        cv2.fillPoly(object_frame, pts=[xy_seg], color=255)

        contours, _ = cv2.findContours(object_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = contours[0]

        ## check https://theailearner.com/tag/cv2-minarearect/
        ## ^ (?) The 4 corner points are ordered clockwise starting from the point with the highest y.
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        # print(f"{box=}")

        return box
    
    ## TODO - temporary assumption that the points at the front are the ones with the lowest y
    def reorder_points_bbox_oriented(self, points_bbox_oriented):
        ## better (?): sorted(data,key=itemgetter(1))
        pts = sorted(points_bbox_oriented, key=lambda x: x[1])

        p0 = pts[0]
        p1 = pts[1]

        ## just a test to see if it performs a 360-like rotation when front and back are flipped
        # p0 = pts[-1]
        # p1 = pts[-2]

        points_reodered = []

        points_neighbors = points_bbox_oriented + [points_bbox_oriented[0]]
        for i in range(len(points_neighbors) - 1):
            if ((points_neighbors[i] == p0 and points_neighbors[i+1] == p1) or
                (points_neighbors[i] == p1 and points_neighbors[i+1] == p0)):
                points_reodered = points_bbox_oriented[i:] + points_bbox_oriented[:i]
                break
        
        return points_reodered

    
    def velocity_to_command(self, velocities):
        roll = 100 * velocities[0] / self.max_linear_speed
        pitch = -100 * velocities[1] / self.max_linear_speed
        
        # gaz = 100 * velocities[2] / self.max_height_linear_speed
        gaz = 0
        
        yaw = 100 * velocities[2] / self.max_angular_speed

        # roll *= 1000
        # pitch *= 1000
        # yaw *= 1000

        print(f"{roll=} | {pitch=} | {gaz=} | {yaw=}")

        # self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=0, dt=0.15)
        self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=int(yaw), dt=0.15)
        # self.drone_commander.piloting(x=0, y=0, z=0, z_rot=int(yaw), dt=0.15)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.detector.predict(frame, stream=False, verbose=False)
        return [frame, results[0]]
    
    def _display_frame(self, frame_data: list) -> None:
        ## TODO add the take-off detection conditions from Sebi // check the code from sebnae/pid_with_rot

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
            if (best_conf >= 0.6):
                ## Segmentation
                ## ------------------------------------------------
                xy_seg = predictions.masks.xy[best_idx]
                xy_seg = [list(xy) for xy in xy_seg]
                xy_seg = np.array(xy_seg).astype(np.int32)
                
                # seg_color = (0, 200, 0)
                # cv2.fillPoly(plotted_frame, pts=[xy_seg], color=seg_color)
                ## ------------------------------------------------

                ## Bounding box ORIENTED
                ## ------------------------------------------------
                bbox_oriented = self.compute_bbox_oriented(frame_data[0], xy_seg)
                points_bbox_oriented = [tuple(p) for p in bbox_oriented]

                ## TODO reorder bbox points
                points_bbox_oriented = self.reorder_points_bbox_oriented(points_bbox_oriented)
                
                points_bbox_oriented_int = [tuple(map(int, p)) for p in points_bbox_oriented]

                xl, yl = points_bbox_oriented_int[0]
                xr, yr = points_bbox_oriented_int[2]

                xc = (xl + xr) // 2
                yc = (yl + yr) // 2
                cv2.circle(plotted_frame, (xc, yc), 5, (255, 255, 0), -1)
                ## ------------------------------------------------

                ## STABILITY
                ## ------------------------------------------------

                # if (not check_stability_bbox_oriented(points_bbox_oriented)):
                #     cv2.imshow("Drone View", plotted_frame)
                #     cv2.waitKey(1)

                #     return

                ## ------------------------------------------------

                ## Plotting
                ## ------------------------------------------------
                box_color = (0, 255, 0)
                # cv2.rectangle(plotted_frame, (xl, yl), (xr, yr), box_color, 2)
                cv2.drawContours(plotted_frame, [np.int0(bbox_oriented)], 0, (36,255,12), 3)
                
                plot_bbox_keypoints(plotted_frame, self.goal_points_bbox)
                plot_bbox_keypoints(plotted_frame, points_bbox_oriented_int)
                ## ------------------------------------------------

                ## Compute and display depths for each point
                ## ------------------------------------------------
                depths = self.ibvs.compute_depths(points_bbox_oriented_int)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                depth_color = (255, 255, 255)  # White text
                
                for i, (point, depth) in enumerate(zip(points_bbox_oriented_int, depths)):
                    x, y = point
                    depth_text = f"Z{i}: {depth:.2f}m"
                    
                    # Calculate text size for background rectangle
                    text_size, _ = cv2.getTextSize(depth_text, font, font_scale, font_thickness)
                    text_w, text_h = text_size
                    
                    # Position text with offset from point
                    text_x = x + 10
                    text_y = y - 10 if y > 30 else y + text_h + 10
                    
                    # Draw background rectangle for better readability
                    cv2.rectangle(plotted_frame, 
                                (text_x - 2, text_y - text_h - 2), 
                                (text_x + text_w + 2, text_y + 2), 
                                (0, 0, 0), -1)
                    
                    # Draw the depth text
                    cv2.putText(plotted_frame, depth_text, (text_x, text_y), 
                            font, font_scale, depth_color, font_thickness, cv2.LINE_AA)
                ## ------------------------------------------------

                ## confidence
                # TODO have a plotter class with methods for each thing
                # (bbox, ellipse, points) to be plotted
                ## ------------------ 
                # conf_text = f"Conf: {best_conf:.2f}"
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.6
                # font_thickness = 1
                # text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
                # text_x = xl
                # text_y = yl - 10 if yl > 20 else yl + text_size[1] + 5
                # cv2.putText(plotted_frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)
                ## ------------------

                # self.ibvs.set_current_points(points_bbox_oriented)
                self.ibvs.set_current_points(points_bbox_oriented_int)
                vel = self.ibvs.compute_velocities()
                # print(vel)

                distance_from_goal = compute_distance(frame_center, (xc, yc))
                # print(f"Object center {(xc, yc)} | Frame center {frame_center} | {distance_from_goal=}")
                # if (distance_from_goal <= 8):
                #     vel = np.zeros(6)
                
                if (self._frame_count % 120 == 0):
                    self.ibvs.plot_values()

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
