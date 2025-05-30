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

import math

import matplotlib.pyplot as plt

# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

## Constants
## --------------------------------

# in mm
# PIXEL_WIDTH = 0.001078
# PIXEL_HEIGHT = 0.001069

FOCAL_LENGTH = 465.60298

PIXEL_WIDTH = 1 / FOCAL_LENGTH
PIXEL_HEIGHT = 1 / FOCAL_LENGTH

Px = FOCAL_LENGTH / PIXEL_WIDTH
Py = FOCAL_LENGTH / PIXEL_HEIGHT


U0 = 320
V0 = 180

## --------------------------------

def e2h(v):
    """
    Convert from Euclidean to homogeneous form

    :param v: Euclidean vector or matrix
    :type v: array_like(n), ndarray(n,m)
    :return: homogeneous vector
    :rtype: ndarray(n+1,m)

    - If ``v`` is an N-vector, return an (N+1)-column vector where a value of 1 has
      been appended as the last element.
    - If ``v`` is a matrix (NxM), return a matrix (N+1xM), where each column has
      been appended with a value of 1, ie. a row of ones has been appended to the matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> e2h([2, 4, 6])
        >>> e = np.c_[[1,2], [3,4], [5,6]]
        >>> e
        >>> e2h(e)

    .. note:: The result is always a 2D array, a 1D input results in a column vector.

    :seealso: e2h
    """
    if isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return np.vstack([v, np.ones((1, v.shape[1]))])

    elif isvector(v):
        # dealing with shape (N,) array
        v = getvector(v, out="col")
        return np.vstack((v, 1))

    else:
        raise ValueError("bad type")

def isvector(v, dim) -> bool:
    """
    Test if argument is a real vector

    :param v: value to test
    :param dim: required dimension
    :type dim: int or None
    :return: whether value is a valid vector
    :rtype: bool

    - ``isvector(vec)`` is ``True`` if ``vec`` is a vector, ie. any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``isvector(vec, N)`` as above but must also be an ``N``-element vector.

    .. runblock:: pycon

        >>> from spatialmath.base import isvector
        >>> import numpy as np
        >>> isvector([1,2])  # list
        >>> isvector((1,2))  # tuple
        >>> isvector(np.r_[1,2,3])  # numpy array
        >>> isvector(1)  # scalar
        >>> isvector([1,2], 3)  # list

    :seealso: :func:`getvector`, :func:`assertvector`
    """
    if (
        isinstance(v, (list, tuple))
        and (dim is None or len(v) == dim)
        and all(map(lambda x: isinstance(x, _scalartypes), v))
    ):
        return True  # list or tuple

    if isinstance(v, np.ndarray):
        s = v.shape
        if dim is None:
            return (
                (len(s) == 1 and s[0] > 0)
                or (s[0] == 1 and s[1] > 0)
                or (s[0] > 0 and s[1] == 1)
            )
        else:
            return s == (dim,) or s == (1, dim) or s == (dim, 1)

    if (dim is None or dim == 1) and isinstance(v, _scalartypes):
        return True

    return False

def getvector(
    v,
    dim,
    out,
    dtype,
):
    """
    Return a vector value

    :param v: passed vector
    :param dim: required dimension, or None if any length is ok
    :type dim: int or None
    :param out: output format, default is 'array'
    :type out: str
    :param dtype: datatype for numPy array return (default np.float64)
    :type dtype: numPy type
    :return: vector value in specified format
    :raises TypeError: value is not a list or NumPy array
    :raises ValueError: incorrect number of elements

    - ``getvector(vec)`` is ``vec`` converted to the output format ``out``
      where ``vec`` is any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``getvector(vec, N)`` as above but must be an ``N``-element vector.

    The returned vector will be in the format specified by ``out``:

    ==========  ===============================================
    format      return type
    ==========  ===============================================
    'sequence'  Python list, or tuple if a tuple was passed in
    'list'      Python list
    'array'     1D numPy array, shape=(N,)  [default]
    'row'       row vector, a 2D numPy array, shape=(1,N)
    'col'       column vector, 2D numPy array, shape=(N,1)
    ==========  ===============================================

    .. runblock:: pycon

        >>> from spatialmath.base import getvector
        >>> import numpy as np
        >>> getvector([1,2])  # list
        >>> getvector([1,2], out='row')  # list
        >>> getvector([1,2], out='col')  # list
        >>> getvector((1,2))  # tuple
        >>> getvector(np.r_[1,2,3], out='sequence')  # numpy array
        >>> getvector(1)  # scalar
        >>> getvector([1])
        >>> getvector([[1]])
        >>> getvector([1,2], 2)
        >>> # getvector([1,2], 3)  --> ValueError

    .. note::
        - For 'array', 'row' or 'col' output the NumPy dtype defaults to the
          ``dtype`` of ``v`` if it is a NumPy array, otherwise it is
          set to the value specified by the ``dtype`` keyword which defaults
          to ``np.float64``.
        - If ``v`` is symbolic the ``dtype`` is retained as ``'O'``

    :seealso: :func:`isvector`
    """
    dt = dtype

    if isinstance(v, _scalartypes):  # handle scalar case
        v = [v]  # type: ignore
    if isinstance(v, (list, tuple)):
        # list or tuple was passed in

        # if issymbol(v):
        #     dt = None

        if dim is not None and v and len(v) != dim:
            raise ValueError(
                "incorrect vector length: expected {}, got {}".format(dim, len(v))
            )
        if out == "sequence":
            return v
        elif out == "list":
            return list(v)
        elif out == "array":
            return np.array(v, dtype=dt)
        elif out == "row":
            return np.array(v, dtype=dt).reshape(1, -1)
        elif out == "col":
            return np.array(v, dtype=dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")

    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim is not None:
            if not (s == (dim,) or s == (1, dim) or s == (dim, 1)):
                raise ValueError(
                    "incorrect vector length: expected {}, got {}".format(dim, s)
                )

        v = v.flatten()

        if v.dtype.kind == "O":
            dt = "O"

        if out in ("sequence", "list"):
            return list(v.flatten())
        elif out == "array":
            return v.astype(dt)
        elif out == "row":
            return v.astype(dt).reshape(1, -1)
        elif out == "col":
            return v.astype(dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise TypeError("invalid input type")

# def get_rectangle_sides(corners) -> tuple[int, int]:
#     """
#     :param corners: list of 4 (x, y) tuples representing rectangle corners
#     Returns: Tuple of (short_side_length, long_side_length)
#     """
#     distances = []
#     for i in range(4):
#         x1, y1 = corners[i]
#         x2, y2 = corners[(i + 1) % 4] 
#         distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         distances.append(distance)
    
#     unique_distances = list(set(round(d, 6) for d in distances))
#     print(f"{unique_distances=}")
    
#     if len(unique_distances) != 2:
#         raise ValueError("Invalid rectangle: should have exactly 2 unique side lengths")
    
#     short_side = min(unique_distances)
#     long_side = max(unique_distances)
    
#     return short_side, long_side

def get_rectangle_sides(corners) -> tuple[float, float]:
    """
    :param corners: list of 4 (x, y) tuples representing rectangle corners
    :returns: Tuple of (short_side_length, long_side_length)
    """
    # Calculate only two adjacent sides
    x1, y1 = corners[0]
    x2, y2 = corners[1] 
    x3, y3 = corners[2]
    
    side1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    side2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    
    if side1 < side2:
        return side1, side2
    return side2, side1

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

        self.drone_commander.move_by(forward=0, right=2, down=-10, rotation=0)
        # self.drone_commander.move_by(forward=0, right=1, down=0, rotation=0) # for the bunker env

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

        xc = (self.goal_points_bbox[0][0] + self.goal_points_bbox[3][0]) / 2
        yc = (self.goal_points_bbox[0][1] + self.goal_points_bbox[3][1]) / 2
        self.goal_points_bbox.append((xc, yc))
        
        goal_points_bbox_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in self.goal_points_bbox])
        # self.goal_points_bbox_xy = np.hstack(goal_points_bbox_xy)
        
        path_to_file = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/camera-parameters/sim-anafi-4k/intrinsic_matrix_half_size.pkl"
        with open(path_to_file, 'rb') as f:
            self.K = pickle.load(f)
        
        print(f"{self.K=}")
        self.Kinv = np.linalg.inv(self.K)

        goal_new = []
        for p in self.goal_points_bbox:
            p_array = np.array([[p[0]], [p[1]]])

            # print(p_array, p_array.shape)

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]
            goal_new.append((x, y))
        
        goal_new = np.array(goal_new)
        goal_new = np.hstack(goal_new)

        # print(f"old={self.goal_points_bbox_xy}")
        # print(f"new={goal_new}")

        self.goal_points_bbox_xy = goal_new

        ## ---------------------------------

        ## ellipse
        ## ---------------------------------
        self.goal_points_ellipse = [np.array(p) for p in goal_points["ellipse_points"]]
        self.goal_points_ellipse = self.goal_points_ellipse[:4]
        
        goal_points_ellipse_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in self.goal_points_ellipse])

        goal_new_ellipse = []
        for p in self.goal_points_ellipse:
            p_array = np.array([[p[0]], [p[1]]])

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]
            goal_new_ellipse.append((x, y))
        
        goal_new_ellipse = np.array(goal_new_ellipse)
        goal_new_ellipse = np.hstack(goal_new_ellipse)

        self.goal_points_ellipse_xy = goal_new_ellipse
        ## ---------------------------------

        self.goal_colors = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255), (51,87,255)]

        ## -------------------------------------

        self.frame_mod = 8
        self.frame_count = 0

        ## IBVS

        ## TODO integrate the entire IBVS logic into this class
        # self.ibvs = ImageBasedVisualServo()

        self.lambda_factor = 0.1

        diagonal = [self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor, self.lambda_factor * 2]
        self.lambda_factor = np.diag(diagonal)

        self.max_linear_speed = 2 # m/s
        self.max_height_linear_speed = 1 # m/s
        self.max_angular_speed = np.deg2rad(60) # rad/s

        ## -----------------------------

        self.jconds = []
        self.errs = []
    
    def velocity_to_command(self, velocities):
        roll = 100 * velocities[0] / self.max_linear_speed
        pitch = -100 * velocities[1] / self.max_linear_speed
        gaz = 100 * velocities[2] / self.max_height_linear_speed
        yaw = 100 * velocities[5] / self.max_angular_speed

        roll *= 1000
        pitch *= 1000
        yaw *= 1000

        print(f"{roll=} | {pitch=} | {gaz=} | {yaw=}")

        # self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=0, dt=0.15)
        self.drone_commander.piloting(x=int(roll), y=int(pitch), z=0, z_rot=int(yaw), dt=0.15)

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

        ## check https://theailearner.com/tag/cv2-minarearect/
        ## ^ The 4 corner points are ordered clockwise starting from the point with the highest y.
        rect = cv2.minAreaRect(cont)
        box = np.int0(cv2.boxPoints(rect))
        # print(f"{box=}")

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
        kp_m1, kp_m2, kp_M1, kp_M2 = keypoints

        # cv2.line(plotted_frame, tuple(kp_m1.astype(np.int32)), tuple(kp_m2.astype(np.int32)), (255, 255, 0), 1)
        cv2.circle(plotted_frame, tuple(kp_m1.astype(np.int32)), 5, (255, 0, 0), -1)
        cv2.circle(plotted_frame, tuple(kp_m2.astype(np.int32)), 5, (0,255,0), -1)

        # cv2.line(plotted_frame, tuple(kp_M1.astype(np.int32)), tuple(kp_M2.astype(np.int32)), (255, 255, 0), 1)
        cv2.circle(plotted_frame, tuple(kp_M1.astype(np.int32)), 5, (0,0,255), -1)
        cv2.circle(plotted_frame, tuple(kp_M2.astype(np.int32)), 5, (255, 255, 255), -1)
        

    def convert_pixel_to_image_plane_coordinates(self, u, v):
        x = (u - U0) / Px
        y = (v - V0) / Py

        return (x, y)
    
    def compute_interaction_matrix(self, points):
        ## TODO set a proper value for Z
        Z = 12

        J_list = []
        for p in points:
            p_array = np.array([[p[0]], [p[1]]])
            # print(p_array)

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]

            # x, y = self.convert_pixel_to_image_plane_coordinates(u, v)

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
    
    def compute_velocities_ibvs(self, points, goal_points_xy, bbox=False):
        J = self.compute_interaction_matrix(points)
        
        jcond = np.linalg.cond(J)

        if (bbox):
            self.jconds.append(jcond)
            print(f"J_cond: {jcond}")

        J_pinv = np.linalg.pinv(J)

        # points_xy = np.array([self.convert_pixel_to_image_plane_coordinates(point[0], point[1]) for point in points])
        # points_xy = np.hstack(points_xy)

        ## TODO here
        goal_new = []
        for p in points:
            p_array = np.array([[p[0]], [p[1]]])
            # print(p_array)

            xy = self.Kinv @ e2h(p_array)
            x = xy[0,0]
            y = xy[1,0]
            goal_new.append((x, y))
        
        goal_new = np.array(goal_new)
        goal_new = np.hstack(goal_new)

        # print(f"old={points_xy}")
        # print(f"new={goal_new}")

        points_xy = goal_new

        e = goal_points_xy - points_xy
        if (bbox):
            self.errs.append(np.linalg.norm(e))

        print(f"err: {e}")
        print(f"Current: {points_xy} \n Goal: {goal_points_xy}")
        print("--------------------------------------------")

        if (isinstance(self.lambda_factor, np.ndarray)):
            vel = self.lambda_factor @ J_pinv @ e
        else:
            vel = self.lambda_factor * J_pinv @ e

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

        # if (self._frame_count % 3 != 0):
        #     return


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

                ## works on NOT ORIENTED BBOX
                ## ---------------------------
                # dx = xr - xl
                # dy = yr - yl
                # r = (1. * dy) / dx
                # if (abs(r - 2.275) >= 0.207):
                #     cv2.imshow("Drone View", plotted_frame)
                #     cv2.waitKey(1)

                #     return

                xc = (xl + xr) // 2
                yc = (yl + yr) // 2
                object_center = (xc, yc)

                box_color = (0, 255, 0)
                box_thickness = 3
                # cv2.rectangle(plotted_frame, (xl, yl), (xr, yr), box_color, box_thickness)

                points_bbox = [(xl, yl), (xl, yr), (xr, yl), (xr, yr), object_center]
                # points_bbox = points_bbox[:3]
                # self.plot_bbox_keypoints(plotted_frame, tuple(points_bbox[:4]))
                # self.plot_bbox_keypoints(plotted_frame, tuple(self.goal_points_bbox[:4]))
                ## ------------------------------------------------
                
                cv2.circle(plotted_frame, object_center, 5, (255, 0, 0), -1)
                cv2.line(plotted_frame, frame_center, object_center, (0, 165, 255), 2)

                # with open("xy.txt", "a") as f:
                #     f.write(f"{object_center[0]} {object_center[1]}\n")

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

                cv2.drawContours(plotted_frame, [box], 0, (36,255,12), 3)
                ## they are ordered _clockwise_
                # self.plot_bbox_keypoints(plotted_frame, tuple(box))

                sides = get_rectangle_sides(box)
                # print(f"{sides=}")

                ## works on ORIENTED BBOX
                ## ---------------------------
                r = (1. * sides[1]) / sides[0]
                if (abs(r - 2.401) >= 0.198):
                    cv2.imshow("Drone View", plotted_frame)
                    cv2.waitKey(1)

                    return

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
                
                self.plot_ellipse_keypoints(plotted_frame, (kp_m1, kp_m2, kp_M1, kp_M2))
                # print(f"ellipse - {tuple(self.goal_points_ellipse)}")
                self.plot_ellipse_keypoints(plotted_frame, tuple(self.goal_points_ellipse))
                
                points_ellipse = [tuple(kp_m1), tuple(kp_m2), tuple(kp_M1), tuple(kp_M2)]

                ## ------------------

                ## velocity
                ## ------------------
                
                distance_from_goal = self.find_distance(frame_center, object_center)
                print(f"Object center {object_center} | Frame center {frame_center} | distance from goal: {distance_from_goal}")

                # vel = self.compute_velocities_ibvs(points_bbox, self.goal_points_bbox_xy, bbox=True)
                # print(f"Velocities bbox: {vel}")

                ## set bbox to True only if you want to plot jcond and the errors
                vel_ellipse = self.compute_velocities_ibvs(points_ellipse, self.goal_points_ellipse_xy, bbox=True)
                print(f"Velocities ellipse: {vel_ellipse}")
                vel = vel_ellipse
                # vel[2] = vel_ellipse[2]
                # vel[5] = vel_ellipse[5]

                self.velocity_to_command(vel)

                ## plot
                ## -----------------

                if (self._frame_count % 120 == 0):
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Plot X values
                    ax1.plot(range(len(self.jconds)), self.jconds, 'b-o')
                    ax1.set_title('J_cond')
                    ax1.set_ylabel('value')
                    ax1.grid(True)
                    
                    # Plot Y values
                    ax2.plot(range(len(self.errs)), self.errs, 'r-o')
                    ax2.set_title('error')
                    ax2.set_xlabel('idx')
                    ax2.set_ylabel('err norm')
                    ax2.grid(True)

                    plt.savefig("_check_errs.jpg", bbox_inches='tight')


                ## -----------------

                # vel_ellipse = self.compute_velocities_ibvs(points_ellipse, self.goal_points_ellipse_xy)
                # print(f"Velocities ellipse: {vel_ellipse}")
                # vel[2] = vel_ellipse[2]
                # vel[5] = vel_ellipse[5]

                # if (distance_from_goal <= self.offset_threshold):
                #     vel = np.zeros(6)
                
                # if (abs(vel[1]) > 200):
                #     cv2.imwrite("check_frame.png", plotted_frame)
                #     with open("check_vel.txt", "w") as f:
                #         f.write(f"{vel}\n")

                # with open("rpy.txt", "a") as f:
                #     f.write(f"{vel[0]} {vel[1]} {vel[5]}\n")

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

    print(f"STOP")
