import time
from dataclasses import dataclass

import numpy as np

from math import ceil

from drone_base.config.video import VideoConfig
from auto_follow.detection.yolo_engine import TargetIBVS
from auto_follow.controllers.ibvs_controller import ImageBasedVisualServo

@dataclass(frozen=True)
class CommandInfo:
    timestamp: float
    x_cmd: int
    y_cmd: int
    z_cmd: int
    rot_cmd: int
    x_offset: float
    y_offset: float
    p_rot: float
    d_rot: float
    status: str


class TargetTracker:
    """Handles tracking logic and generates movement commands based on target position"""

    def __init__(self, video_config: VideoConfig):
        self.video_config = video_config

        # --- Control Gains ---
        self.kp_rot = 20  # Proportional gain for rotation (yaw) based on x_offset
        self.kd_rot = 5  # Derivative gain for rotation (damping)
        self.kp_alt = 0  # Proportional gain for altitude (z) based on y_offset (DISABLED)
        self.kp_fwd = -25  # Proportional gain for forward (y) based on y_offset

        # --- Thresholds ---
        self.offset_threshold = 0.1  # Deadband for x/y offset corrections

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time()

        # State tracking
        self.moved_up = False
        self.MIN_DT = 0.001

    def calculate_movement(
            self, object_center: tuple[int, int], box_size: tuple | None = None, target_lost: bool = False
    ) -> CommandInfo:
        """
        Calculate movement commands based on target position relative to frame center

        :param object_center: Coordinates of the center of the target object.
        :param box_size: The size of the target object bounding box.
        :param target_lost: Is the target seen in the image?
        :return: Instance of CommandInfo which contains details about the movement to send and drawing data.
        """
        current_time = time.time()
        # Calculate dt, handle potential first run or zero dt
        dt = current_time - self.last_command_time
        if dt <= self.MIN_DT:  # Avoid division by zero or excessively large derivatives
            dt = 0.15  # Assume a nominal dt if issue occurs
        self.last_command_time = current_time

        # Initialize commands and state variables
        x_movement, y_movement, z_movement, z_rot = 0, 0, 0, 0
        x_offset, y_offset = 0.0, 0.0
        derivative_rot_term = 0
        proportional_rot_term = 0
        status = "Lost/Hover"

        # Calculate state and commands ONLY if target is NOT lost
        if not target_lost and object_center is not None and box_size is not None:
            object_center_x, object_center_y = object_center
            x_offset = (object_center_x - self.video_config.frame_center_x) / (self.video_config.width / 2) \
                if self.video_config.width > 0 else 0
            y_offset = (object_center_y - self.video_config.frame_center_y) / (self.video_config.height / 2) \
                if self.video_config.height > 0 else 0

            # --- PD Control for Rotation ---
            if abs(x_offset) > self.offset_threshold:
                proportional_rot_term = self.kp_rot * x_offset

            # Calculate Derivative Term for Rotation
            delta_x_offset = x_offset - self.previous_x_offset
            derivative_rot_term = self.kd_rot * (delta_x_offset / dt)

            # Combine P and D terms for rotation command
            z_rot = int(proportional_rot_term + derivative_rot_term)

            # --- P Control for Altitude (Disabled) ---
            if abs(y_offset) > self.offset_threshold:
                z_movement = -int(self.kp_alt * y_offset)

            # --- P Control for Forward/Backward ---
            if abs(y_offset) > self.offset_threshold:
                y_movement = int(self.kp_fwd * y_offset)

            # Update previous offset for next calculation *after* using it
            self.previous_x_offset = x_offset
            status = "Tracking"

        # Clamp commands
        max_speed_command = 30  # Limit to 30% of max
        max_rot_command = 50  # Limit rotation rate

        x_movement = max(-max_speed_command, min(max_speed_command, x_movement))
        y_movement = max(-max_speed_command, min(max_speed_command, y_movement))
        z_movement = max(-max_speed_command, min(max_speed_command, z_movement))
        z_rot = max(-max_rot_command, min(max_rot_command, z_rot))

        return CommandInfo(
            timestamp=current_time,
            x_cmd=x_movement,
            y_cmd=y_movement,
            z_cmd=z_movement,
            rot_cmd=z_rot,
            x_offset=x_offset,
            y_offset=y_offset,
            p_rot=proportional_rot_term,
            d_rot=derivative_rot_term,
            status=status
        )

class TargetTrackerIBVS(TargetTracker):
    def __init__(self, video_config: VideoConfig, ibvs_controller: ImageBasedVisualServo):
        super().__init__(video_config)
        self.max_linear_speed = 2 # m/s
        self.max_height_linear_speed = 1 # m/s
        self.max_angular_speed = np.deg2rad(60) # rad/s
        self.ibvs_controller = ibvs_controller

    def calculate_movement(self, target_data: TargetIBVS) -> CommandInfo:

        self.ibvs_controller.set_current_points(target_data.bbox_oriented)
        velocities = self.ibvs_controller.compute_velocities(verbose=True)

        roll = ceil(100 * velocities[0] / self.max_linear_speed)
        pitch = ceil(-100 * velocities[1] / self.max_linear_speed)
        gaz = 0

        yaw = 100 * velocities[2] / self.max_angular_speed

        const_yaw_threshold = 2

        if (abs(yaw) < const_yaw_threshold):
            yaw = 0
        # else:
        #     roll = 0
        #     pitch = 0

        cmd_info = CommandInfo(
            timestamp=time.time(),
            x_cmd=roll,
            y_cmd=pitch,
            z_cmd=gaz,
            rot_cmd=ceil(yaw),
            x_offset=0,
            y_offset=0,
            p_rot=0,
            d_rot=0,
            status="IBVS"
        )

        print(f"{cmd_info}")

        return cmd_info
