import time
from dataclasses import dataclass

from drone_base.config.video import VideoConfig
from auto_follow.detection.yolo_engine import Target


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
    angle_error: float | None
    status: str


class TargetTracker:
    """Handles tracking logic and generates movement commands based on target position and orientation"""

    def __init__(self, video_config: VideoConfig):
        self.video_config = video_config

        # --- Position Control Gains & Thresholds ---
        self.kp_rot = 20  # Reduced from 20
        self.kd_rot = 5   # Reduced from 5
        self.kp_alt = 0  # Proportional gain for altitude (z) based on y_offset (DISABLED)

        self.kp_fwd = -50
        
        self.kp_fwd_x = -150  # Proportional gain for forward (y) based on y_offset
        self.kp_fwd_y = -150  # Proportional gain for forward (y) based on y_offset

        self.offset_threshold = 0.1  # Deadband for x/y offset corrections for P control
        self.centering_threshold = 0.08

        # --- Angle Control Gains & Thresholds ---
        self.kp_angle = 0.2
        self.angle_threshold = 30 # Degrees
        self.target_angle = 180.0

        # --- PD Controller State ---
        self.previous_x_offset = 0.0
        self.last_command_time = time.time()

        # State tracking
        self.moved_up = False
        self.MIN_DT = 0.001

    def calculate_movement(
            self,
            # Accept the full Target object
            target_data: Target 
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
        if not target_data.is_lost and target_data.center is not None:
            object_center_x, object_center_y = target_data.center
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
        max_speed_command = 70  # Limit to 30% of max
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
            status=status,
            angle_error=None
        )

    def calculate_movement_segmentation(
            self,
            # Accept the full Target object
            target_data: Target 
    ) -> CommandInfo:
        """
        Calculate movement commands based on target position and orientation relative to frame center.

        :param target_data: Target object containing detection and ellipse info.
        :return: Instance of CommandInfo with movement commands and state.
        """
        current_time = time.time()
        dt = current_time - self.last_command_time
        if dt <= self.MIN_DT:
            dt = 0.15
        self.last_command_time = current_time

        # Initialize commands and state variables
        x_movement, y_movement, z_movement = 0, 0, 0
        x_offset, y_offset = 0.0, 0.0
        derivative_rot_term = 0.0
        proportional_rot_term = 0.0
        z_rot = 0 # Final rotation command
        angle_error = None
        status = "Lost/Hover"

        # Calculate state and commands ONLY if target is NOT lost
        if not target_data.is_lost and target_data.center is not None:
            status = "Tracking"
            object_center_x, object_center_y = target_data.center
            x_offset = (object_center_x - self.video_config.frame_center_x) / (self.video_config.width / 2) \
                if self.video_config.width > 0 else 0
            y_offset = (object_center_y - self.video_config.frame_center_y) / (self.video_config.height / 2) \
                if self.video_config.height > 0 else 0

            # --- PD Control for Rotation (Position-based) ---
            proportional_rot_term = 0.0
            if abs(x_offset) > self.offset_threshold:
                proportional_rot_term = self.kp_rot * x_offset
            
            delta_x_offset = x_offset - self.previous_x_offset
            derivative_rot_term = self.kd_rot * (delta_x_offset / dt)
            z_rot_pos = proportional_rot_term + derivative_rot_term # Position-based rotation command
            z_rot = int(z_rot_pos) # Default to position-based rotation

            # --- P Control for Altitude (Disabled) ---
            if abs(y_offset) > self.offset_threshold:
                z_movement = -int(self.kp_alt * y_offset)

            # --- P Control for Forward/Backward ---
            if abs(y_offset) > self.offset_threshold:
                y_movement = int(-200 * y_offset)
            
            # --- P Control for Sideways (Strafe) ---
            # Consider if this is desired alongside rotation. Maybe only if rotation is small?
            # Temporarily disable sideways strafing based on x_offset, let rotation handle it.
            if abs(x_offset) > self.offset_threshold:
                x_movement = int(-self.kp_fwd_x * x_offset) # Using kp_fwd might be too aggressive?

            # --- Angle-based Rotation Override --- 
            if target_data.ellipse_angle is not None and target_data.ellipse_axes is not None:
                # Calculate angle error (-180 to 180)
                angle_error = (self.target_angle - target_data.ellipse_angle + 180) % 360 - 180
                print(f"angle_error: {angle_error}")

                # Check if target is centered and angle error is significant (Removed elongation check)
                if abs(x_offset) < self.centering_threshold and abs(angle_error) > self.angle_threshold:
                    # Override rotation command with angle correction
                    z_rot_angle = self.kp_angle * angle_error
                    z_rot = int(z_rot_angle) # Apply angle-based rotation
                    status = "Aligning Angle" # Update status

            # Update previous offset for next calculation
            self.previous_x_offset = x_offset
        else:
            # Target lost, reset previous offset
            self.previous_x_offset = 0.0

        # Clamp commands
        max_speed_command = 70
        max_rot_command = 70

        x_movement = max(-max_speed_command, min(max_speed_command, x_movement))
        y_movement = max(-max_speed_command, min(max_speed_command, y_movement))
        z_movement = max(-max_speed_command, min(max_speed_command, z_movement))
        # Clamp final rotation command
        z_rot = max(-max_rot_command, min(max_rot_command, z_rot))

        return CommandInfo(
            timestamp=current_time,
            x_cmd=x_movement,
            y_cmd=y_movement,
            z_cmd=z_movement,
            rot_cmd=z_rot, # Use the potentially overridden rotation command
            x_offset=x_offset,
            y_offset=y_offset,
            p_rot=proportional_rot_term, # Position-based P term
            d_rot=derivative_rot_term, # Position-based D term
            angle_error=angle_error, # Store calculated angle error
            status=status
        )
