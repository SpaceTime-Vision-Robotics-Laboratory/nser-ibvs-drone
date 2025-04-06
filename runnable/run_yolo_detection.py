from pathlib import Path
import sys
import time
import argparse
# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp, GimbalType
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from main.processors.yolo_processor import YOLOProcessor

class BasicController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self, takeoff_height=2.0, gimbal_angle=-45, back_distance=2.0, left_distance=0.0):
        """
        Initialize the drone position by:
        1. Taking off
        2. Moving to a specified height
        3. Moving backward to create distance from the subject
        4. Tilting the gimbal to the specified angle
        Args:
            takeoff_height: Height to ascend to after takeoff (meters)
            gimbal_angle: Gimbal tilt angle (negative for downward tilt, in degrees)
            back_distance: Distance to move backward (meters)
            left_distance: Distance to move left (meters)
        """
        # Connect to the drone if not already connected
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        # Take off
        print("Taking off...")
        self.drone_commander.take_off()
        time.sleep(5)  # Wait for stable takeoff

        # Ascend to specified height
        print(f"Ascending to {takeoff_height}m...")
        # Move up (negative Z value) by the specified amount
        self.drone_commander.move_by(forward=0, right=0, down=-takeoff_height, rotation=0)
        time.sleep(3)  # Wait for movement to complete

        # Move backward to create distance
        print(f"Moving backward {back_distance}m...")
        self.drone_commander.move_by(forward=-back_distance, right=0, down=0, rotation=0)
        time.sleep(3)  # Wait for movement to complete

        # Move left to create distance
        print(f"Moving left {left_distance}m...")
        self.drone_commander.move_by(forward=0, right=-left_distance, down=0, rotation=0)
        time.sleep(3)  # Wait for movement to complete

        # Tilt the gimbal downward
        print(f"Tilting gimbal to {gimbal_angle} degrees...")
        self.drone_commander.tilt_camera(
            pitch_deg=gimbal_angle,
            control_mode=GimbalType.MODE_POSITION,
            reference_type=GimbalType.REF_ABSOLUTE
        )
        time.sleep(2)  # Wait for gimbal to stabilize
        self.frame_processor.frame_queue.empty()
        print("Initialization complete. Ready for tracking.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default=DroneIp.SIMULATED)
    args.add_argument("--speed", type=int, default=35)
    args.add_argument("--simulated", action="store_true")
    args = args.parse_args()

    controller = BasicController(
        ip=args.ip,
        processor_class=YOLOProcessor,
        speed=args.speed
    )

    # Initialize drone position before starting tracking
    if args.simulated:
        controller.initialize_position(takeoff_height=3.25, gimbal_angle=-35, back_distance=2.5, left_distance=0.0)

    # Start tracking
    controller.run()
