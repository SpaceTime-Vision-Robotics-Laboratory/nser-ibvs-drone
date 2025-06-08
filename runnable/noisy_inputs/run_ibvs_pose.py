import argparse

import time

from auto_follow.processors.ibvs_yolo_processor import IBVSPoseYoloProcessor
from drone_base.config.drone import DroneIp
from drone_base.config.drone import GimbalType
from drone_base.stream.base_streaming_controller import BaseStreamingController

class IBVSController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self, forward_distance=1.0, right_distance=0.0, down_distance=0.5):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-45, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)

        self.drone_commander.move_by(forward=forward_distance, right=right_distance, down=down_distance, rotation=0)

        self.frame_processor.frame_queue.empty()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default=DroneIp.SIMULATED)
    args.add_argument("--speed", type=int, default=35)
    args.add_argument("--simulated", action="store_true")
    args = args.parse_args()


    controller = IBVSController(
        ip=args.ip,
        processor_class=IBVSPoseYoloProcessor,
        speed=args.speed
    )

    controller.frame_processor.noisy_inputs = True

    if args.simulated:
        controller.initialize_position(forward_distance=1.0, right_distance=0.0, down_distance=0.5)

    controller.run()
