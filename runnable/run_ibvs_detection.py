import argparse

from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from drone_base.config.drone import DroneIp
from drone_base.stream.base_streaming_controller import BaseStreamingController


class IBVSController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_position(self):
        pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default=DroneIp.SIMULATED)
    args.add_argument("--speed", type=int, default=35)
    args.add_argument("--simulated", action="store_true")
    args = args.parse_args()


    controller = IBVSController(
        ip=args.ip,
        processor_class=IBVSYoloProcessor,
        speed=args.speed
    )

    if args.simulated:
        controller.initialize_position()

    controller.run()
