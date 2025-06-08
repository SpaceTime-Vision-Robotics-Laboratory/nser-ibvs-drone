import argparse

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from auto_follow.controllers.ibvs_pose_controller import IBVSPoseController
from auto_follow.processors.ibvs_pose_yolo_processor import IBVSPoseYoloProcessor
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp


def main(args: argparse.Namespace):
    controller = IBVSPoseController(
        ip=args.ip,
        processor_class=IBVSPoseYoloProcessor,
        speed=args.speed,
        # log_path=Paths.OUTPUT_DIR / "logs",
        # results_path=Paths.OUTPUT_DIR / "results",
    )

    if args.simulated:
        controller.initialize_position()

    controller.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default=DroneIp.SIMULATED)
    args.add_argument("--speed", type=int, default=35)
    args.add_argument("--simulated", action="store_true")
    args = args.parse_args()

    main(args)
