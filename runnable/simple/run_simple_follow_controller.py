from auto_follow.controllers.simple_follow_controller import SimpleFollowController
from auto_follow.processors.simple_yolo_processor import SimpleYoloProcessor
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp


def main():
    controller = SimpleFollowController(
        ip=DroneIp.SIMULATED,
        processor_class=SimpleYoloProcessor,
        speed=35,
        log_path=Paths.OUTPUT_DIR / "logs",
        results_path=Paths.OUTPUT_DIR / "results",
    )

    controller.run()


if __name__ == '__main__':
    main()
