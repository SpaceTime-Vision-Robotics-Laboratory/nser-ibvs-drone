from nser_ibvs_drone.controllers.simple_follow_controller import SimpleFollowController
from nser_ibvs_drone.processors.simple_yolo_processor import SimpleYoloProcessor
from nser_ibvs_drone.utils.path_manager import Paths
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
