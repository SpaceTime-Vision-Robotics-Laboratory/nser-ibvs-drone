from nser_ibvs_drone.controllers.state_follow_controller import StateFollowController
from nser_ibvs_drone.processors.state_processor import StateYoloProcessor
from nser_ibvs_drone.utils.path_manager import Paths
from drone_base.config.drone import DroneIp


def main():
    controller = StateFollowController(
        ip=DroneIp.SIMULATED,
        processor_class=StateYoloProcessor,
        speed=35,
        log_path=Paths.OUTPUT_DIR / "logs",
        results_path=Paths.OUTPUT_DIR / "results",
    )

    controller.run()


if __name__ == '__main__':
    main()
