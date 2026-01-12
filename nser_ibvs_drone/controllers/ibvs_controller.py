import time

from nser_ibvs_drone.processors.ibvs_yolo_processor import IBVSYoloProcessor
from drone_base.config.drone import GimbalType, DroneIp
from drone_base.stream.base_streaming_controller import BaseStreamingController


class IBVSController(BaseStreamingController):
    def __init__(self, ip: DroneIp, processor_class=IBVSYoloProcessor, **kwargs):
        super().__init__(ip=ip, processor_class=processor_class, **kwargs)

    def initialize_position(self):
        if not self.drone.connection_state():
            print("Connecting to drone...")
            self.drone_commander.connect()

        self.drone_commander.take_off()
        time.sleep(3)
        self.drone_commander.tilt_camera(pitch_deg=-45, reference_type=GimbalType.REF_ABSOLUTE)
        time.sleep(2)

        self.drone_commander.move_by(forward=0, right=1, down=0, rotation=0)

        self.frame_processor.frame_queue.empty()


if __name__ == '__main__':
    ip = DroneIp.SIMULATED
    controller = IBVSController(
        ip=ip,
        processor_class=IBVSYoloProcessor,
    )

    if ip == DroneIp.SIMULATED:
        controller.initialize_position()

    controller.run()
