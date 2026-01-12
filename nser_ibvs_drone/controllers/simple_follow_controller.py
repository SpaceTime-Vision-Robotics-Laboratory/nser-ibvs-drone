from nser_ibvs_drone.processors.simple_yolo_processor import SimpleYoloProcessor
from drone_base.config.drone import DroneIp
from drone_base.stream.base_streaming_controller import BaseStreamingController


class SimpleFollowController(BaseStreamingController):
    def __init__(self, ip: DroneIp, processor_class=SimpleYoloProcessor, **kwargs):
        super().__init__(ip=ip, processor_class=processor_class, **kwargs)


if __name__ == '__main__':
    controller = SimpleFollowController(
        ip=DroneIp.SIMULATED,
        processor_class=SimpleYoloProcessor,
    )

    controller.run()
