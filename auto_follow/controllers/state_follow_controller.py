from auto_follow.processors.state_processor import StateYoloProcessor
from drone_base.config.drone import DroneIp
from drone_base.stream.base_streaming_controller import BaseStreamingController


class StateFollowController(BaseStreamingController):
    def __init__(self, ip: DroneIp, processor_class=StateYoloProcessor, **kwargs):
        super().__init__(ip=ip, processor_class=processor_class, **kwargs)


if __name__ == '__main__':
    controller = StateFollowController(
        ip=DroneIp.SIMULATED,
        processor_class=StateYoloProcessor,
    )

    controller.run()
