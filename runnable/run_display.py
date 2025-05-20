import olympe

from drone_base.config.drone import DroneIp
from drone_base.stream.base_streaming_controller import BaseStreamingController
from drone_base.stream.display_only_processor import DisplayOnlyProcessor

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})


class DisplayOnlyStreamingController(BaseStreamingController):
    """Controller for basic display and manual control only functionalities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    controller = DisplayOnlyStreamingController(
        ip=DroneIp.SIMULATED,
        processor_class=DisplayOnlyProcessor,
        speed=35,
        log_path="./logs",
        results_path="./results"
    )
    controller.run()