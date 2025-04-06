from pathlib import Path
import sys

# Modify this line to also make drone_base/main visible as a top-level module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "drone_base"))

from drone_base.main.config.drone import DroneIp
from drone_base.main.stream.base_streaming_controller import BaseStreamingController

from main.processors.yolo_processor import YOLOProcessor

class BasicController(BaseStreamingController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__ == "__main__":
    controller = BasicController(
        ip=DroneIp.SIMULATED,
        processor_class=YOLOProcessor,
        speed=35
    )
    controller.run()
