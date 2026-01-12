from time import sleep, time

from drone_base.control.operations import TiltCommand
from drone_base.stream.base_streaming_controller import BaseStreamingController


class EvaluationStreamingController(BaseStreamingController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.will_stop_after_mission = True
        self.is_random_start = False

    def start(self):
        """Starts all components of the streaming controller."""
        if not self.drone_commander.connect():
            raise RuntimeError("Failed to connect to the drone...")
        self.streaming_manager.start_streaming()
        self.manual_controller.start()
        self.frame_processor.start()

        if self.will_stop_after_mission:
            self.drone_commander.take_off()
            sleep(3)
            # self.drone_commander.move_by(forward=0, right=0, down=0, rotation=0)

            self.drone_commander.execute_command(TiltCommand(pitch_deg=-45), is_blocking=False)
            if self.started_mission_time is None:
                self.started_mission_time = time()
