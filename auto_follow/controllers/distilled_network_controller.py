import argparse
from functools import partial
from pathlib import Path

from auto_follow.controllers.eval_streaming_controller import EvaluationStreamingController
from auto_follow.processors.distilled_network_processor import DistilledNetworkProcessor
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.config.video import VideoConfig
from drone_base.utils.readable_time import date_time_now_to_file_name


class DistilledStudentController(EvaluationStreamingController):
    def __init__(self, ip: DroneIp, parquet_log_path: str | Path | None = None, **kwargs):
        if parquet_log_path is None:
            parquet_log_path = Paths.LOG_PARQUET_DIR
        self.start_time = date_time_now_to_file_name()
        self.parquet_log_path = parquet_log_path / self.start_time
        _processor = partial(DistilledNetworkProcessor, logs_parquet_path=self.parquet_log_path)
        super().__init__(ip=ip, processor_class=_processor, **kwargs)


if __name__ == '__main__':
    # experiment_name = "real-student-down-left"
    # experiment_name = "real-student-down-right"
    # experiment_name = "real-student-front-small-offset-right"
    # experiment_name = "real-student-front-small-offset-left"
    # experiment_name = "real-student-left"
    # experiment_name = "real-student-right"
    # experiment_name = "real-student-up-left"
    # experiment_name = "real-student-up-right"

    # experiment_name = "real-student-random-spawn-test"
    

    parser = argparse.ArgumentParser(description='Distilled Student Controller (No Splitter Validation)')
    parser.add_argument("--log_path", type=str, default=Paths.OUTPUT_DIR / experiment_name / "logs")
    parser.add_argument("--results_path", type=str, default=Paths.OUTPUT_DIR / experiment_name / "results")
    parser.add_argument("--parquet_logs_path", type=str, default=Paths.OUTPUT_DIR / experiment_name / "parquet-logs")
    args = parser.parse_args()
    ip = DroneIp.WIRELESS
    controller = DistilledStudentController(
        ip=ip,
        log_path=Path(args.log_path),
        results_path=Path(args.results_path),
        parquet_log_path=Path(args.parquet_logs_path),
        video_config=VideoConfig(width=640, height=360, cam_mode="recording", save_extension="jpg"),
    )

    controller.run()
