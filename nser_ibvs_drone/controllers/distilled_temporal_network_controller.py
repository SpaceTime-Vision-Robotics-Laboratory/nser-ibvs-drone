import argparse
from functools import partial
from pathlib import Path

from nser_ibvs_drone.controllers.eval_streaming_controller import EvaluationStreamingController
from nser_ibvs_drone.processors.distilled_temporal_network_processor import DistilledTemporalNetworkProcessor
from nser_ibvs_drone.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.config.video import VideoConfig
from drone_base.utils.readable_time import date_time_now_to_file_name


class DistilledTemporalStudentController(EvaluationStreamingController):
    def __init__(self, ip: DroneIp, parquet_log_path: str | Path | None = None, **kwargs):
        if parquet_log_path is None:
            parquet_log_path = Paths.LOG_PARQUET_DIR
        self.start_time = date_time_now_to_file_name()
        self.parquet_log_path = parquet_log_path / self.start_time
        _processor = partial(DistilledTemporalNetworkProcessor, logs_parquet_path=self.parquet_log_path)
        super().__init__(ip=ip, processor_class=_processor, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distilled Student Controller (No Splitter Validation)')
    parser.add_argument("--log_path", type=str, default=Paths.OUTPUT_DIR / "logs")
    parser.add_argument("--results_path", type=str, default=Paths.OUTPUT_DIR / "results")
    parser.add_argument("--parquet_logs_path", type=str, default=Paths.OUTPUT_DIR / "parquet-logs")
    args = parser.parse_args()
    ip = DroneIp.SIMULATED
    controller = DistilledTemporalStudentController(
        ip=ip,
        log_path=Path(args.log_path),
        results_path=Path(args.results_path),
        parquet_log_path=Path(args.parquet_logs_path),
        video_config=VideoConfig(width=640, height=360, cam_mode="recording", save_extension="jpg"),
    )

    controller.run()
