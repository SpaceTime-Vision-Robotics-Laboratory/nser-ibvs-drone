import argparse
from functools import partial
from pathlib import Path

from auto_follow.controllers.eval_streaming_controller import EvaluationStreamingController
from auto_follow.processors.ibvs_splitter_processor import IBVSSplitterProcessor
from auto_follow.utils.path_manager import Paths
from drone_base.config.drone import DroneIp
from drone_base.config.video import VideoConfig
from drone_base.utils.readable_time import date_time_now_to_file_name


class IBVSSplitterController(EvaluationStreamingController):
    def __init__(
            self,
            ip: DroneIp,
            parquet_log_path: str | Path | None = None,
            seg_model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            **kwargs
    ):
        if parquet_log_path is None:
            parquet_log_path = Paths.LOG_PARQUET_DIR
        self.start_time = date_time_now_to_file_name()
        self.parquet_log_path = parquet_log_path / self.start_time
        _processor = partial(
            IBVSSplitterProcessor,
            model_path=seg_model_path,
            splitter_model_path=splitter_model_path,
            logs_parquet_path=self.parquet_log_path
        )
        super().__init__(ip=ip, processor_class=_processor, **kwargs)


def main_ibvs_splitter_controller():
    parser = argparse.ArgumentParser(description='IBVS Splitter Controller')
    parser.add_argument("--log_path", type=str, default=Paths.OUTPUT_DIR / "logs")
    parser.add_argument("--results_path", type=str, default=Paths.OUTPUT_DIR / "results")
    parser.add_argument("--parquet_logs_path", type=str, default=Paths.OUTPUT_DIR / "parquet-logs")
    parser.add_argument("--seg_model", type=str, default=Paths.SIM_CAR_IBVS_YOLO_PATH)
    parser.add_argument("--splitter", type=str, default=Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH)
    parser.add_argument("--is_real_world", action="store_true", help="Set to run on physical drone")
    parser.add_argument("--experiment_name", type=str, default="", help="Subdir for real world runs")
    args = parser.parse_args()

    if args.is_real_world:
        ip = DroneIp.WIRELESS
        seg_model = Paths.REAL_CAR_IBVS_YOLO_PATH
        splitter_model = Paths.REAL_MASK_SPLITTER_CAR
        experiment_name = args.experiment_name if args.experiment_name else "real-world"
        base_dir = Paths.OUTPUT_DIR / experiment_name
        log_path = base_dir / "logs"
        results_path = base_dir / "results"
        parquet_log_path = base_dir / "parquet-logs"
    else:
        ip = DroneIp.SIMULATED
        log_path = args.log_path
        results_path = args.results_path
        parquet_log_path = args.parquet_logs_path
        seg_model = args.seg_model
        splitter_model = args.splitter

    controller = IBVSSplitterController(
        ip=ip,
        log_path=Path(log_path),
        results_path=Path(results_path),
        parquet_log_path=Path(parquet_log_path),
        seg_model_path=Path(seg_model),
        splitter_model_path=Path(splitter_model),
        video_config=VideoConfig(width=640, height=360, cam_mode="recording", save_extension="jpg"),
    )

    controller.run()


if __name__ == '__main__':
    """
    Experiment names:
        real-ibvs-down-left
        real-ibvs-down-right
        real-ibvs-front-small-offset-right
        real-ibvs-front-small-offset-left
        real-ibvs-left
        real-ibvs-right
        real-ibvs-up-left
        real-ibvs-up-right
    """
    main_ibvs_splitter_controller()
