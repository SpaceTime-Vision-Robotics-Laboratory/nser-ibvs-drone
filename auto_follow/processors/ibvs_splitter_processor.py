from pathlib import Path

import numpy as np
import time

from auto_follow.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from auto_follow.utils.path_manager import Paths


class IBVSSplitterProcessor(IBVSYoloProcessor):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_LOW_PATH,
            **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.detector = MaskSplitterEngineIBVS(
            model_path=model_path,
            splitter_model_path=splitter_model_path
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        timestamp = time.perf_counter()

        parquet_row = {
            "timestamp": timestamp,
            "frame_idx": self._frame_count,
        }

        if (self._frame_count % 2 == 1):
            return frame

        target_data = self.detector.find_best_target(frame, None)

        if target_data.confidence == -1:
            return frame

        self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)

        command_info, logs = self.target_tracker.calculate_movement(target_data)

        self.logger.info("Command info: %s", command_info)
        self.logger.info(
            "Velocities IBVS: %s, Jcond: %.5f, Err norm: %.5f",
            logs["velocity"],
            logs["jcond"],
            np.linalg.norm(logs["err_uv"])
        )

        self._save_parquet_logs(parquet_row, command_info, logs)

        self.perform_movement(command_info)

        return frame
