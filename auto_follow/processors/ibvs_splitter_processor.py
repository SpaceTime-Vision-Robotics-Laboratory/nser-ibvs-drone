from pathlib import Path

import numpy as np

from auto_follow.detection.mask_splitter_ibvs import MaskSplitterEngineIBVS
from auto_follow.processors.ibvs_yolo_processor import IBVSYoloProcessor
from auto_follow.utils.path_manager import Paths


class IBVSSplitterProcessor(IBVSYoloProcessor):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_HIGH_PATH,
            **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.detector = MaskSplitterEngineIBVS(
            model_path=model_path,
            splitter_model_path=splitter_model_path
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        target_data = self.detector.find_best_target(frame, None)

        if target_data.confidence == -1:
            return frame

        self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)

        command_info = self.target_tracker.calculate_movement(target_data)
        self.perform_movement(command_info)

        if (self._frame_count % 120 == 0):
            self.ibvs_controller.plot_values()

        return frame
