from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics.engine.results import Results

from auto_follow.utils.path_manager import Paths


@dataclass(frozen=True)
class Target:
    confidence: float
    center: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    box: tuple[int, int, int, int] | None = None
    is_lost: bool = True


class YoloEngine:
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_YOLO_PATH):
        self.model = ultralytics.YOLO(model_path)
        self._default_target = Target(confidence=-1.0)

    def detect(self, image: np.ndarray) -> Results:
        return self.model.predict(image, stream=False, verbose=False)[0]

    def find_best_target(self, results: Results) -> Target:
        """
        Extract the most confident detection from the YOLO results.

        :param results: YOLO prediction.
        :return: Target dataclass with detection info.
        """
        boxes = results.boxes
        if not (boxes and boxes.conf is not None and len(boxes.conf) > 0):
            return self._default_target

        best_conf_index = boxes.conf.argmax()
        best_conf = boxes.conf[best_conf_index].item()
        coords = boxes.xyxy[best_conf_index].int().tolist()
        x1, y1, x2, y2 = coords

        return Target(
            confidence=best_conf,
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            size=(x2 - x1, y2 - y1),
            box=(x1, y1, x2, y2),
            is_lost=False
        )
