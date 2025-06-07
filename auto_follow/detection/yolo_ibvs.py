from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

from auto_follow.detection.targets import TargetIBVS
from auto_follow.detection.yolo_engine import YoloEngine
from auto_follow.utils.path_manager import Paths


class YoloEngineIBVS(YoloEngine):
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH):
        """YOLO engine for Image-Based Visual Servoing"""
        super().__init__(model_path)
        self._default_target = TargetIBVS(confidence=-1.0)

        self.confidence_threshold = 0.85

    def _compute_bbox_oriented(self, frame: np.ndarray, xy_seg: np.ndarray) -> list[tuple[int, ...]]:
        """
        Computes the oriented bounding box of a segmented object using its polygon coordinates.

        :param frame: Input image frame.
        :param xy_seg: Segmentation polygon points.
        :returns: List of 4 points (tuples) representing the oriented bounding box.
        """
        obj_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(obj_frame, pts=[xy_seg], color=255)
        contours, _ = cv2.findContours(obj_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found for the segmentation mask")

        cont = contours[0]
        rect = cv2.minAreaRect(cont)
        box = [tuple(int(x) for x in point) for point in cv2.boxPoints(rect)]
        return box

    def _reorder_bbox_oriented(self, box: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        """
        Reoreders the points of an oriented bounding box to ensure consistent ordering,
        starting from the topmost pair of points in image coordinates.

        :param box: List of 4 bounding box corner points.
        :returns: Reordered list of 4 corner points.
        """
        pts = sorted(box, key=lambda x: x[1])

        pts0 = pts[0]
        pts1 = pts[1]

        points_reordered = [pts0, pts1]

        points_neighbours = [*box, box[0]]
        for i in range(len(points_neighbours) - 1):
            if ((points_neighbours[i] == pts0 and points_neighbours[i + 1] == pts1) or
                    (points_neighbours[i] == pts1 and points_neighbours[i + 1] == pts0)):
                points_reordered = box[i:] + box[:i]
                break

        points_reordered = [tuple(map(int, p)) for p in points_reordered]

        return points_reordered

    def find_best_target(self, frame: np.ndarray, results: Results) -> TargetIBVS:
        boxes = results.boxes

        if not (
                boxes
                and boxes.conf is not None
                and len(boxes.conf) > 0
                and boxes.conf.max() >= self.confidence_threshold
        ):
            return self._default_target

        best_conf_index = boxes.conf.argmax()
        best_conf = boxes.conf[best_conf_index].item()
        coords = boxes.xyxy[best_conf_index].int().tolist()
        x1, y1, x2, y2 = coords

        masks_xy = results.masks.xy[best_conf_index]
        masks_xy = [list(xy) for xy in masks_xy]
        masks_xy = np.array(masks_xy).astype(np.int32)

        points_bbox_oriented = self._compute_bbox_oriented(frame, masks_xy)
        points_bbox_oriented = self._reorder_bbox_oriented(points_bbox_oriented)

        return TargetIBVS(
            confidence=best_conf,
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            size=(x2 - x1, y2 - y1),
            box=(x1, y1, x2, y2),
            is_lost=False,
            masks_xy=masks_xy,
            bbox_oriented=points_bbox_oriented
        )
