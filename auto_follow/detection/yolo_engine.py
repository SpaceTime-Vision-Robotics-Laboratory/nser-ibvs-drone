from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
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

@dataclass(frozen=True)
class TargetIBVS(Target):
    masks_xy: list[list[tuple[int, int]]] | None = None
    bbox_oriented: list[tuple[int, int, int, int]] | None = None

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

class YoloEngineIBVS(YoloEngine):
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH):
        super().__init__(model_path)
        self._default_target = TargetIBVS(confidence=-1.0)

    def _compute_bbox_oriented(
        self, frame: np.ndarray, xy_seg: list[tuple[int, int]]
    ) -> list[tuple[int, int, int, int]]:

        obj_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(obj_frame, pts=[xy_seg], color=255)
        contours, _ = cv2.findContours(obj_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found for the segmentation mask")

        cont = contours[0]
        rect = cv2.minAreaRect(cont)
        box = [tuple(int(x) for x in point) for point in cv2.boxPoints(rect)]
        return box

    def _reorder_bbox_oriented(self, box: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        pts = sorted(box, key=lambda x: x[1])

        pts0 = pts[0]
        pts1 = pts[1]

        points_reordered = [pts0, pts1]

        points_neighbours = [*box, box[0]]
        for i in range(len(points_neighbours) - 1):
            if ((points_neighbours[i] == pts0 and points_neighbours[i+1] == pts1) or
                (points_neighbours[i] == pts1 and points_neighbours[i+1] == pts0)):
                points_reordered = box[i:] + box[:i]
                break

        points_reordered = [tuple(map(int, p)) for p in points_reordered]

        return points_reordered

    def find_best_target(self, frame: np.ndarray, results: Results) -> TargetIBVS:
        boxes = results.boxes
        const_confidence_threshold = 0.85

        if not (
            boxes
            and boxes.conf is not None
            and len(boxes.conf) > 0
            and boxes.conf.max() >= const_confidence_threshold
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

class YoloEngineIBVSPose(YoloEngineIBVS):
    def __init__(self, model_path_pose: str | Path = Paths.SIM_CAR_POSE_IBVS_YOLO_PATH):
        super().__init__()
        self.model_pose = ultralytics.YOLO(model_path_pose)
        self._default_target = TargetIBVS(confidence=-1.0)

    def _reorder_bbox_oriented(
        self,
        box: list[tuple[int, int]],
        best_front: dict,
        best_back: dict
    ) -> list[tuple[int, int]]:
        """
        Reorder bounding box points to achieve a consistent clockwise order:
        [front-left, front-right, back-right, back-left],
        relative to the car's orientation determined by front and back mask centroids.
        Falls back to parent's ordering if masks are insufficient.
        """
        # Early validation of mask data
        if not self._validate_mask_data(best_front, best_back):
            return super()._reorder_bbox_oriented(box)

        # Process mask points and centroids
        front_mask_points = np.array(best_front["masks_xy"])
        back_mask_points = np.array(best_back["masks_xy"])
        centroid_front = np.mean(front_mask_points, axis=0)
        centroid_back = np.mean(back_mask_points, axis=0)
        tol_err = 1e-3

        # Convert box points to numpy arrays
        box_np = [np.array(p, dtype=float) for p in box]

        # Separate points into front and back candidates
        front_points, back_points = self._separate_points_by_centroids(
            box_np, centroid_front, centroid_back
        )
        if not front_points or not back_points:
            return super()._reorder_bbox_oriented(box)

        # Calculate car orientation vector
        vec_car_orientation = centroid_back - centroid_front
        if np.linalg.norm(vec_car_orientation) < tol_err:
            return super()._reorder_bbox_oriented(box)

        # Order points based on orientation
        ordered_points = self._order_points_by_orientation(
            front_points, back_points, vec_car_orientation, centroid_front, centroid_back
        )
        if not ordered_points:
            return super()._reorder_bbox_oriented(box)

        return [tuple(map(int, p)) for p in ordered_points]

    def _validate_mask_data(self, best_front: dict, best_back: dict) -> bool:
        """Validate that both front and back masks exist and are non-empty."""
        front_masks = best_front.get("masks_xy")
        back_masks = best_back.get("masks_xy")
        return (front_masks is not None and back_masks is not None and
                len(front_masks) > 0 and len(back_masks) > 0)

    def _separate_points_by_centroids(
        self, points: list[np.ndarray], centroid_front: np.ndarray, centroid_back: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Separate points into front and back based on distance to centroids."""
        front_points = []
        back_points = []
        for point in points:
            dist_to_front = np.linalg.norm(point - centroid_front)
            dist_to_back = np.linalg.norm(point - centroid_back)
            if dist_to_front < dist_to_back:
                front_points.append(point)
            else:
                back_points.append(point)
        return front_points, back_points

    def _order_points_by_orientation(
        self,
        front_points: list[np.ndarray],
        back_points: list[np.ndarray],
        vec_car_orientation: np.ndarray,
        centroid_front: np.ndarray,
        centroid_back: np.ndarray
    ) -> list[np.ndarray] | None:
        """Order points based on car orientation and cross products."""
        # Order front points
        fp1, fp2 = front_points
        cross_fp1 = self._calculate_cross_product(vec_car_orientation, fp1, centroid_front)
        front_ordered = self._order_pair_by_cross_product(fp1, fp2, cross_fp1)
        if not front_ordered:
            return None

        # Order back points
        bp1, bp2 = back_points
        cross_bp1 = self._calculate_cross_product(vec_car_orientation, bp1, centroid_back)
        back_ordered = self._order_pair_by_cross_product(bp1, bp2, cross_bp1)
        if not back_ordered:
            return None

        return [*front_ordered, *back_ordered]

    def _calculate_cross_product(
        self, vec_orientation: np.ndarray, point: np.ndarray, centroid: np.ndarray
    ) -> float:
        """Calculate cross product for point ordering."""
        return (vec_orientation[0] * (point[1] - centroid[1]) -
                vec_orientation[1] * (point[0] - centroid[0]))

    def _order_pair_by_cross_product(
        self, p1: np.ndarray, p2: np.ndarray, cross_p1: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Order a pair of points based on cross product."""
        tol_err = 1e-9

        if cross_p1 > tol_err:
            return p1, p2
        elif cross_p1 < -tol_err:
            return p2, p1
        return None

    def find_best_target(self, frame: np.ndarray, results: Results) -> TargetIBVS:
        results_pose = self.model_pose.predict(frame, stream=False, verbose=False)[0]

        best_back = {"conf": -1.0, "idx": None, "masks_xy": []}
        best_front = {"conf": -1.0, "idx": None, "masks_xy": []}

        for i, (cls, conf) in enumerate(zip(results_pose.boxes.cls, results_pose.boxes.conf)):
            cls_id = int(cls.item())
            conf_val = conf.item()

            if cls_id == 0 and conf_val > best_back["conf"]:
                best_back = {"conf": conf_val, "idx": i, "masks_xy": []}
            elif cls_id == 1 and conf_val > best_front["conf"]:
                best_front = {"conf": conf_val, "idx": i, "masks_xy": []}

        combined_masks = []
        bbox_oriented = []

        confidence_threshold = 0.5

        if best_back["conf"] > confidence_threshold and results_pose.masks is not None:
            idx = best_back["idx"]
            masks_xy = results_pose.masks.xy[idx]
            masks_xy = [list(xy) for xy in masks_xy]
            masks_xy = np.array(masks_xy).astype(np.int32)
            best_back["masks_xy"] = masks_xy
            combined_masks.append(masks_xy)

            cv2.fillPoly(frame, pts=[masks_xy], color=(255, 0, 0, 8))

        if best_front["conf"] > confidence_threshold and results_pose.masks is not None:
            idx = best_front["idx"]
            masks_xy = results_pose.masks.xy[idx]
            masks_xy = [list(xy) for xy in masks_xy]
            masks_xy = np.array(masks_xy).astype(np.int32)
            best_front["masks_xy"] = masks_xy
            combined_masks.append(masks_xy)

            cv2.fillPoly(frame, pts=[masks_xy], color=(0, 0, 255, 8))

        if not combined_masks:
            return self._default_target

        all_points = np.vstack(combined_masks) if combined_masks else np.array([])

        if len(all_points) > 0:
            x1, y1 = all_points.min(axis=0)
            x2, y2 = all_points.max(axis=0)
            box = (int(x1), int(y1), int(x2), int(y2))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            size = (int(x2 - x1), int(y2 - y1))

            bbox_oriented = self._compute_bbox_oriented(frame, all_points)
            print(f"Bbox oriented: {bbox_oriented} | Type: {type(bbox_oriented)}")
            bbox_oriented = self._reorder_bbox_oriented(bbox_oriented, best_front, best_back)

            return TargetIBVS(
                confidence=max(best_back["conf"], best_front["conf"]),
                center=center,
                size=size,
                box=box,
                is_lost=False,
                masks_xy=all_points,
                bbox_oriented=bbox_oriented
            )

        return self._default_target
