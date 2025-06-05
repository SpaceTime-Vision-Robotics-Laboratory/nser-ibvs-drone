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

    def _reorder_bbox_oriented(self, box: list[tuple[int, int]], best_front: dict, best_back: dict) -> list[tuple[int, int]]:
        """
        Reorder bounding box points to achieve a consistent clockwise order:
        [front-left, front-right, back-right, back-left],
        relative to the car's orientation determined by front and back mask centroids.
        Falls back to parent's ordering if masks are insufficient.
        """
        front_masks_xy_raw = best_front.get("masks_xy")
        back_masks_xy_raw = best_back.get("masks_xy")

        if front_masks_xy_raw is None or back_masks_xy_raw is None:
            return super()._reorder_bbox_oriented(box)

        front_mask_points = np.array(front_masks_xy_raw)
        back_mask_points = np.array(back_masks_xy_raw)

        if front_mask_points.size == 0 or back_mask_points.size == 0:
            return super()._reorder_bbox_oriented(box)

        centroid_front = np.mean(front_mask_points, axis=0)
        centroid_back = np.mean(back_mask_points, axis=0)

        box_np = [np.array(p, dtype=float) for p in box]

        front_points_candidates = []
        back_points_candidates = []
        for point_np in box_np:
            dist_to_front = np.linalg.norm(point_np - centroid_front)
            dist_to_back = np.linalg.norm(point_np - centroid_back)
            if dist_to_front < dist_to_back:
                front_points_candidates.append(point_np)
            else:
                back_points_candidates.append(point_np)

        if len(front_points_candidates) != 2 or len(back_points_candidates) != 2:
            return super()._reorder_bbox_oriented(box)

        vec_car_orientation = centroid_back - centroid_front
        if np.linalg.norm(vec_car_orientation) < 1e-3: # Centroids are too close
             return super()._reorder_bbox_oriented(box)

        # Order front points
        fp1, fp2 = front_points_candidates[0], front_points_candidates[1]
        # Cross product: vec_car_orientation X (point - relevant_centroid)
        # Positive Z means point is to the "left" of the car's axis.
        cross_fp1 = vec_car_orientation[0] * (fp1[1] - centroid_front[1]) - vec_car_orientation[1] * (fp1[0] - centroid_front[0])

        if cross_fp1 > 1e-9: # fp1 is left (allowing for small floating point inaccuracies)
            p1_front_left = fp1
            p2_front_right = fp2
        elif cross_fp1 < -1e-9: # fp1 is right
            p1_front_left = fp2
            p2_front_right = fp1
        else: # Collinear or very close to axis, fallback or use arbitrary but consistent order
            # Fallback for simplicity if points are perfectly collinear with axis in an unexpected way
            return super()._reorder_bbox_oriented(box)


        # Order back points
        bp1, bp2 = back_points_candidates[0], back_points_candidates[1]
        cross_bp1 = vec_car_orientation[0] * (bp1[1] - centroid_back[1]) - vec_car_orientation[1] * (bp1[0] - centroid_back[0])

        if cross_bp1 > 1e-9: # bp1 is left
            p4_back_left = bp1
            p3_back_right = bp2
        elif cross_bp1 < -1e-9: # bp1 is right
            p4_back_left = bp2
            p3_back_right = bp1
        else: # Collinear or very close to axis
            return super()._reorder_bbox_oriented(box)

        ordered_box_np = [p1_front_left, p2_front_right, p3_back_right, p4_back_left]
        return [tuple(map(int, p)) for p in ordered_box_np]

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
