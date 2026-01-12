from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

from nser_ibvs_drone.detection.targets import TargetIBVS
from nser_ibvs_drone.detection.yolo_ibvs import YoloEngineIBVS
from nser_ibvs_drone.utils.path_manager import Paths


class YoloEngineIBVSPose(YoloEngineIBVS):
    def __init__(self, model_path_pose: str | Path = Paths.SIM_CAR_POSE_IBVS_YOLO_PATH):
        super().__init__(model_path=model_path_pose)
        self._default_target = TargetIBVS(confidence=-1.0)

        self.max_number_of_points = 2
        self.tol_err_norm = 1e-3
        self.tol_err_cross_product = 1e-9

    def _reorder_bbox_oriented(
            self,
            box: list[tuple[int, ...]],
            best_front: dict,
            best_back: dict
    ) -> list[tuple[int, ...]]:
        """
        Reorder bounding box points to achieve a consistent clockwise order:
        [front-left, front-right, back-right, back-left],
        relative to the car's orientation determined by front and back mask centroids.
        Falls back to parent's ordering if masks are insufficient.
        """
        front_mask_points = np.array(best_front.get("masks_xy", []))
        back_mask_points = np.array(best_back.get("masks_xy", []))

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

        if (len(front_points_candidates) != self.max_number_of_points
                or len(back_points_candidates) != self.max_number_of_points):
            return super()._reorder_bbox_oriented(box)

        vec_car_orientation = centroid_back - centroid_front
        if np.linalg.norm(vec_car_orientation) < self.tol_err_norm:  # Centroids are too close
            return super()._reorder_bbox_oriented(box)

        # Order front points
        fp1, fp2 = front_points_candidates[0], front_points_candidates[1]
        # Cross product: vec_car_orientation X (point - relevant_centroid)
        # Positive Z means point is to the "left" of the car's axis.
        cross_fp1 = (
                vec_car_orientation[0] * (fp1[1] - centroid_front[1]) -
                vec_car_orientation[1] * (fp1[0] - centroid_front[0])
        )

        if cross_fp1 > self.tol_err_cross_product:  # fp1 is left (allowing for small floating point inaccuracies)
            p1_front_left = fp1
            p2_front_right = fp2
        elif cross_fp1 < -self.tol_err_cross_product:  # fp1 is right
            p1_front_left = fp2
            p2_front_right = fp1
        else:  # Collinear or very close to axis, fallback or use arbitrary but consistent order
            # Fallback for simplicity if points are perfectly collinear with axis in an unexpected way
            return super()._reorder_bbox_oriented(box)

        # Order back points
        bp1, bp2 = back_points_candidates[0], back_points_candidates[1]
        cross_bp1 = (
                vec_car_orientation[0] * (bp1[1] - centroid_back[1]) -
                vec_car_orientation[1] * (bp1[0] - centroid_back[0])
        )

        if cross_bp1 > self.tol_err_cross_product:  # bp1 is left
            p4_back_left = bp1
            p3_back_right = bp2
        elif cross_bp1 < -self.tol_err_cross_product:  # bp1 is right
            p4_back_left = bp2
            p3_back_right = bp1
        else:  # Collinear or very close to axis
            return super()._reorder_bbox_oriented(box)

        ordered_box_np = [p1_front_left, p2_front_right, p3_back_right, p4_back_left]
        return [tuple(map(int, p)) for p in ordered_box_np]

    def find_best_target(self, frame: np.ndarray, results: Results) -> TargetIBVS:
        results_pose = self.model.predict(frame, stream=False, verbose=False)[0]

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

        if best_back["conf"] > self.mask_confidence and results_pose.masks is not None:
            idx = best_back["idx"]
            masks_xy = results_pose.masks.xy[idx]
            masks_xy = [list(xy) for xy in masks_xy]
            masks_xy = np.array(masks_xy).astype(np.int32)
            best_back["masks_xy"] = masks_xy
            combined_masks.append(masks_xy)

            cv2.fillPoly(frame, pts=[masks_xy], color=(255, 0, 0, 8))

        if best_front["conf"] > self.mask_confidence and results_pose.masks is not None:
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


if __name__ == '__main__':
    from torchinfo import summary

    y = YoloEngineIBVSPose()
    summary(y.model)
