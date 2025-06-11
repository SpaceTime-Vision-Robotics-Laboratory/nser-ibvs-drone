from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

from auto_follow.detection.targets import TargetIBVS
from auto_follow.detection.yolo_ibvs import YoloEngineIBVS
from auto_follow.splitter.infer import MaskSplitterInference
from auto_follow.utils.path_manager import Paths


class MaskSplitterEngineIBVS(YoloEngineIBVS):
    def __init__(
            self,
            model_path: str | Path = Paths.SIM_CAR_IBVS_YOLO_PATH,
            splitter_model_path: str | Path = Paths.SIM_MASK_SPLITTER_CAR_HIGH_PATH
    ):
        super().__init__(model_path=model_path)
        self.splitter_model = MaskSplitterInference(
            model_path=splitter_model_path,
            image_size=(360, 640),
            confidence_threshold=0.5,
            is_model_compiled=False
        )

        self.max_number_of_points = 2
        self.contour_2d_dimensions = 2
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

        pts0 = tuple(map(int, front_points_candidates[0]))
        pts1 = tuple(map(int, front_points_candidates[1]))

        points_reordered = [pts0, pts1]

        points_neighbours = [*box, box[0]]
        for i in range(len(points_neighbours) - 1):
            if ((points_neighbours[i] == pts0 and points_neighbours[i + 1] == pts1) or
                    (points_neighbours[i] == pts1 and points_neighbours[i + 1] == pts0)):
                points_reordered = box[i:] + box[:i]
                break

        points_reordered = [tuple(map(int, p)) for p in points_reordered]

        return points_reordered

    def find_best_target(self, frame: np.ndarray, results: Results | None) -> TargetIBVS:
        _, mask, masks_xy = self.segment_image(frame=frame)

        ## check that the returned masks_xy is the one containing the points
        ## and not the default_mask
        if (masks_xy.shape[1] != 2): # noqa: PLR2004
            return self._default_target

        front_mask, back_mask = self.splitter_model.infer(image=frame, mask=mask)

        best_back = {"conf": 0.8, "idx": None, "masks_xy": []}
        best_front = {"conf": 0.8, "idx": None, "masks_xy": []}

        if np.any(back_mask > 0):
            contours_back, _ = cv2.findContours(back_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_back:
                largest_contour_back = max(contours_back, key=cv2.contourArea)
                masks_xy_back = largest_contour_back.squeeze().astype(np.int32)
                if masks_xy_back.ndim == self.contour_2d_dimensions:
                    best_back["masks_xy"] = masks_xy_back
                    cv2.fillPoly(frame, pts=[masks_xy_back], color=(255, 0, 0, 8))

        if np.any(front_mask > 0):
            contours_front, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_front:
                largest_contour_front = max(contours_front, key=cv2.contourArea)
                masks_xy_front = largest_contour_front.squeeze().astype(np.int32)
                if masks_xy_front.ndim == self.contour_2d_dimensions:
                    best_front["masks_xy"] = masks_xy_front
                    cv2.fillPoly(frame, pts=[masks_xy_front], color=(0, 0, 255, 8))

        combined_masks = []
        if len(best_back["masks_xy"]) > 0:
            combined_masks.append(best_back["masks_xy"])
        if len(best_front["masks_xy"]) > 0:
            combined_masks.append(best_front["masks_xy"])

        if not combined_masks:
            return self._default_target

        all_points = np.vstack(combined_masks) if combined_masks else np.array([])

        if len(all_points) > 0:
            x1, y1 = all_points.min(axis=0)
            x2, y2 = all_points.max(axis=0)
            box = (int(x1), int(y1), int(x2), int(y2))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            size = (int(x2 - x1), int(y2 - y1))

            bbox_oriented = self._compute_bbox_oriented(frame, masks_xy)
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

    m = MaskSplitterEngineIBVS()
    print(summary(m.model))
    print("\n\n")
    print(summary(m.splitter_model.model))
