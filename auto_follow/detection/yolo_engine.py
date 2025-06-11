from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics.engine.results import Results

from auto_follow.detection.targets import Target
from auto_follow.utils.path_manager import Paths


class YoloEngine:
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_YOLO_PATH):
        self.model = ultralytics.YOLO(model_path)

        self.alpha = 0.5
        self.mask_confidence = 0.5
        self._default_target = Target(confidence=-1.0)
        self.segmentation_color = (0, 200, 0)
        self.bounding_box_color = (0, 255, 0)
        self.text_color = (36, 255, 12)

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

    def segment_image(
            self, frame: np.ndarray, are_results_returned: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Results]:
        """
        Runs segmentation on an image using YOLO and returns both the annotated image
        and the first binary mask.

        :param frame: Image to be segmented.
        :param are_results_returned: Will also return YOLO results.
        :return: Tuple of (annotated_frame, default_mask)
        """
        results = self.detect(frame)

        frame_height, frame_width = frame.shape[:2]
        default_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        boxes = results.boxes

        ## check for the box with the largest confidence to be above the threshold
        if not (
            boxes
            and boxes.conf is not None
            and len(boxes.conf) > 0
            and boxes.conf.max() >= self.confidence_threshold
        ):
            # print(f"conf: {self.confidence_threshold}")
            return (frame, default_mask, default_mask, results) \
                if are_results_returned else (frame, default_mask, default_mask)

        if results.masks is None or len(results.masks.data) == 0:
            return (frame, default_mask, default_mask, results) \
                  if are_results_returned else (frame, default_mask, default_mask)

        annotated_frame = np.array(frame)
        masks = []

        ## -----------------------------------

        # for i, mask in enumerate(results.masks.data):
        #     binary_mask = self._process_mask(mask, frame_width, frame_height)
        #     masks.append(binary_mask)

        #     overlay = np.zeros_like(frame)
        #     mask_bool = binary_mask > 0
        #     for c in range(3):
        #         overlay[:, :, c][mask_bool] = self.segmentation_color[c]
        #     annotated_frame = cv2.addWeighted(annotated_frame, self.alpha, overlay, 1 - self.alpha, 0, overlay)

        ## -----------------------------------

        ## =======================================

        best_conf_index = boxes.conf.argmax()

        ## will also return the xy point coordinates of the mask
        ## ^ for the oriented bbox computation
        masks_xy = results.masks.xy[best_conf_index]
        masks_xy = [list(xy) for xy in masks_xy]
        masks_xy = np.array(masks_xy).astype(np.int32)

        ## select only the mask corresponding to the largest confidence value
        mask = results.masks.data[best_conf_index]

        binary_mask = self._process_mask(mask, frame_width, frame_height)
        masks.append(binary_mask)

        overlay = np.zeros_like(frame)
        mask_bool = binary_mask > 0
        for c in range(3):
            overlay[:, :, c][mask_bool] = self.segmentation_color[c]
        annotated_frame = cv2.addWeighted(annotated_frame, self.alpha, overlay, 1 - self.alpha, 0, overlay)

        ## =======================================

        self._draw_boxes(annotated_frame, results)

        output = (annotated_frame, masks[0], masks_xy if masks else default_mask)
        return (*output, results) if are_results_returned else output

    def _draw_boxes(self, frame: np.ndarray, results: Results):
        """Draw bounding boxes and class labels on the frame."""
        if results.boxes is None or len(results.boxes) == 0:
            return

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = f"{results.names[int(cls_id)]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.bounding_box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2)

    def _process_mask(self, mask_tensor, width: int, height: int) -> np.ndarray:
        """Convert a YOLO mask tensor to a binary mask (uint8)."""
        mask = mask_tensor.cpu().numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > self.mask_confidence).astype(np.uint8) * 255
