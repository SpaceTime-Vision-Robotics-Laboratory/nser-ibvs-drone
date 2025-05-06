from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics.engine.results import Results
import cv2

from auto_follow.utils.path_manager import Paths


@dataclass
class Target:
    confidence: float
    center: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    box: tuple[int, int, int, int] | None = None
    mask_coords: np.ndarray | None = None
    # Ellipse properties derived from mask_coords
    ellipse_center: tuple[float, float] | None = None
    ellipse_axes: tuple[float, float] | None = None # (minor_axis_length, major_axis_length)
    ellipse_angle: float | None = None # Angle in degrees
    # Ellipse keypoints: center, major_end1, major_end2, minor_end1, minor_end2
    ellipse_keypoints: np.ndarray | None = None # Shape (5, 2)
    is_lost: bool = True


class YoloEngine:
    def __init__(self, model_path: str | Path = Paths.SIM_CAR_YOLO_SEG_PATH):
        self.model = ultralytics.YOLO(model_path)
        self._default_target = Target(confidence=-1.0)

    def detect(self, image: np.ndarray) -> Results:
        return self.model.predict(image, stream=False, verbose=False)[0]
    
    def detect_stream(self, image: np.ndarray) -> Results:
        # Consume the first result from the generator
        results_generator = self.model.predict(image, stream=True, verbose=False)
        return next(results_generator) # Get the first item

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

        # --- Extract Mask Coordinates ---
        mask_coords = None
        if results.masks is not None and len(results.masks) > best_conf_index:
            mask_tensor = results.masks.xy[best_conf_index]  # Get mask polygon for the best detection
            if mask_tensor is not None:
                # Convert tensor to numpy array of shape (N, 2) for (x, y) coordinates
                # mask_tensor is likely already a numpy array
                mask_coords = mask_tensor.astype(np.int32)
        # --- End Mask Extraction ---

        # --- Fit Ellipse ---
        ellipse_center, ellipse_axes, ellipse_angle, ellipse_keypoints = None, None, None, None
        if mask_coords is not None and len(mask_coords) >= 5:
            try:
                # Reshape for fitEllipse if needed, assuming (N, 2) already
                # fitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(mask_coords)
                ellipse_center, ellipse_axes, ellipse_angle = ellipse
                # Note: ellipse_axes are (minorAxisLength, majorAxisLength)
                
                # Calculate keypoints for visualization
                center_x, center_y = ellipse_center
                minor_axis, major_axis = ellipse_axes[0] / 2, ellipse_axes[1] / 2 # Use half lengths
                angle_rad = np.radians(ellipse_angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                major_dx, major_dy = major_axis * cos_a, major_axis * sin_a
                minor_dx, minor_dy = minor_axis * -sin_a, minor_axis * cos_a # Perpendicular vector
                
                kp_center = np.array([center_x, center_y])
                kp_major1 = kp_center + np.array([major_dx, major_dy])
                kp_major2 = kp_center - np.array([major_dx, major_dy])
                kp_minor1 = kp_center + np.array([minor_dx, minor_dy])
                kp_minor2 = kp_center - np.array([minor_dx, minor_dy])
                
                ellipse_keypoints = np.array([kp_center, kp_major1, kp_major2, kp_minor1, kp_minor2], dtype=np.float32)
                
            except cv2.error as e:
                # Handle cases where fitEllipse fails (e.g., points are collinear)
                print(f"Warning: cv2.fitEllipse failed: {e}")
                ellipse_center, ellipse_axes, ellipse_angle, ellipse_keypoints = None, None, None, None
        # --- End Ellipse Fitting ---

        return Target(
            confidence=best_conf,
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            size=(x2 - x1, y2 - y1),
            box=(x1, y1, x2, y2),
            mask_coords=mask_coords,
            ellipse_center=ellipse_center,
            ellipse_axes=ellipse_axes,
            ellipse_angle=ellipse_angle,
            ellipse_keypoints=ellipse_keypoints,
            is_lost=False
        )
