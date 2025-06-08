import cv2
import numpy as np

from auto_follow.detection.targets import Target, TargetIBVS
from auto_follow.ibvs.ibvs_controller import ImageBasedVisualServo
from auto_follow.ibvs.ibvs_math_fcn import plot_bbox_keypoints
from drone_base.config.video import VideoConfig


class FrameVisualizer:
    """Handles all visualization and UI overlay for the drone video feed"""

    def __init__(self, video_config: VideoConfig):
        self.video_config = video_config
        self.frame_center = self.video_config.frame_center_x, self.video_config.frame_center_y
        self.window_name = "Drone View"

        self.TEXT_PADDING_THRESHOLD = 20

    def draw_frame(self, frame: np.ndarray, target_data: Target, moved_up: bool = False) -> tuple[np.ndarray, str]:
        """Draw frame with overlays showing tracking status and targets"""
        overlay = np.array(frame)

        cv2.circle(frame, self.frame_center, 7, (0, 0, 255), -1)  # Slightly larger red dot
        cv2.circle(frame, self.frame_center, 9, (255, 255, 255), 1)  # White outline

        if not target_data.is_lost and target_data.center is not None:
            # --- Target Found Drawing ---
            x1, y1, x2, y2 = target_data.box
            best_center = target_data.center
            confidence = target_data.confidence

            # Bounding Box Style
            box_color = (0, 255, 0)  # Green
            box_thickness = 3
            fill_alpha = 0.15  # Transparency level for fill

            # Draw semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw line from frame center to target center
            cv2.line(frame, self.frame_center, best_center, box_color, 2)

            # Draw target center marker
            cv2.circle(frame, best_center, 5, box_color, -1)  # Solid dot
            cv2.circle(frame, best_center, 7, (255, 255, 255), 1)  # White outline

            # Confidence text
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 > self.TEXT_PADDING_THRESHOLD else y1 + text_size[1] + 5

            cv2.putText(frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness,
                        cv2.LINE_AA)

            # Status text
            status_text = "TRACKING"
            text_color = (0, 255, 0)  # Green

            if moved_up:
                status_text = "MOVING DOWN"
                text_color = (255, 165, 0)  # Orange
        else:  # noqa: PLR5501
            # Target lost status
            if moved_up:
                status_text = "LOST - SEARCHING UP"
                text_color = (0, 165, 255)  # Orange
            else:
                status_text = "NO TARGET DETECTED"
                text_color = (0, 0, 255)  # Red

        # Display status text
        status_font_scale = 0.8
        status_font_thickness = 2
        status_font = cv2.FONT_HERSHEY_TRIPLEX
        (w, h), _ = cv2.getTextSize(status_text, status_font, status_font_scale, status_font_thickness)
        padding = 5

        rect_x1, rect_y1 = padding, padding
        rect_x2, rect_y2 = padding + w + padding * 2, padding + h + padding * 2
        text_x, text_y = padding * 2, padding + h + padding

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (text_x, text_y), status_font, status_font_scale, text_color,
                    status_font_thickness, cv2.LINE_AA)

        return frame, self.window_name

    def display_frame(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)


class FrameVisualizerIBVS(FrameVisualizer):
    def __init__(self, video_config: VideoConfig):
        super().__init__(video_config)

    def display_frame(
            self,
            frame: np.ndarray,
            target_data: TargetIBVS,
            ibvs_controller: ImageBasedVisualServo,
            goal_points: list[tuple[int, int]]
    ) -> None:
        xy_seg = target_data.masks_xy
        xl, yl = target_data.bbox_oriented[0]
        xr, yr = target_data.bbox_oriented[2]
        xc = (xl + xr) // 2
        yc = (yl + yr) // 2

        cv2.circle(frame, (xc, yc), 5, (255, 255, 0), -1)
        cv2.drawContours(frame, [np.array(target_data.bbox_oriented, dtype=int)], 0, (36, 255, 12), 3)

        plot_bbox_keypoints(frame, target_data.bbox_oriented)
        plot_bbox_keypoints(frame, goal_points)

        depths = ibvs_controller.compute_depths(xy_seg)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        depth_color = (255, 255, 255)  # White text

        # for i, (point, depth) in enumerate(zip(target_data.bbox_oriented, depths)):
        #     x, y = point
        #     depth_text = f"Z{i}: {depth:.2f}m"

        #     text_size, _ = cv2.getTextSize(depth_text, font, font_scale, font_thickness)
        #     text_w, text_h = text_size

        #     constant_point_above = 30

        #     text_x = x + 10
        #     text_y = y - 10 if y > constant_point_above else y + text_h + 10

        #     cv2.rectangle(frame, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
        #     cv2.putText(frame, depth_text, (text_x, text_y), font, font_scale, depth_color, font_thickness, cv2.LINE_AA)
