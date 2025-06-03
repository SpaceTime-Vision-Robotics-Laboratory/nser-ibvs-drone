import cv2
import numpy as np

from auto_follow.detection.yolo_engine import Target
from drone_base.config.video import VideoConfig


class FrameVisualizer:
    """Handles all visualization and UI overlay for the drone video feed"""

    def __init__(self, video_config: VideoConfig):
        self.video_config = video_config
        self.frame_center = self.video_config.frame_center_x, self.video_config.frame_center_y
        self.window_name = "Drone View"

        self.TEXT_PADDING_THRESHOLD = 20

    def display_frame(self, frame: np.ndarray, target_data: Target, moved_up: bool = False) -> None:
        """Display frame with overlays showing tracking status and targets"""
        plotted_frame = frame.copy()
        overlay = plotted_frame.copy()

        cv2.circle(plotted_frame, self.frame_center, 7, (0, 0, 255), -1)  # Slightly larger red dot
        cv2.circle(plotted_frame, self.frame_center, 9, (255, 255, 255), 1)  # White outline

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
            cv2.addWeighted(overlay, fill_alpha, plotted_frame, 1 - fill_alpha, 0, plotted_frame)
            cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw line from frame center to target center
            cv2.line(plotted_frame, self.frame_center, best_center, box_color, 2)

            # Draw target center marker
            cv2.circle(plotted_frame, best_center, 5, box_color, -1)  # Solid dot
            cv2.circle(plotted_frame, best_center, 7, (255, 255, 255), 1)  # White outline

            # Confidence text
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 > self.TEXT_PADDING_THRESHOLD else y1 + text_size[1] + 5

            cv2.putText(plotted_frame, conf_text, (text_x, text_y), font, font_scale, box_color, font_thickness,
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

        cv2.rectangle(plotted_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(plotted_frame, status_text, (text_x, text_y), status_font, status_font_scale, text_color,
                    status_font_thickness, cv2.LINE_AA)

        # Display the frame
        cv2.imshow(self.window_name, plotted_frame)
        cv2.waitKey(1)
