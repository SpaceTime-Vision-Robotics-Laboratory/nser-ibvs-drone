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
        
    def display_segmentation(self, frame: np.ndarray, target_data: Target, moved_up: bool = False) -> None:
        """Display frame with overlays showing tracking status and targets"""
        plotted_frame = frame.copy()
        overlay = plotted_frame.copy()
        
        # --- Draw Frame Center and 85% Area Box ---
        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = self.frame_center
        
        # Draw center dot
        cv2.circle(plotted_frame, self.frame_center, 7, (0, 0, 255), -1)  # Slightly larger red dot
        cv2.circle(plotted_frame, self.frame_center, 9, (255, 255, 255), 1)  # White outline
        
        # Calculate 85% area box dimensions
        scale_factor = 0.85 ** 0.5  # sqrt(0.85) for 85% area
        box_width_85 = int(frame_width * scale_factor)
        box_height_85 = int(frame_height * scale_factor)
        x1_85 = int(frame_center_x - box_width_85 / 2)
        y1_85 = int(frame_center_y - box_height_85 / 2)
        x2_85 = int(frame_center_x + box_width_85 / 2)
        y2_85 = int(frame_center_y + box_height_85 / 2)

        # Draw the 85% box (e.g., yellow dashed line)
        cv2.rectangle(plotted_frame, (x1_85, y1_85), (x2_85, y2_85), (0, 255, 255), 2) # Yellow, thickness 2
        
        # Draw Frame Axes Lines (Horizontal and Vertical through center)
        cv2.line(plotted_frame, (0, frame_center_y), (frame_width, frame_center_y), (128, 128, 128), 1) # Gray horizontal line
        cv2.line(plotted_frame, (frame_center_x, 0), (frame_center_x, frame_height), (128, 128, 128), 1) # Gray vertical line
        # --- End Frame Center/Box Drawing ---

        if not target_data.is_lost and target_data.center is not None:
            # --- Target Found Drawing ---
            mask_coords = target_data.mask_coords
            x1, y1, x2, y2 = target_data.box
            best_center = target_data.center
            confidence = target_data.confidence

            # Draw semi-transparent fill
            if mask_coords is not None:
                mask_coords_np = np.array(mask_coords, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [mask_coords_np], (0, 255, 0))
                cv2.addWeighted(overlay, 0.5, plotted_frame, 0.5, 0, plotted_frame)

            # Draw target center marker
            cv2.circle(plotted_frame, best_center, 5, (0, 255, 0), -1)  # Solid dot
            cv2.circle(plotted_frame, best_center, 7, (255, 255, 255), 1)  # White outline

            # --- Draw Fitted Ellipse and Axes (if available) ---
            if target_data.ellipse_keypoints is not None:
                keypoints = target_data.ellipse_keypoints.astype(np.int32)
                center, major1, major2, minor1, minor2 = keypoints
                ellipse_center_int = tuple(center)
                # Ensure axes tuple contains integers for cv2.ellipse
                ellipse_axes_int = tuple(map(int, target_data.ellipse_axes))
                ellipse_angle = target_data.ellipse_angle

                # Draw the fitted ellipse (e.g., blue)
                cv2.ellipse(plotted_frame, ellipse_center_int, ellipse_axes_int, ellipse_angle, 0, 360, (255, 0, 0), 1) 

                # Draw major axis (e.g., red)
                cv2.line(plotted_frame, tuple(major1), tuple(major2), (0, 0, 255), 1)
                # Draw minor axis (e.g., cyan)
                cv2.line(plotted_frame, tuple(minor1), tuple(minor2), (255, 255, 0), 1)
            # --- End Ellipse Drawing ---

            # Confidence text
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            conf_x = x1
            conf_y = y1 - 10 if y1 > self.TEXT_PADDING_THRESHOLD else y1 + conf_h + 5
            cv2.putText(plotted_frame, conf_text, (conf_x, conf_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

            # Ellipse Angle text (if available)
            if target_data.ellipse_angle is not None:
                angle_text = f"Angle: {target_data.ellipse_angle:.1f} deg"
                (angle_w, angle_h), _ = cv2.getTextSize(angle_text, font, font_scale, font_thickness)
                # Position below confidence text
                angle_x = conf_x
                angle_y = conf_y + angle_h + 5 
                # Adjust if it goes off screen (simple check)
                # if angle_y > plotted_frame.shape[0] - 10: 
                #     angle_y = conf_y - angle_h - 5 # Try placing above confidence if below is bad

                cv2.putText(plotted_frame, angle_text, (angle_x, angle_y), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA) # Blue text

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
        cv2.imshow("Segmentation", plotted_frame)
        cv2.waitKey(1)
        
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

            # --- Draw Fitted Ellipse and Axes (if available) ---
            if target_data.ellipse_keypoints is not None:
                keypoints = target_data.ellipse_keypoints.astype(np.int32)
                center, major1, major2, minor1, minor2 = keypoints
                ellipse_center_int = tuple(center)
                # Ensure axes tuple contains integers for cv2.ellipse
                ellipse_axes_int = tuple(map(int, target_data.ellipse_axes))
                ellipse_angle = target_data.ellipse_angle

                # Draw the fitted ellipse (e.g., blue)
                cv2.ellipse(img=plotted_frame,
                            center=ellipse_center_int,
                            axes=ellipse_axes_int,
                            angle=ellipse_angle, 
                            startAngle=0, 
                            endAngle=360, 
                            color=(255, 0, 0),
                            thickness=1)

                # Draw major axis (e.g., red)
                # cv2.line(plotted_frame, tuple(major1), tuple(major2), (0, 0, 255), 1)
                # Draw minor axis (e.g., cyan)
                # cv2.line(plotted_frame, tuple(minor1), tuple(minor2), (255, 255, 0), 1)
            # --- End Ellipse Drawing ---

            # Confidence text
            conf_text = f"Conf: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            conf_x = x1
            conf_y = y1 - 10 if y1 > self.TEXT_PADDING_THRESHOLD else y1 + conf_h + 5
            cv2.putText(plotted_frame, conf_text, (conf_x, conf_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

            # Ellipse Angle text (if available)
            if target_data.ellipse_angle is not None:
                angle_text = f"Angle: {target_data.ellipse_angle:.1f} deg"
                (angle_w, angle_h), _ = cv2.getTextSize(angle_text, font, font_scale, font_thickness)
                # Position below confidence text
                angle_x = conf_x
                angle_y = conf_y + angle_h + 5 
                # Adjust if it goes off screen (simple check)
                # if angle_y > plotted_frame.shape[0] - 10: 
                #     angle_y = conf_y - angle_h - 5 # Try placing above confidence if below is bad

                cv2.putText(plotted_frame, angle_text, (angle_x, angle_y), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA) # Blue text

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