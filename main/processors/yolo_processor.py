import numpy as np
import ultralytics
import threading
import queue
import time
import cv2

from drone_base.main.stream.base_video_processor import BaseVideoProcessor

class YOLOProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str | None = "models/yolo11n.pt", **kwargs):
        self.detector = ultralytics.YOLO(model_path)
        super().__init__(**kwargs)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Get the first result from the generator
        preds = next(iter(self.detector.predict(frame, stream=True)))
        return [frame, preds]

    def _display_frame(self, frame: list[np.ndarray]) -> None:
        # Get the original frame and predictions
        original_frame = frame[0].copy()
        results = frame[1]

        # Plot the predictions on the frame
        plotted_frame = results.plot()

        # Add center point to the frame
        frame_height, frame_width = plotted_frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        cv2.circle(plotted_frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)  # Green dot
        cv2.circle(original_frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)  # Green dot on original too

        # Add center points to each bounding box
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                cv2.circle(plotted_frame, (box_center_x, box_center_y), 5, (255, 0, 0), -1)  # Blue dot

        # Display frames
        cv2.imshow("YOLO Detection", plotted_frame)
        cv2.imshow("Live Feed", original_frame)
        cv2.waitKey(1)

    def _run_processing_loop(self):
        self.logger.info("Starting frame processing loop.")

        with self._frame_display_context():
            while self._running.is_set() and threading.main_thread().is_alive():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    with self._lock:
                        self._frame_count += 1
                        if self.is_frame_saved:
                            self.frame_saver.add_frame(frame=np.array(frame), timestamp=time.perf_counter())
                        self._display_frame(self._process_frame(frame))
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error("Unable to process frame...")
                    self.logger.critical(e, exc_info=True)
                    if not self._running.is_set():
                        break

        self.logger.info("Frame processing loop terminated.")
