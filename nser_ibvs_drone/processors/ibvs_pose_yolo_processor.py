from nser_ibvs_drone.detection.yolo_pose_ibvs import YoloEngineIBVSPose
from nser_ibvs_drone.processors.ibvs_yolo_processor import IBVSYoloProcessor
from nser_ibvs_drone.utils.path_manager import Paths


class IBVSPoseYoloProcessor(IBVSYoloProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = YoloEngineIBVSPose(model_path_pose=Paths.SIM_CAR_POSE_IBVS_YOLO_PATH)

    # def _process_frame(self, frame: np.ndarray) -> np.ndarray:
    #     results = self.detector.detect(frame)
    #     target_data = self.detector.find_best_target(frame, results)
    #
    #     if target_data.confidence == -1:
    #         return frame
    #
    #     self.visualizer.display_frame(frame, target_data, self.ibvs_controller, self.ibvs_controller.goal_points)
    #
    #     command_info = self.target_tracker.calculate_movement(target_data)
    #     self.perform_movement(command_info)
    #
    #     return frame
