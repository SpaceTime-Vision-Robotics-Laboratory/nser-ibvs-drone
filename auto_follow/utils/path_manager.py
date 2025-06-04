from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

    MODELS_DIR: Final[Path] = BASE_DIR / "models"
    UNTRAINED_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n.pt"
    REAL_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolon_car_detector.pt"
    SIM_CAR_POSE_IBVS_YOLO_PATH: Final[Path] = MODELS_DIR / "30_05_best__yolo11n-seg_sim_car_bunker__front_back.pt"
    SIM_CAR_IBVS_YOLO_PATH: Final[Path] = MODELS_DIR / "29_05_best__yolo11n-seg_sim_car_bunker__all.pt"
    SIM_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolov11n_car_detector_sim.pt"
    SIM_CAR_SIMPLE_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_simple.pt"
    SIM_CAR_CARLA_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_carla.pt"

    PID_CONFIG_DIR: Final[Path] = BASE_DIR / "config" / "pid"
    PID_FWD_PATH: Final[Path] = PID_CONFIG_DIR / "pid_forward.yaml"
    PID_X_PATH: Final[Path] = PID_CONFIG_DIR / "pid_x.yaml"
    LOW_PASS_FILTER_PATH: Final[Path] = PID_CONFIG_DIR / "low_pass_filter.yaml"

    OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
    DETECTOR_LOG_DIR: Final[Path] = OUTPUT_DIR / "detector-logs"

    CAMERA_PARAMS_PATH: Final[Path] = (
        BASE_DIR / "assets" / "camera_parameters" / "intrinsic_matrix.pkl"
    )
    CAMERA_PARAMS_HALF_SIZE_PATH: Final[Path] = (
        BASE_DIR / "assets" / "camera_parameters" / "intrinsic_matrix_half_size.pkl"
    )
    
    GOAL_FRAME_PATH_90: Final[Path] = BASE_DIR / "assets" / "reference" / "images" / "frame_001009_10636251.png"
    GOAL_FRAME_POINTS_PATH_90: Final[Path] = BASE_DIR / "assets" / "reference" / "data" / "frame_001009_10636251.json"

    GOAL_FRAME_PATH_45: Final[Path] = BASE_DIR / "assets" / "reference" / "images" / "frame_002041_12392083__45.png"
    GOAL_FRAME_POINTS_PATH_45: Final[Path] = BASE_DIR / "assets" / "reference" / "data" / "frame_002041_12392083__45.json"


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.SIM_CAR_YOLO_PATH)
