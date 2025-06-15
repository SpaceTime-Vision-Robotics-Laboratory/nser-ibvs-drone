from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

    # NN Models:
    MODELS_DIR: Final[Path] = BASE_DIR / "models"
    UNTRAINED_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n.pt"
    REAL_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolon_car_detector.pt"
    SIM_CAR_POSE_IBVS_YOLO_PATH: Final[Path] = MODELS_DIR / "30_05_best__yolo11n-seg_sim_car_bunker__front_back.pt"
    SIM_CAR_IBVS_YOLO_PATH: Final[Path] = MODELS_DIR / "29_05_best__yolo11n-seg_sim_car_bunker__all.pt"
    SIM_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolov11n_car_detector_sim.pt"
    SIM_CAR_SIMPLE_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_simple.pt"
    SIM_CAR_CARLA_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_carla.pt"
    SIM_MASK_SPLITTER_CAR_HIGH_PATH: Final[
        Path] = MODELS_DIR / "mask_splitter-sim-high-quality-partition-v10-dropout_0-augmentations_multi_scenes.pt"

    SIM_MASK_SPLITTER_CAR_LOW_PATH: Final[
        Path] = MODELS_DIR / "mask_splitter-epoch_10-dropout_0-low_x2-and-high_x0_quality_early_stop.pt"
    SIM_STUDENT_NET_TEMPORAL_PATH: Final[Path] = MODELS_DIR / "sim_temporal_student_best_model.pth"
    SIM_STUDENT_NET_PATH: Final[Path] = MODELS_DIR / "student_best_model.pth"
    SIM_STUDENT_NEW_PATH_REAL_WORLD_DISTRIBUTION: Final[Path] = MODELS_DIR / "student_model_sim_on_real_world_distribution.pth"
    
    REAL_STUNDENT_NET_PRETRAINED_REAL_WORLD_DISTRIBUTION: Final[Path] = MODELS_DIR / "student_real_pretrained_on_real_distribution.pth"

    # PID config:
    PID_CONFIG_DIR: Final[Path] = BASE_DIR / "config" / "pid"
    PID_FWD_PATH: Final[Path] = PID_CONFIG_DIR / "pid_forward.yaml"
    PID_X_PATH: Final[Path] = PID_CONFIG_DIR / "pid_x.yaml"
    LOW_PASS_FILTER_PATH: Final[Path] = PID_CONFIG_DIR / "low_pass_filter.yaml"

    # Simulator Config
    SIMULATOR_CONFIG_DIR: Final[Path] = BASE_DIR / "config" / "simulator"
    BUNKER_ANAFI_4K_CONFIG_PATH: Final[Path] = SIMULATOR_CONFIG_DIR / "bunker-online-4k-config.yaml"


    # Results / Output:
    OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
    DETECTOR_LOG_DIR: Final[Path] = OUTPUT_DIR / "detector-logs"
    LOG_PARQUET_DIR: Final[Path] = OUTPUT_DIR / "parquet-logs"

    # Camera parameters:
    CAMERA_PARAMS_DIR: Final[Path] = BASE_DIR / "assets" / "camera_parameters"
    CAMERA_SIM_ANAFI_4k_DIR: Final[Path] = CAMERA_PARAMS_DIR / "sim-anafi-4k"
    CAMERA_SIM_ANAFI_AI_DIR: Final[Path] = CAMERA_PARAMS_DIR / "sim-anafi-ai"
    CAMERA_REAL_FULL_ANAFI_4K_DIR: Final[Path] = CAMERA_PARAMS_DIR / "real-full-res"

    # Reference images:
    REFERENCE_DATA_DIR: Final[Path] = BASE_DIR / "assets" / "reference" / "data"
    REFERENCE_IMAGES_DIR: Final[Path] = BASE_DIR / "assets" / "reference" / "images"
    GOAL_FRAME_PATH_90: Final[Path] = REFERENCE_IMAGES_DIR / "frame_001009_10636251.png"
    GOAL_FRAME_POINTS_PATH_90: Final[Path] = REFERENCE_DATA_DIR / "frame_001009_10636251.json"

    GOAL_FRAME_PATH_45: Final[Path] = REFERENCE_IMAGES_DIR / "frame_001030_26999571__45__takeoff.png"
    GOAL_FRAME_POINTS_PATH_45: Final[Path] = REFERENCE_DATA_DIR / "frame_001030_26999571__45__takeoff.json"


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.SIM_CAR_YOLO_PATH)
