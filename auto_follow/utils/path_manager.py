from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

    MODELS_DIR: Final[Path] = BASE_DIR / "models"
    UNTRAINED_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n.pt"
    REAL_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolon_car_detector.pt"
    SIM_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolov11n_car_detector_sim.pt"
    SIM_CAR_YOLO_SEG_PATH: Final[Path] = MODELS_DIR / "yolo11n-seg_car_sim_simple.pt"
    SIM_CAR_SIMPLE_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_simple.pt"
    SIM_CAR_CARLA_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n_car_detector_sim_carla.pt"

    PID_CONFIG_DIR: Final[Path] = BASE_DIR / "config" / "pid"
    PID_FWD_PATH: Final[Path] = PID_CONFIG_DIR / "pid_forward.yaml"
    PID_X_PATH: Final[Path] = PID_CONFIG_DIR / "pid_x.yaml"
    LOW_PASS_FILTER_PATH: Final[Path] = PID_CONFIG_DIR / "low_pass_filter.yaml"

    REAL_CAR_YOLO_SEG_PATH: Final[Path] = MODELS_DIR / "seg_bunker_cars.pt"

    OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
    DETECTOR_LOG_DIR: Final[Path] = OUTPUT_DIR / "detector-logs"


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.SIM_CAR_YOLO_PATH)
