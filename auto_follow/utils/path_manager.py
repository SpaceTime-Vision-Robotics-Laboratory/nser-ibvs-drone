from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

    MODELS_DIR: Final[Path] = BASE_DIR / "models"
    UNTRAINED_YOLO_PATH: Final[Path] = MODELS_DIR / "yolo11n.pt"
    REAL_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolon_car_detector.pt"
    SIM_CAR_YOLO_PATH: Final[Path] = MODELS_DIR / "yolov11n_car_detector_sim.pt"

    PID_CONFIG_DIR: Final[Path] = BASE_DIR / "config" / "pid"
    PID_FWD_PATH: Final[Path] = PID_CONFIG_DIR / "pid_forward.yaml"
    PID_X_PATH: Final[Path] = PID_CONFIG_DIR / "pid_x.yaml"

    OUTPUT_DIR: Final[Path] = BASE_DIR / "output"


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.SIM_CAR_YOLO_PATH)
