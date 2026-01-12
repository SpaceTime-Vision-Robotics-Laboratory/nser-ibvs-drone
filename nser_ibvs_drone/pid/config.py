import os
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class PIDConfig:
    """
    Configuration parameters for a PID controller.

    Attributes:
        - kp (float): Proportional gain.
            Further the drone is from the target error, the harder it will try to reach the desired value.
        - ki (float): Integral gain.
            The more time you wait to get to the target error, the harder it will try to reach the desired value.
        - kd (float): Derivative gain.
            If you get to the destination quickly, it will slow down to reach the desired value.
        - dead_zone (float): If the error reaches this value, it will output 0 (no movement).
        - target_error (float): The value we want to optimize towards.
        - thresholds (tuple[float, float]): Clips the PID output to the given speed limits.
        - max_integral (float): Anti-windup integral limit.
    """
    kp: float
    ki: float
    kd: float
    dead_zone: float
    target_error: float = 0
    thresholds: tuple[float, float] | None = None
    max_integral: float = 100

    def __post_init__(self):
        if self.thresholds is not None:
            max_length = 2
            if len(self.thresholds) != max_length:
                raise ValueError(f"Thresholds must be a tuple of (min, max) of length 2. Got {len(self.thresholds)}")
            if self.thresholds[0] > self.thresholds[1]:
                raise ValueError(f"{self.thresholds[0]=} cannot be greater than {self.thresholds[1]=}")

        if self.max_integral < 0:
            raise ValueError("Value for \"max_integral\" must be non-negative.")

        if self.dead_zone < 0:
            raise ValueError("Value for \"dead_zone\" must be non-negative.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PIDConfig":
        """Load PID configuration from a YAML file and returns a PIDConfig instance."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"YAML file not found at: {path}")

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        if "thresholds" in data and isinstance(data["thresholds"], list):
            data["thresholds"] = tuple(data["thresholds"])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save current PID configuration to a YAML file."""
        data = asdict(self)
        if self.thresholds is not None:
            data["thresholds"] = list(self.thresholds)

        with open(path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)


@dataclass
class FilterConfig:
    """
    Configuration parameters for a Low Pass Filter.

    Attributes:
        - alpha: Smoothing factor, between 0 (no smoothing) and 1 (tons of smoothing).
        - wp: Pass-band edge frequency, normalized from 0 to 1 (where 1 is π rad/sample).
        - ws: Stop-band edge frequency, normalized from 0 to 1 (where 1 is π rad/sample).
        - filter_order: Order of the FIR filter (length of the window).
        - mode: The type of filtering, can be either "fir" or "exp"
    """
    alpha: float = 0.2
    wp: float = 0.2
    ws: float = 0.3
    filter_order: int = 30
    mode: str = "fir"

    def __post_init__(self):
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"Filter alpha must \"{self.alpha}\" be between 0 and 1.")

        if self.wp < 0 or self.wp > 1:
            raise ValueError(f"Pass-band edge frequency (wp) \"{self.wp}\" must be between 0 and 1. ")

        if self.ws < 0 or self.ws > 1:
            raise ValueError(f"Stop-band edge frequency (ws) \"{self.ws}\" must be between 0 and 1. ")

        if self.filter_order < 0:
            raise ValueError(f"Filter order \"{self.filter_order}\" must be non-negative.")

        if self.mode not in ("fir", "exp"):
            raise ValueError(f"Filter mode \"{self.mode}\" must be \"fir\" or \"exp\".")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FilterConfig":
        """Load Filter configuration from a YAML file and returns a FilterConfig instance."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"YAML file not found at: {path}")

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save current Filter configuration to a YAML file."""
        data = asdict(self)

        with open(path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
