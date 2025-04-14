import time

from auto_follow.pid.config import PIDConfig
from auto_follow.pid.filters import LowPassFilter


class PIDController:
    """
    PID controller with adaptive gains and anti-windup protection.

    This controller implements a standard PID algorithm with:
        - Low-pass filtering to reduce noise
        - Adaptive proportional and integral gains
        - Output limiting with anti-windup protection
        - Dead zone to prevent oscillation near the target error
    """

    def __init__(self, config: PIDConfig):
        self.config = config

        self.dead_zone_adaptive_threshold = self.config.dead_zone * 2 + (self.config.dead_zone * 0.2)

        self.last_error = 0.0
        self.integral = 0.0
        self.low_pass_filter = LowPassFilter(alpha=self.config.filter_alpha)
        self.prev_time = time.perf_counter()

    def compute(self, process_error: float) -> float:
        """
        Compute a new control output based on the value.

        :param process_error: Current process error measurement.
        :return: The control output value.
        """
        if abs(process_error) < self.config.dead_zone:
            self.reset()
            return 0.0

        current_time = time.perf_counter()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        filtered_error = self.low_pass_filter.filter(self.config.target_error - process_error)

        if self.config.thresholds is None or self.config.thresholds[0] < filtered_error < self.config.thresholds[1]:
            self.integral += filtered_error * dt
        self.integral = max(-self.config.max_integral, min(self.integral, self.config.max_integral))

        derivative = (filtered_error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = filtered_error

        kp = self._adapt_kp(process_error)
        ki = self._adapt_ki(process_error)

        output = kp * filtered_error + ki * self.integral + self.config.kd * derivative

        if self.config.thresholds:
            if output > self.config.thresholds[1]:
                output = self.config.thresholds[1]
                self.integral -= filtered_error * dt  # Anti-windup adjustment
            elif output < self.config.thresholds[0]:
                output = self.config.thresholds[0]
                self.integral -= filtered_error * dt  # Anti-windup adjustment

        return output

    def _adapt_kp(self, error: float) -> float:
        """
        Adaptive proportional gain. It will be more aggressive for large errors, and keep the initial value for
         smaller ones.
        """
        scaling_factor = 1 + 0.5 * (abs(error) / self.dead_zone_adaptive_threshold)
        return self.config.kp * scaling_factor

    def _adapt_ki(self, error: float) -> float:
        """Adaptive integral gain. Reduce integral action for large errors to prevent overshoot."""
        return self.config.ki / (1 + 0.1 * abs(error) / self.dead_zone_adaptive_threshold)

    def reset(self) -> None:
        """Resets the controller state."""
        self.last_error = 0.0
        self.integral = 0.0
        self.low_pass_filter.reset()
        self.prev_time = time.perf_counter()

    def update_config(
            self,
            kp: float | None = None,
            ki: float | None = None,
            kd: float | None = None,
            target_error: float | None = None,
            thresholds: tuple[float, float] | None = None
    ) -> None:
        if kp is not None:
            self.config.kp = kp
        if ki is not None:
            self.config.ki = ki
        if kd is not None:
            self.config.kd = kd
        if target_error is not None:
            self.config.target_error = target_error
        if thresholds is not None:
            self.config.thresholds = thresholds


if __name__ == '__main__':
    # Usage:
    import random
    from auto_follow.utils.path_manager import Paths

    pid_controller_x = PIDController(config=PIDConfig.from_yaml(path=Paths.PID_X_PATH))
    print(f"{pid_controller_x.config=}")
    current_value = 100.0
    target_value = 0
    pid_controller_x.update_config(target_error=target_value)

    for i in range(100):
        output = pid_controller_x.compute(current_value)
        current_value += output * 0.1

        current_value += random.uniform(-0.5, 0.5)  # Noise

        print(f"Step {i}: Value -> {output:.2f} | Output -> {output:.2f}")
        time.sleep(0.1)
