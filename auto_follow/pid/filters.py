class LowPassFilter:
    """Smooths the error signal and reduces the effect of sudden fluctuations by removing the high-frequency noise."""

    def __init__(self, alpha: float = 0.2):
        """

        :param alpha: Smoothing factor, between 0 (no smoothing) and 1 (tons of smoothing).
        """
        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha \"{alpha}\" must be between 0 and 1. ")
        self.alpha = alpha
        self.filtered_value = None

    def filter(self, raw_value: float) -> float:
        if self.filtered_value is None:
            self.filtered_value = raw_value
        else:
            self.filtered_value = self.alpha * raw_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

    def reset(self) -> None:
        self.filtered_value = None
