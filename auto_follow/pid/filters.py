import numpy as np
from collections import deque
from scipy.signal import firwin

from auto_follow.pid.config import FilterConfig


class LowPassFilter:
    """
    Simple low-pass filter.

    Smooths the error signal and reduces the effect of sudden fluctuations by removing the high-frequency noise.
    It ignores quick sudden changes, and it pays attention to slow and important changes (like needing to turn).

    This class provides two filtering methods:
        1. Exponential smoothing.
        2. Convolution-based FIR filtering with configurable frequency response.
    """

    def __init__(self, config: FilterConfig):
        """
        LowPassFilter. wn represents the cutoff frequency (transition between pass-band and stop-band)
        """
        self.config = config
        self.wn = (self.config.wp + self.config.ws) / 2
        self.buffer = deque(maxlen=self.config.filter_order + 1)
        self.filtered_value = None

        self.coefficients = self._design_filter()

    def __call__(self, raw_value: float) -> float:
        if self.config.mode == "fir":
            return self.fir_filter(raw_value)
        return self.exp_filter(raw_value)

    def _design_filter(self) -> np.ndarray:
        """
        Design a low-pass FIR filter using the window method.
        Returns the filter coefficients.
        """
        return firwin(self.config.filter_order + 1, self.wn, window='hamming', pass_zero=True, fs=2)

    def exp_filter(self, raw_value: float) -> float:
        """
        Apply exponential smoothing filter.

        :param raw_value: The input value to be filtered.
        :return: Filtered value.
        """
        if self.filtered_value is None:
            self.filtered_value = raw_value
        else:
            self.filtered_value = self.config.alpha * raw_value + (1 - self.config.alpha) * self.filtered_value
        return self.filtered_value

    def fir_filter(self, raw_value: float) -> float:
        """
        Apply FIR filtering using convolution with the designed filter.
        This method maintains a buffer and performs the convolution as each new sample arrives.

        If there are not enough samples it will use the exponential smoothing filter.

        :param raw_value: The input value to be filtered.
        :return: Filtered value.
        """
        self.buffer.append(raw_value)

        if len(self.buffer) < len(self.coefficients):
            return self.exp_filter(raw_value)

        return np.dot(self.coefficients, np.array(self.buffer))

    def reset(self) -> None:
        """Reset filter state."""
        self.filtered_value = None
        self.buffer.clear()
