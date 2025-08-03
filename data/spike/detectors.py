

from typing import Optional
import numpy as np

from data.spikes import SpikeShape
from .processors import SavGolFilter

class SpikeStartFinder:
    """
    A functor class for finding the start index of a spike shape.

    Spike shape's start is defined as the first index where the first derivative
    is at or above a certain threshold relative to the maximum derivative value.
    Optionally, it can smooth the signal with a Savitzky-Golay filter before
    calculating the derivative.
    """

    def __init__(
        self,
        start_threshold: float = 0.1,
        smoother: Optional[SavGolFilter] = None
    ):
        self._smooth = smoother
        self._start_threshold = start_threshold

    def __call__(self, spike: SpikeShape) -> Optional[int]:
        """
        Find the start index of a spike.

        It first smooths the signal, then calculates the first derivative.
        We consider threshold dV value as the given fraction of the maximum
        derivative value before the maximum. If the signal never reaches this
        threshold, it returns None. Else, it returns the first index where the
        derivative is at or above the threshold.
        """
        dv = self._calculate_dv(spike)
        if len(dv) == 0:
            return None
        max_dv_i = np.argmax(dv)
        threshold_dv = self._start_threshold * dv[max_dv_i]
        below_index = np.where(dv[:max_dv_i] < threshold_dv)[0]

        return below_index[-1] + 1 if len(below_index) > 0 else None

    def _calculate_dv(self, spike: SpikeShape) -> np.ndarray:
        """
        Calculates the first derivative to search for the start on it.

        The signal is first cut up to the peak index, and after an optional
        smoothing, the first derivative is returned.
        """
        smoothed = self._smooth(spike) if self._smooth else spike
        peak_index = np.argmin(np.abs(smoothed.times - 0)).astype(int)
        v = smoothed.values[:peak_index + 1]
        return np.diff(v)
