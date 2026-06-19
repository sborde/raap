

from dataclasses import dataclass
from typing import Self

import numpy as np

@dataclass
class Signal:
    """
    Represents a signal with time points and corresponding values.
    """
    t: np.ndarray
    v: np.ndarray
    si: float

    @property
    def start(self) -> float:
        """Returns the start time of the signal."""
        return self.t[0]

    @property
    def end(self) -> float:
        """Returns the end time of the signal."""
        return self.t[-1]

    @property
    def duration(self) -> float:
        """Returns the duration of the signal."""
        return self.end - self.start

    @property
    def fs(self) -> int:
        """Returns the sampling frequency of the signal."""
        return int(1 / self.si)

    def resample(self, fs: int) -> Self:
        """Returns a resampled Signal with the specified sampling frequency."""
        if self.fs == fs:
            return self
        si = 1 / fs
        n_samples = int(self.duration / si) + 1
        t = np.array(range(n_samples)) * si + self.start
        v = np.interp(t, self.t, self.v)
        return type(self)(t, v, si)

def count_samples(dt: float, si: float) -> int:
    """
    Counts the number of samples in a given time interval.
    
    Args:
        dt (float): Duration of the interval in seconds.
        si (float): Sampling interval in seconds.
    
    Returns:
        int: Number of samples in the interval.
    """
    return int(np.round(dt / si))

def cut_at(signal: Signal, time: float, window: tuple[float, float]) -> Signal:
    """
    Cut the signal around the given timestamp within a specified time window.
    """
    time_i = int(np.round((time - signal.start) / signal.si))
    window_i = tuple(int(np.round(t / signal.si)) for t in window)
    i, j = time_i + window_i[0], time_i + window_i[1]

    return Signal(signal.t[i:j+1] - time, signal.v[i:j+1], signal.si)
