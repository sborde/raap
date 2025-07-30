"""
This module contains the classes for representing series and matrices of series.
"""

import numpy as np

class Signal:
    """
    A class to represent a signal.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray):
        self._times = times
        self._values = values

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the signal.
        """
        return self._times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the signal.
        """
        return self._values


class DataSeries:
    """
    A class to represent a general data series.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self._y = y

    @property
    def x(self) -> np.ndarray:
        """
        Get the x values of the series.
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        Get the y values of the series.
        """
        return self._y

    @property
    def area(self) -> float:
        """
        Get the area of the series.
        """
        return 0.5 * np.abs(
            np.dot(self.x, np.roll(self.y, 1)) -
            np.dot(self.y, np.roll(self.x, 1))
        )

    def __len__(self) -> int:
        """
        Get the number of samples in the series.
        """
        return len(self.x)


class SeriesMatrix:
    """
    A class to represent a matrix of series.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self._y = y

    @property
    def x(self) -> np.ndarray:
        """
        Get the x values of the series.
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        Get the y values of the series.
        """
        return self._y

    def __len__(self) -> int:
        """
        Get the number of series.
        """
        return len(self._x)

    def __getitem__(self, index: int) -> DataSeries:
        """
        Get the series at the given index.
        """
        return DataSeries(self._x[index], self._y[index])


class SegmentMatrix:
    """
    A class to represent a matrix of segments with a common time axis.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray):
        self._times = times
        self._values = values

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the segments.
        """
        return self._times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the segments.
        """
        return self._values

    def __len__(self) -> int:
        """
        Get the number of segments.
        """
        return self._values.shape[0]

    def __getitem__(self, index: int) -> Signal:
        """
        Get the segment at the given index.
        """
        return Signal(self._times, self._values[index])
