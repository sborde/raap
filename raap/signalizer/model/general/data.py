from typing import Self

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable


class Interval(HDFExportable):
    def __init__(self, start: float, end: float):
        if start > end:
            raise ValueError("Interval start must be less than or equal to interval end.")
        self._start = start
        self._end = end

    def __repr__(self) -> str: return f"Interval({self._start}, {self._end})"

    def __str__(self) -> str: return f'[{self._start}, {self._end}]'

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Interval(self._start / other, self._end / other)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.start + other.start, self.end + other.end)
        if isinstance(other, (float, int)):
            return Interval(self.start + other, self.end + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.start - other.start, self.end - other.end)
        if isinstance(other, (float, int)):
            return Interval(self.start - other, self.end - other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Interval(self.start * other, self.end * other)
        else:
            return NotImplemented

    def __len__(self) -> int: return 2

    def __getitem__(self, index):
        if index == 0:
            return self._start
        elif index == 1:
            return self._end
        else:
            raise IndexError("Interval index out of range")

    @property
    def start(self): return self._start

    @property
    def end(self): return self._end

    @property
    def length(self): return self.end - self.start

    def contains(self, value: float | Self) -> bool:
        """Checks if a value is within the interval."""
        if isinstance(value, Interval):
            return self.start <= value.start and self.end >= value.end
        return self.start <= value <= self.end

    def astype(self, dtype):
        if dtype is int:
            return Interval(int(self.start), int(self.end))
        if dtype is float:
            return Interval(float(self.start), float(self.end))
        raise ValueError('Unsupported dtype.')

    @classmethod
    def as_interval(cls, iterable):
        """Converts an iterable (list, NumPy array, Interval) to an Interval."""
        if isinstance(iterable, cls):
            return iterable

        if isinstance(iterable, (list, np.ndarray)):
            if len(iterable) != 2:
                raise ValueError("Input must have length 2.")
            try:
                return cls(iterable[0], iterable[1])
            except TypeError:
                raise TypeError("Input must contain numerical values.")

        raise TypeError("Input must be an Interval, list, or NumPy array.")

    def save(self, group: h5py.Group):
        group.create_dataset('start', data=self.start)
        group.create_dataset('end', data=self.end)

    @classmethod
    def load(cls, group: h5py.Group):
        return Interval(group['start'][()], group['end'][()])

