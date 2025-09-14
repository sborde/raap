from typing import Iterable, Self

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable
from signalizer.model.general.data import Interval


class IntervalSequence(HDFExportable):

    def __init__(self, starts: Iterable[float], ends: Iterable[float]):
        starts = np.asarray(starts).squeeze()
        ends = np.asarray(ends).squeeze()

        if starts.ndim == 0:
            starts = np.array([starts])

        if ends.ndim == 0:
            ends = np.array([ends])


        if starts.ndim != 1 or ends.ndim != 1:
            raise ValueError('Input vectors must be 1D vector or row or column matrix.')
        if starts.size != ends.size:
            raise ValueError('Input vectors must have the same size.')
        if np.any(ends < starts):
            raise ValueError('Interval starts must precede its end.')

        self._starts = starts
        self._ends = ends

    def __repr__(self):
        return f'IntervalSequence({self._starts, self._ends})'

    def __str__(self) -> str:
        string = 'IntervalSequence:\n'
        for start, end in zip(self._starts, self._ends):
            string += f'[{start}, {end}]\n'
        return string

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index) -> Interval | Self:
        if isinstance(index, slice):
            return IntervalSequence(self._starts[index], self._ends[index])
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise ValueError('Index is out of bounds.')
            return Interval(float(self._starts[index]), float(self._ends[index]))
        raise TypeError('Invalid index.')

    def __delitem__(self, key):
        if isinstance(key, int):
            self._starts = np.delete(self._starts, key)
            self._ends = np.delete(self._ends, key)
        else:
            raise TypeError('Invalid index.')

    def __iter__(self):
        for start, end in zip(self._starts, self._ends):
            yield Interval(start, end)

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return IntervalSequence(self._starts / other, self._ends / other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return IntervalSequence(self._starts * other, self._ends * other)
        raise TypeError('Unsupported type for multiplication.')

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    def save(self, group: h5py.Group):
        group.create_dataset('starts', data=self._starts)
        group.create_dataset('ends', data=self._ends)

    @classmethod
    def load(cls, group: h5py.Group):
        starts = group['starts'][:]
        ends = group['ends'][:]
        return IntervalSequence(starts, ends)


class ValueSequence(HDFExportable, Iterable[float]):

    def __init__(self, values: Iterable[float]):
        self._values = np.asarray(values).squeeze()

    def __str__(self): return str(self.values)

    def __len__(self) -> int: return self._values.size

    def __getitem__(self, index) -> float | Self:
        if isinstance(index, slice):
            return ValueSequence(self._values[index])
        if isinstance(index, int):
            if not (0 <= index < len(self)):
                raise IndexError('Index is out of bounds.')
            return float(self._values[index])
        raise TypeError('Invalid index.')

    def __delitem__(self, key):
        if isinstance(key, int):
            self._values = np.delete(self._values, key)
        else:
            raise TypeError('Invalid index.')

    def __iter__(self): return iter(self._values)

    @property
    def values(self) -> np.ndarray:
        return self._values

    def save(self, group: h5py.Group):
        group.create_dataset('values', data=self._values)

    @classmethod
    def load(cls, group: h5py.Group):
        values = group['values'][:]
        return ValueSequence(values)


