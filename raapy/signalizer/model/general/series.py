from typing import Self

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable


class DataSeries(HDFExportable):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.squeeze(x)
        y = np.squeeze(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('x and y must be vectors or a 1D matrix.')

        if x.size != y.size:
            raise ValueError('x and y must have the same size.')

        self._x = x
        self._y = y

    @property
    def x(self) -> np.ndarray: return self._x

    @property
    def y(self) -> np.ndarray: return self._y

    def _get_slice(self, start_i: int, end_i: int) -> Self:
        """Returns a slice of the DataSeries between the two indices"""
        if not (0 <= start_i < len(self.x)) or not (0 <= end_i < len(self.x)):
            raise IndexError('Slice is out of bound.')
        return DataSeries(self.x[start_i:(end_i + 1)], self.y[start_i:(end_i + 1)])

    def save(self, group: h5py.Group):
        group.create_dataset('x', data=self.x)
        group.create_dataset('y', data=self.y)

    @classmethod
    def load(cls, group: h5py.Group):
        return DataSeries(group['x'][:], group['y'][:])


