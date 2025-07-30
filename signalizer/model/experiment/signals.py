import abc
from typing import Self, Optional, Protocol

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable
from signalizer.model.general.series import DataSeries
from signalizer.model.measurements import TimeUnit, VoltageUnit
from signalizer.model.temporal.data import TimeInterval, TimeStamp
from signalizer.model.temporal.series import TimeSeries


class SignalLike(Protocol):
    @property
    @abc.abstractmethod
    def times(self) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def values(self) -> np.ndarray: ...

    @property
    @abc.abstractmethod
    def sample_count(self) -> int: ...

    @property
    @abc.abstractmethod
    def value_unit(self) -> VoltageUnit: ...

    @property
    @abc.abstractmethod
    def time_unit(self) -> TimeUnit: ...


class Signal(TimeSeries):

    def __init__(self, times: np.ndarray, values: np.ndarray,
                 si: float = None,
                 time_unit: TimeUnit = TimeUnit.SECONDS,
                 value_unit: VoltageUnit = VoltageUnit.VOLTS):
        super().__init__(times, values, si, time_unit)
        self._value_unit = value_unit

    @property
    def si(self) -> float: return self.dt

    @property
    def fs(self) -> int: return int(1 / self.time_unit.to_seconds(self.si))

    @property
    def value_unit(self) -> VoltageUnit: return self._value_unit

    def triggered_portion(self, trigger: TimeStamp, window: TimeInterval):
        return Signal.from_series(super().triggered_portion(trigger, window))

    def interval_portion(self, interval: TimeInterval):
        return Signal.from_series(super().interval_portion(interval))

    def time_shifted(self, offset: float) -> Self:
        return Signal.from_series(super().time_shifted(offset))

    def resampled(self, target_fs: int) -> Self:
        return Signal.from_series(super().resampled(target_fs))

    @classmethod
    def from_series(cls, series: DataSeries):
        if isinstance(series, cls):
            return series
        if isinstance(series, TimeSeries):
            return Signal(series.times, series.values, series.dt, series.time_unit)
        if isinstance(series, DataSeries):
            return Signal(series.x, series.y)

        raise TypeError('Input must be a DataSeries object.')

    def save(self, group: h5py.Group):
        super().save(group)
        group.create_dataset('value_unit', data=self.value_unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        time_series = super().load(group)
        value_unit_name = group['value_unit'][()].decode('utf-8')
        value_unit = getattr(VoltageUnit, value_unit_name)
        return Signal(time_series.times, time_series.values, time_series.dt,
                      time_series.time_unit, value_unit)


class SignalMatrix(HDFExportable):

    def __init__(self,
                 times: np.ndarray,
                 values: np.ndarray,
                 si: Optional[float] = None,
                 time_unit: TimeUnit = TimeUnit.SECONDS,
                 value_unit: VoltageUnit = VoltageUnit.VOLTS):
        times = times.squeeze()

        if times.ndim != 1:
            raise ValueError('times must be vector or a 1D matrix.')

        if times.size != values.shape[0]:
            raise ValueError('times and values must have the same number of samples.')

        self._times = times
        self._values = values
        self._si = si or float(np.mean(np.diff(times)))
        self._time_unit = time_unit
        self._value_unit = value_unit

    def __len__(self) -> int: return self._values.shape[1]

    def __getitem__(self, item) -> Signal:
        return Signal(self._times, self._values[:, item], self.si, self._time_unit, self._value_unit)

    def __delitem__(self, key):
        self._values = np.delete(self._values, key, axis=1)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def times(self) -> np.ndarray: return self._times

    @property
    def si(self): return self._si

    @property
    def fs(self) -> int: return int(1 / self._time_unit.to_seconds(self.si))

    @property
    def time_unit(self) -> TimeUnit: return self._time_unit

    @property
    def value_unit(self) -> VoltageUnit: return self._value_unit

    def average(self) -> Signal:
        """Returns the average of the signals."""
        return Signal(self._times, np.mean(self._values, axis=1), self.si, self._time_unit, self._value_unit)

    def as_matrix(self) -> np.ndarray: return self._values.T

    def save(self, group: h5py.Group):
        group.create_dataset('times', data=self._times)
        group.create_dataset('values', data=self._values)
        group.create_dataset('si', data=self._si)
        group.create_dataset('time_unit', data=self._time_unit.name)
        group.create_dataset('value_unit', data=self._value_unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        time_unit_name = group['time_unit'][()].decode('utf-8')
        time_unit = getattr(TimeUnit, time_unit_name)
        value_unit_name = group['value_unit'][()].decode('utf-8')
        value_unit = getattr(VoltageUnit, value_unit_name)
        return SignalMatrix(group['times'][:], group['values'][:], group['si'][()],
                            time_unit, value_unit)

    @classmethod
    def concat(cls, matrices: list['SignalMatrix']) -> 'SignalMatrix':

        for matrix in matrices:
            if not np.array_equal(matrix.times, matrices[0].times):
                raise ValueError('All matrices must have the same time vector.')

        times = matrices[0].times
        values = np.concat([matrix.as_matrix() for matrix in matrices], axis=0)
        return cls(times, values.T,
                   matrices[0].si,
                   matrices[0].time_unit,
                   matrices[0].value_unit)