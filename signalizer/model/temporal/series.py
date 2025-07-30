import abc
from typing import Self, Protocol

import h5py
import numpy as np

from signalizer.model.general.data import Interval
from signalizer.model.general.series import DataSeries
from signalizer.model.measurements import TimeUnit
from signalizer.model.temporal.data import TimeInterval, TimeStamp


class TimeSeriesLike(Protocol):

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
    def time_unit(self) -> TimeUnit: ...


class TimeSeries(DataSeries):

    def __init__(self, time: np.ndarray, y: np.ndarray, dt: float = None, unit: TimeUnit = TimeUnit.SECONDS):
        super().__init__(time, y)
        self._dt = dt or float(np.mean(np.diff(time)))
        self._unit = unit

    @property
    def times(self) -> np.ndarray: return self.x

    @property
    def values(self) -> np.ndarray: return self.y

    @property
    def sample_count(self) -> int: return len(self.times)

    @property
    def dt(self) -> float: return self._dt

    @property
    def time_unit(self) -> TimeUnit: return self._unit

    @property
    def duration(self) -> float:
        """Returns the duration of the TimeSeries"""
        return float(self.times[-1] - self.times[0])

    def time_range(self) -> TimeInterval:
        """Returns the time range interval of the TimeSeries"""
        interval = Interval(float(self.times[0]), float(self.times[-1]))
        return TimeInterval(interval, self.time_unit)

    def is_valid_trigger(self, trigger: TimeStamp, window: TimeInterval) -> bool:
        """Checks if the trigger is valid in the TimeSeries"""
        return self.time_range().contains(window + trigger)

    def triggered_portion(self, trigger: TimeStamp, window: TimeInterval) -> Self:
        """Returns a TimeSeries portion within the given interval relative to trigger"""

        # Convert trigger and window to the same unit as the timeseries
        trigger = trigger.converted(self.time_unit)
        window = window.converted(self.time_unit)

        trigger_i = self._get_time_index(trigger.time)
        cut_i = (window.interval / self.dt + trigger_i).astype(int)

        return self._get_slice(cut_i.start, cut_i.end)

    def interval_portion(self, time_interval: TimeInterval):
        """Returns a TimeSeries portion within the given interval"""

        # Convert interval to the same unit as the timeseries
        interval = time_interval.converted(self.time_unit).interval

        start_i = self._get_time_index(interval[0])
        end_i = self._get_time_index(interval[1])

        return self._get_slice(start_i, end_i)

    def time_shifted(self, offset: float) -> Self:
        """Returns a new TimeSeries with a shifted time vector"""
        return TimeSeries(self.x + offset, self.y)

    def resampled(self, target_fs: int) -> Self:
        """Returns a new TimeSeries resampled to the target rate"""
        target_si = self.time_unit.from_seconds(1 / target_fs)

        new_sample_count = int(self.duration / target_si)  # truncate, else last point could be nan
        corrected_duration = new_sample_count * target_si  # new time vector should end on sampling point

        resampled_time = np.linspace(self.times[0], self.times[0] + corrected_duration, new_sample_count + 1)
        resampled_value = np.interp(resampled_time, self.times, self.values)

        return TimeSeries(resampled_time, resampled_value, target_si, self.time_unit)

    def _get_time_index(self, time: float) -> int:
        """Returns the index of the sampling point closest to time"""
        return int(np.round((time - self.times[0]) / self.dt))

    def _get_slice(self, start_i: int, end_i: int) -> Self:
        """Returns a slice of the TimeSeries between the two indices"""
        series = super()._get_slice(start_i, end_i)
        return TimeSeries(series.x, series.y, self.dt, self.time_unit)

    @classmethod
    def from_series(cls, series: DataSeries):
        if isinstance(series, cls):
            return series
        if isinstance(series, DataSeries):
            return TimeSeries(series.x, series.y)

        raise TypeError('Input must be a DataSeries object.')

    def save(self, group: h5py.Group):
        super().save(group)
        group.create_dataset('dt', data=self.dt)
        group.create_dataset('unit', data=self.time_unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        series = super().load(group)
        dt = group['dt'][()]
        unit_name = group['unit'][()].decode('utf-8')
        unit = getattr(TimeUnit, unit_name)
        return TimeSeries(series.x, series.y, dt, unit)
