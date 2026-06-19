from typing import Self

import h5py

from signalizer.io.protocols import HDFExportable
from signalizer.model.general.data import Interval
from signalizer.model.measurements import TimeUnit


class TimeStamp(HDFExportable):
    """Represents a timestamp."""

    def __init__(self, time: float | int, unit: TimeUnit = TimeUnit.SECONDS):
        self._time = time
        self._unit = unit

    def __repr__(self):
        return f'TimeStamp({self._time}, {self._unit})'

    def __str__(self):
        return f'{self._time} {self._unit}'

    @property
    def time(self) -> float: return self._time

    @property
    def unit(self) -> TimeUnit: return self._unit

    def converted(self, new_unit: TimeUnit) -> 'TimeStamp':
        """Convert the timestamp to a new unit."""
        converted_time = self._unit.convert(self._time, new_unit)
        return TimeStamp(converted_time, new_unit)

    def save(self, group: h5py.Group):
        group.create_dataset('time', data=self._time)
        group.create_dataset('unit', data=self._unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        time = group['time'][()]
        unit_name = group['unit'][()].decode('utf-8')
        unit = getattr(TimeUnit, unit_name)
        return TimeStamp(time, unit)


class TimeInterval(HDFExportable):
    """Represents a time interval in a signal."""
    def __init__(self, interval: Interval, unit: TimeUnit = TimeUnit.SECONDS):
        self._interval = interval
        self._unit = unit

    def __repr__(self):
        return f'TimeInterval({self._interval}, {repr(self._unit)})'

    def __str__(self):
        return f'{self.interval} {self._unit}'

    def __add__(self, other : TimeStamp | float | int):
        if isinstance(other, TimeStamp):
            other = other.converted(self._unit).time
        return TimeInterval(self._interval + other, self._unit)

    def __sub__(self, other : TimeStamp | float | int):
        if isinstance(other, TimeStamp):
            other = other.converted(self._unit).time
        return TimeInterval(self._interval - other, self._unit)

    def __mul__(self, other):
        return TimeInterval(self._interval * other, self._unit)

    def __truediv__(self, other):
        return TimeInterval(self._interval / other, self._unit)

    @property
    def interval(self) -> Interval: return self._interval

    @property
    def unit(self) -> TimeUnit: return self._unit

    @property
    def start(self): return self._interval.start

    @property
    def end(self): return self._interval.end

    @property
    def length(self): return self.end - self.start

    def contains(self, value: TimeStamp | Self) -> bool:
        """Checks if a value is within the interval."""
        if isinstance(value, TimeInterval):
            value = value.converted(self._unit)
            return self.start <= value.start and self.end >= value.end
        if isinstance(value, TimeStamp):
            value = value.converted(self._unit).time
            return self.start <= value <= self.end
        else:
            raise TypeError("Value must be a TimeStamp or TimeInterval.")

    def converted(self, new_unit: TimeUnit) -> 'TimeInterval':
        """Convert the interval to a new unit."""
        converted_interval = Interval(
            self._unit.convert(self._interval.start, new_unit),
            self._unit.convert(self._interval.end, new_unit)
        )
        return TimeInterval(converted_interval, new_unit)

    def save(self, group: h5py.Group):
        self._interval.save(group)
        group.create_dataset('unit', data=self.unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        interval = Interval.load(group)
        unit_name = group['unit'][()].decode('utf-8')
        unit = getattr(TimeUnit, unit_name)
        return TimeInterval(interval, unit)
