from typing import Self

import h5py

from signalizer.io.protocols import HDFExportable
from signalizer.model.general.sequences import IntervalSequence, ValueSequence
from signalizer.model.measurements import TimeUnit
from signalizer.model.temporal.data import TimeInterval, TimeStamp


class TimeStampSequence(HDFExportable):

    def __init__(self, timestamps: ValueSequence, unit: TimeUnit = TimeUnit.SECONDS):
        self._timestamps = timestamps
        self._unit = unit

    def __repr__(self): return f'TimeStampSequence({repr(self._timestamps)})'

    def __str__(self) -> str:
        string = 'TimeStampSequence:\n'
        for timestamp in self:
            string += f'{timestamp}\n'
        return string

    def __len__(self) -> int: return len(self._timestamps)

    def __getitem__(self, index) -> TimeStamp:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise ValueError('Index is out of bounds.')
            return TimeStamp(self._timestamps[index], self._unit)
        raise TypeError('Invalid index.')

    def __delitem__(self, key):
        if isinstance(key, int):
            del self._timestamps[key]
        else:
            raise TypeError('Invalid index.')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def unit(self): return self._unit

    def converted(self, new_unit: TimeUnit) -> 'TimeStampSequence':
        """Convert the timestamps to a new unit."""
        converted_timestamps = ValueSequence(
            self.unit.convert(self._timestamps.values, new_unit)
        )
        return TimeStampSequence(converted_timestamps, new_unit)

    def save(self, group: h5py.Group):
        self._timestamps.save(group)
        group.create_dataset('unit', data=self._unit.name)

    @classmethod
    def from_list(cls, timestamps: list[TimeStamp]) -> 'TimeStampSequence':
        """Create a TimeStampSequence from a list of timestamps."""
        unit = timestamps[0].unit
        value_sequence = ValueSequence([timestamp.converted(unit).time for timestamp in timestamps])
        return cls(value_sequence, unit)

    @classmethod
    def load(cls, group: h5py.Group):
        timestamps = ValueSequence.load(group)
        unit_name = group['unit'][()].decode('utf-8')
        unit = getattr(TimeUnit, unit_name)
        return TimeStampSequence(timestamps, unit)


class TimeIntervalSequence(HDFExportable):

    def __init__(self, intervals: IntervalSequence, unit: TimeUnit = TimeUnit.SECONDS):
        self._intervals = intervals
        self._unit = unit

    def __repr__(self): return f'TimeIntervalSequence({repr(self._intervals), self._unit})'

    def __str__(self) -> str:
        string = 'TimeIntervalSequence:\n'
        for interval in self:
            string += f'{interval}\n'
        return string

    def __len__(self) -> int: return len(self._intervals)

    def __getitem__(self, index) -> TimeInterval:
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise ValueError('Index is out of bounds.')
            return TimeInterval(self._intervals[index], self._unit)
        raise TypeError('Invalid index.')

    def __delitem__(self, key):
        if isinstance(key, int):
            del self._intervals[key]
        else:
            raise TypeError('Invalid index.')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def unit(self) -> TimeUnit: return self._unit

    def converted(self, new_unit: TimeUnit) -> 'TimeIntervalSequence':
        """Convert the interval to a new unit."""

        converted_intervals = IntervalSequence(
            self.unit.convert(self._intervals.starts, new_unit),
            self.unit.convert(self._intervals.ends, new_unit)
        )

        return TimeIntervalSequence(converted_intervals, new_unit)

    def save(self, group: h5py.Group):
        self._intervals.save(group)
        group.create_dataset('unit', data=self.unit.name)

    @classmethod
    def load(cls, group: h5py.Group):
        intervals = IntervalSequence.load(group)
        unit_name = group['unit'][()].decode('utf-8')
        unit = getattr(TimeUnit, unit_name)
        return TimeIntervalSequence(intervals, unit)
