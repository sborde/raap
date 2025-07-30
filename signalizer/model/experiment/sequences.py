import abc
from typing import Self

import h5py

from signalizer.model.experiment.references import Event, Episode
from signalizer.model.measurements import TimeUnit
from signalizer.model.temporal.sequences import TimeStampSequence, TimeIntervalSequence


class TimeReferencedVector(abc.ABC):
    """A vector of time-referenced data."""

    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def attributes(self) -> dict: pass


class EventVector(TimeReferencedVector):

    def __init__(self, timestamps: TimeStampSequence, name: str, attributes: dict = None):
        self._timestamps = timestamps
        self._name = name
        self._attributes = attributes or {}

    def __repr__(self): return f'EventVector({repr(self._timestamps)}, {self._name})'

    def __str__(self) -> str: return f'{self.name} (n={len(self)})'

    def __len__(self) -> int: return len(self._timestamps)

    def __getitem__(self, index) -> Event | Self:
        if isinstance(index, slice):
            return EventVector(self._timestamps[index], self._name, self._attributes)
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise ValueError('Index is out of bounds.')
            return Event(self._timestamps[index], self._name, self._attributes)
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
    def unit(self): return self._timestamps.unit

    @property
    def name(self) -> str: return self._name

    @property
    def attributes(self) -> dict: return self._attributes

    def converted(self, new_unit: TimeUnit) -> 'EventVector':
        """Convert the timestamps to a new unit."""
        converted_timestamps = self._timestamps.converted(new_unit)
        return EventVector(converted_timestamps, self.name, self._attributes)

    def save(self, group: h5py.Group):
        event_group = group.create_group('events')
        self._timestamps.save(event_group)
        event_group.create_dataset('name', data=self.name)

    @classmethod
    def from_list(cls, events: list[Event]) -> 'EventVector':
        timestamps = TimeStampSequence.from_list([event.timestamp for event in events])
        name = events[0].name if events else ''
        return EventVector(timestamps, name)

    @classmethod
    def load(cls, group: h5py.Group):
        """Load the EventSequence from the HDF5 group."""
        event_group = group['events']
        timestamps = TimeStampSequence.load(event_group)
        name = event_group['name'][()].decode('utf-8')
        return EventVector(timestamps, name)


class EpisodeSequence(TimeReferencedVector):

    def __init__(self, intervals: TimeIntervalSequence, name: str, attributes: dict = None):
        self._intervals = intervals
        self._name = name
        self._attributes = attributes or {}

    def __repr__(self): return f'EpisodeSequence({repr(self._intervals), self._name})'

    def __str__(self) -> str:
        string = f'{len(self)} {self._name} episodes\n'
        for interval in self._intervals:
            string += f'{interval}\n'
        return string

    def __len__(self) -> int: return len(self._intervals)

    def __getitem__(self, index) -> Episode | Self:
        if isinstance(index, slice):
            return EpisodeSequence(self._intervals[index], self._name)
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise ValueError('Index is out of bounds.')
            return Episode(self._intervals[index], self._name)
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
    def unit(self) -> TimeUnit: return self._intervals.unit

    @property
    def name(self) -> str: return self._name

    @property
    def attributes(self) -> dict: return self._attributes

    def converted(self, new_unit: TimeUnit) -> 'EpisodeSequence':
        """Convert the interval to a new unit."""
        converted_intervals = self._intervals.converted(new_unit)
        return EpisodeSequence(converted_intervals, self._name, self._attributes)

    def save(self, group: h5py.Group):
        self._intervals.save(group)
        group.create_dataset('name', data=self.name)

    @classmethod
    def load(cls, group: h5py.Group):
        intervals = TimeIntervalSequence.load(group)
        name = group['name'][()].decode('utf-8')
        return EpisodeSequence(intervals, name)
