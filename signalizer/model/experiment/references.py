import abc

import h5py

from signalizer.model.temporal.data import TimeStamp, TimeInterval


class AttributeContainer:
    """Contains the attributes of named object."""

    def __init__(self, name: str, attributes: dict = None):
        self._name = name
        self._attributes = attributes or {}

    def __repr__(self): return f'AttributeContainer({self._name})'

    def __str__(self): return f'{self._name} {self._attributes}'

    @property
    def name(self) -> str: return self._name

    @property
    def attributes(self) -> dict: return self._attributes

    def save(self, group: h5py.Group):
        group.create_dataset('name', data=self.name)
        if self.attributes:
            property_group = group.create_group('attributes')
            for key, value in self.attributes.items():
                property_group.create_dataset(key, data=value)

    @classmethod
    def load(cls, group: h5py.Group) -> 'AttributeContainer':
        name = group['name'][()].decode('utf-8')
        if 'attributes' in group:
            attributes = {key: group['attributes'][key][()] for key in group['attributes']}
        else:
            attributes = None
        return cls(name, attributes)


class TimeReferenced(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def attributes(self) -> dict: pass


class Event(TimeReferenced):
    """Represents an event in a recording, acts as a named TimeStamp."""
    def __init__(self, timestamp: TimeStamp, name: str, attributes: dict = None):
        self._timestamp = timestamp
        self._attributes = AttributeContainer(name, attributes)

    def __repr__(self):
        return f'Event({self._timestamp}, {self.name})'

    def __str__(self):
        return f'{self.name} ({self._timestamp})'

    @property
    def timestamp(self) -> TimeStamp: return self._timestamp

    @property
    def name(self) -> str: return self._attributes.name

    @property
    def attributes(self) -> dict: return self._attributes.attributes

    @property
    def time(self) -> float: return self._timestamp.time

    def save(self, group: h5py.Group):
        self._attributes.save(group)
        self._timestamp.save(group.create_group('timestamp'))

    @classmethod
    def load(cls, group: h5py.Group):
        attributes = AttributeContainer.load(group)
        timestamp = TimeStamp.load(group['timestamp'])
        return Event(timestamp, attributes.name, attributes.attributes)


class Episode(TimeReferenced):
    """Represents an episode in a recording, acts as a named TimeInterval."""
    def __init__(self, interval: TimeInterval, name: str, attributes: dict = None):
        self._interval = interval
        self._attributes = AttributeContainer(name, attributes)

    def __repr__(self):
        return f'Episode({self._interval}, {self.name})'

    def __str__(self):
        return f'{self.name} {self._interval} episode'

    @property
    def interval(self) -> TimeInterval: return self._interval

    @property
    def name(self) -> str: return self._attributes.name

    @property
    def attributes(self) -> dict: return self._attributes.attributes

    @property
    def start(self) -> float: return self._interval.start

    @property
    def end(self) -> float: return self._interval.end

    def save(self, group: h5py.Group):
        self._attributes.save(group)
        self._interval.save(group.create_group('interval'))

    @classmethod
    def load(cls, group: h5py.Group):
        attributes = AttributeContainer.load(group)
        interval = TimeInterval.load(group['interval'])
        return Episode(interval, attributes.name, attributes.attributes)
