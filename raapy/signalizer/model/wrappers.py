import abc
from typing import Optional, Iterable, Sized, Protocol, TypeVar

import h5py

from signalizer.io.protocols import HDFExportable
from signalizer.model.experiment.segments import MatrixLike


class EntityLike(Protocol):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def version(self) -> str: ...

    @property
    @abc.abstractmethod
    def label(self) -> str: ...


class WrappedData[D](Protocol):
    @property
    @abc.abstractmethod
    def data(self) -> D: ...


class Entity:
    """Represents an analysis Entity. It contains a unique name, version and a set of attributes."""

    def __init__(self, name: str, attributes: dict):

        if 'version' not in attributes:
            raise ValueError("Entity must have a 'version' attribute.")

        self._name = name
        self._attributes = attributes

    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        else:
            raise AttributeError(f"Entity has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}'. Attributes are read-only.")

    def __repr__(self): return f'Entity({self._name})'

    def __str__(self): return self.label

    @property
    def name(self) -> str: return self._name

    @property
    def version(self) -> str: return self._attributes['version']

    @property
    def label(self) -> str: return f'{self.name}_{self.version}'

    def wrap[N](self, data: N) -> 'EntityData[N]':
        return EntityData[N](self, data)

    @classmethod
    def from_name_and_version(cls, name: str, version: str) -> 'Entity':
        """Creates an entity from a name and version"""
        return cls(name, {'version': version})


class EntityData[T]:
    """Represents a data object associated with an entity. It contains the entity and the data itself."""

    def __init__(self, entity: Entity, data: T):
        self._entity = entity
        self._data = data

    def __repr__(self) -> str:
        return f'EntityData({self._entity}, {self._data})'

    def __str__(self) -> str:
        return f'{self._entity} {self._data}'

    @property
    def name(self) -> str: return self._entity.name

    @property
    def version(self) -> str: return self._entity.version

    @property
    def label(self) -> str: return self._entity.label

    @property
    def entity(self) -> Entity: return self._entity

    @property
    def data(self) -> T: return self._data

    def with_data[N](self, new_data: N):
        return self.entity.wrap(new_data)


class Group[T]:

    def __init__(self, name: str, members: list[EntityData[T]], attributes: Optional[dict]):
        self._name = name
        self._members = members
        self._attributes = attributes or dict()

    def __len__(self) -> int: return len(self._members)

    def __getitem__(self, index: int) -> EntityData[T]: return self._members[index]

    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        else:
            raise AttributeError(f"Group has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}'. Attributes are read-only.")

    @property
    def name(self) -> str: return self._name

    @property
    def members(self) -> list[EntityData[T]]: return self._members

    @property
    def size(self) -> int: return len(self)


class Instance[R: HDFExportable, D: HDFExportable](HDFExportable):
    """Represents an instance of an experiment belonging to a reference."""
    def __init__(self, reference: R, data: D):
        self._reference = reference
        self._data = data

    def __str__(self):
        return f'{self._reference} related {self._data}'

    @property
    def reference(self) -> R:
        return self._reference

    @property
    def data(self) -> D:
        return self._data

    def save(self, group: h5py.Group):
        self._reference.save(group.create_group('reference'))
        self._data.save(group.create_group('data'))


R = TypeVar('R', bound=[Sized, Iterable, HDFExportable])
D = TypeVar('D', bound=[Sized, Iterable, HDFExportable, MatrixLike])


class InstanceMatrix[R, D](HDFExportable, Sized, Iterable[Instance[R, D]]):
    """Represents a sequence of experiment instances."""
    def __init__(self, references: R, data: D):
        if len(references) != len(data):
            raise ValueError('References and data must have the same length.')

        self._references = references
        self._data = data

    def __str__(self): return f'{len(self)} analysis instances'

    def __len__(self) -> int: return len(self._references)

    def __iter__(self):
        for reference, data in zip(self._references, self._data):
            yield Instance(reference, data)

    @property
    def reference(self) -> R: return self._references

    @property
    def data(self) -> D: return self._data

    def save(self, group: h5py.Group):
        self._references.save(group)
        self._data.save(group)
