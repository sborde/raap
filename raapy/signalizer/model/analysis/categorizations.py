import abc
from typing import Sequence

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable


class CategorizationSpace(HDFExportable, abc.ABC):
    """
    An interface for the description of the assigned categories.
    """

    @property
    @abc.abstractmethod
    def category_count(self) -> int: pass

    @abc.abstractmethod
    def convert_to_flag(self, assignment: int) -> np.ndarray: pass

    @classmethod
    def load(cls, group: h5py.Group):
        name = group['name'][()].decode('utf-8')

        if name == 'SOM':
            return SomCategorizationSpace.load(group)

        raise ValueError('Invalid categorization space')


class SomCategorizationSpace(CategorizationSpace):

    def __init__(self, map_size: Sequence[int]):
        if len(map_size) != 2:
            raise ValueError("map_size must be a tuple of two integers")

        self._map_size = map_size

    def __repr__(self):
        return f"SomCategorizationSpace(map_size={self._map_size})"

    def __str__(self):
        return f"BMMU on a {self._map_size[0]}x{self._map_size[1]} SOM"

    @property
    def category_count(self) -> int:
        return self._map_size[0] * self._map_size[1]

    @property
    def map_size(self): return self._map_size

    def convert_to_flag(self, assignment: int) -> np.ndarray:
        """
        Returns a category assignment as a binary matrix.
        """
        flag_matrix = np.zeros((self.category_count,), dtype=int)
        flag_matrix[assignment] = 1
        return flag_matrix.reshape(self._map_size, order='F')

    def save(self, group: h5py.Group):
        group.create_dataset('name', data='SOM')
        group.create_dataset('map_size', data=self._map_size)

    @classmethod
    def load(cls, group: h5py.Group) -> 'SomCategorizationSpace':
        map_size = group['map_size'][:]
        return SomCategorizationSpace(map_size)


class Categorization:
    """
    A class to represent a categorization of a set of items.
    """

    def __init__(self, category: int, space: CategorizationSpace):
        self._category = category
        self._space = space

    def __repr__(self):
        return f"Categorization(category={self._category})"

    def __str__(self):
        return f"{self._space}: {self._category}"

    @property
    def category(self) -> int:
        return self._category

    @property
    def space(self) -> CategorizationSpace:
        return self._space

    def as_matrix(self) -> np.ndarray:
        return self._space.convert_to_flag(self._category)


class CategorizationMatrix:
    """
    A class to represent a collection of category assignments.
    """

    def __init__(self, values: np.ndarray, space: CategorizationSpace):

        values = values.squeeze().astype(int)

        if np.any(np.logical_or(values < 0, values >= space.category_count)):
            raise ValueError(f'Invalid category, each value must be between [1, {space.category_count}]')

        self._values = values
        self._space = space

    def __repr__(self): return f"CategorizationMatrix({self._values})"

    def __str__(self): return f"{self._space} of {len(self)} instances"

    def __iter__(self):
        for value in self._values:
            yield Categorization(value, self._space)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        if isinstance(item, int):
            return Categorization(int(self._values[item]), self._space)
        elif isinstance(item, slice):
            return CategorizationMatrix(self._values[item], self._space)
        else:
            raise TypeError("Invalid index type. Must be int or slice.")

    @property
    def space(self) -> CategorizationSpace: return self._space

    def as_matrix(self) -> np.ndarray:
        return np.array([self.space.convert_to_flag(assignment) for assignment in self._values])

    def save(self, group: h5py.Group):
        projection_group = group.create_group('assignments')
        projection_group.create_dataset('values', data=self._values)
        self._space.save(projection_group.create_group('space'))

    @classmethod
    def load(cls, group: h5py.Group) -> 'CategorizationMatrix':
        values = group['assignments']['values'][:]
        space = CategorizationSpace.load(group['assignments']['space'])
        return cls(values, space)
