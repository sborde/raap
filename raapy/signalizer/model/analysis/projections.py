from typing import Optional

import h5py
import numpy as np

from signalizer.model.experiment.segments import MatrixLike


class ProjectionSpace:

    def __init__(self, name: str, input_dimension: int, output_dimension: int, label: Optional[str] = None):
        self._name = name
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._label = label

    def __repr__(self) -> str:
        return f'ProjectionSpace({self._name}, {self._input_dimension}, {self._output_dimension}, {self._label})'

    def __str__(self) -> str:
        string = f'{self._input_dimension}D->{self._output_dimension}D {self._name} projection '
        if self._label:
            string += f' ({self._label})'
        return string

    @property
    def output_dimension(self) -> int: return self._output_dimension

    @property
    def input_dimension(self) -> int: return self._input_dimension

    @property
    def component_names(self) -> list[str]:
        return [f'{self._name}{i + 1}' for i in range(self._output_dimension)]

    def save(self, group: h5py.Group):
        group.create_dataset('name', data=self._name)
        group.create_dataset('input_dimension', data=self._input_dimension)
        group.create_dataset('output_dimension', data=self._output_dimension)
        group.create_dataset('label', data=self._label)

    @classmethod
    def load(cls, group: h5py.Group) -> 'ProjectionSpace':
        name = group['name'][()].decode('utf-8')
        label = group['label'][()].decode('utf-8')
        output_dimension = group['dimension'][()]
        input_dimension = group['input_dimension'][()]
        return cls(name, input_dimension, output_dimension, label)


class Projection:
    """Class that represents the projection of an instance onto a lower-dimensional space."""

    def __init__(self, value: np.ndarray, space: ProjectionSpace):

        value = value.squeeze()

        if value.ndim != 1:
            raise ValueError('Projection must be a 1D vector.')

        if value.size != space.output_dimension:
            raise ValueError('The dimension of the projected value must be equal to the projection dimension.')

        self._value = value
        self._space = space

    def __repr__(self) -> str: return f"Projection({self._value})"

    def __str__(self) -> str: return str(self._space)

    @property
    def value(self) -> np.ndarray: return self._value

    @property
    def space(self) -> ProjectionSpace: return self._space

    def as_matrix(self) -> np.ndarray:
        """Returns the projection as a row vector of a matrix."""
        return self._value.reshape(1, -1)


class ProjectionMatrix(MatrixLike):
    """Class that represents the projection of multiple instances onto a lower-dimensional space."""

    def __init__(self, values: np.ndarray, space: ProjectionSpace):

        if values.shape[1] != space.output_dimension:
            raise ValueError('The dimension of the projected values must be equal to the projection dimension.')

        self._values = values
        self._space = space

    def __repr__(self) -> str:
        return f"ProjectionMatrix({self._values}, {self._space})"

    def __str__(self) -> str:
        return f'{self._space} of {len(self)} instances'

    def __len__(self) -> int: return self._values.shape[0]

    def __iter__(self):
        for projection in self._values:
            yield Projection(projection, self._space)

    def __getitem__(self, item):
        if isinstance(item, int):
            return Projection(self._values[item, :], self._space)
        elif isinstance(item, slice):
            return ProjectionMatrix(self._values[item, :], self._space)
        else:
            raise TypeError("Invalid index type. Must be int or slice.")

    @property
    def space(self) -> ProjectionSpace: return self._space

    def as_matrix(self) -> np.ndarray: return self._values

    def save(self, group: h5py.Group):
        projection_group = group.create_group('projections')
        projection_group.create_dataset('values', data=self._values)
        self._space.save(projection_group.create_group('space'))

    @classmethod
    def concat(cls, segment_matrices: list['ProjectionMatrix']) -> 'ProjectionMatrix':
        matrices = [segment_matrix.as_matrix() for segment_matrix in segment_matrices]
        values = np.concatenate(matrices, axis=0)
        space = segment_matrices[0].space
        return cls(values, space)

    @classmethod
    def load(cls, group: h5py.Group) -> 'ProjectionMatrix':
        values = group['projections']['values'][:]
        space = ProjectionSpace.load(group['projections']['space'])
        return cls(values, space)
