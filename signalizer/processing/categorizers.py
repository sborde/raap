import abc
import pickle as pkl
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from sompy.sompy import SOM
from signalizer.framework.pipeline.protocols import PipelineBlock
from signalizer.model.analysis.categorizations import CategorizationMatrix, SomCategorizationSpace
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.wrappers import InstanceMatrix


class Categorizer[M: MatrixLike](PipelineBlock, Protocol):
    """A protocol for objects that assign category to each sample in a matrix-like object."""

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int: ...

    @property
    @abc.abstractmethod
    def category_count(self) -> int: ...

    @abc.abstractmethod
    def assign(self, data: M) -> CategorizationMatrix: ...

    @classmethod
    def load(cls, config: dict) -> 'Categorizer':
        if config['type'] == 'SOM':
            return SomCategorizer.load(config)
        else:
            raise ValueError(f"Unknown categorizer type: {config['type']}")


class InstanceCategorizer[R: TimeReferenced, M: MatrixLike](PipelineBlock):
    """An adapter class that assigns category for a matrix-like object wrapped in an instance."""

    def __init__(self, categorizer: Categorizer[M]):
        self._categorizer = categorizer

    @property
    def name(self) -> str:
        return self._categorizer.name

    @property
    def details(self) -> str:
        string = '- InstanceCategorizer\n'
        string += self._categorizer.details
        return string

    @property
    def params(self) -> dict:
        return self._categorizer.params

    def save(self, path: Path) -> None:

        self._categorizer.save(path)

    @classmethod
    def load(cls, config: dict) -> 'InstanceCategorizer':
        return InstanceCategorizer(Categorizer.load(config))

    def assign(self, instances: InstanceMatrix[R, M]) -> InstanceMatrix[R, CategorizationMatrix]:
        """Projects the input data and returns a sequence of projections."""
        assignment = self._categorizer.assign(instances.data)
        return InstanceMatrix(instances.reference, assignment)


class SomCategorizer[M: MatrixLike](Categorizer[M]):
    """This class projects a segment collection to a PCA space."""

    def __init__(self, model: SOM):
        self._model = model

    def __repr__(self) -> str: return f'SomCategorizer({self._model})'

    @property
    def name(self) -> str:
        return f'som_{self.map_size[0]}x{self.map_size[1]}'

    @property
    def details(self) -> str:
        string = '- type: SOM\n'
        string += f'\t- map size: {self.map_size[0]}x{self.map_size[1]}\n'
        string += f'\t- map shape: {self.map_shape}\n'
        return string

    @property
    def params(self) -> dict:
        return {
            'type': 'SOM',
            'params': {
                'model': 'categorizer.pkl',
            }
        }

    def save(self, path: Path) -> None:
        """Saves the model to a file."""
        with open(path, 'wb') as file:
            pkl.dump(self._model, file)

    @classmethod
    def load(cls, config: dict) -> 'SomCategorizer':
        with open(config['params']['model'], 'rb') as f:
            model = pkl.load(f)
        return cls(model)

    @property
    def units(self) -> np.ndarray: return self._model.codebook.matrix

    @property
    def category_count(self) -> int: return self._model.codebook.nnodes

    @property
    def input_dimension(self) -> int: return self.units.shape[1]

    @property
    def map_size(self) -> list[int]: return self._model.codebook.mapsize

    @property
    def map_shape(self): return self._model.codebook.lattice

    def assign(self, data: M) -> CategorizationMatrix:
        bmu = self._model.find_bmu(data.as_matrix())[0].astype(int)
        space = SomCategorizationSpace(self.map_size)
        return CategorizationMatrix(bmu, space)
