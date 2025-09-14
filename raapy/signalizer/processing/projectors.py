import abc
import pickle as pkl
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA

from signalizer.framework.pipeline.protocols import PipelineBlock
from signalizer.model.analysis.projections import ProjectionSpace, ProjectionMatrix
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.wrappers import InstanceMatrix


class Projector[M: MatrixLike](PipelineBlock, Protocol):

    @property
    @abc.abstractmethod
    def output_dimension(self) -> int: ...

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int: ...

    @abc.abstractmethod
    def project(self, data: M) -> ProjectionMatrix: ...

    @classmethod
    def load(cls, config: dict) -> 'Projector':
        if config['type'] == 'PCA':
            return PcaProjector.load(config)
        if config['type'] == 'Identity':
            return IdentityProjector.load(config)
        else:
            raise ValueError(f"Unknown projector type: {config['type']}")


class InstanceProjector[R: TimeReferenced, M: MatrixLike](PipelineBlock):
    """An adapter class that projects matrix-like data wrapped in an instance."""

    def __init__(self, projector: Projector[M]):
        self._projector = projector

    @property
    def name(self) -> str:
        return self._projector.name

    @property
    def details(self) -> str:
        string = '- InstanceProjector\n'
        return string + self._projector.details

    @property
    def params(self) -> dict:
        return self._projector.params

    def save(self, path: Path) -> None:
        self._projector.save(path)

    @classmethod
    def load(cls, config: dict) -> 'InstanceProjector':
        return cls(Projector.load(config))

    def project(self, instances: InstanceMatrix[R, M]) -> InstanceMatrix[R, ProjectionMatrix]:
        """Projects the input data and returns a sequence of projections."""
        projection = self._projector.project(instances.data)
        return InstanceMatrix(instances.reference, projection)


class IdentityProjector[M: MatrixLike](PipelineBlock):
    """This class projects a segment collection to a vector space."""

    def __init__(self, dimension: int):
        self._dimension = dimension

    def __repr__(self) -> str: return f'IdentityProjector({self._dimension})'

    @property
    def name(self) -> str:
        return f'identity'

    @property
    def details(self) -> str:
        string = '- type: Identity\n'
        return string

    @property
    def params(self) -> dict:
        return {
            'type': 'Identity',
            'params': {
                'dimension': self._dimension
            }
        }

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, config: dict) -> 'IdentityProjector[M]':
        dimension = config['params']['dimension']
        return cls(dimension)

    @property
    def output_dimension(self) -> int: return self._dimension

    @property
    def input_dimension(self) -> int: return self._dimension

    def project(self, data: M) -> ProjectionMatrix:
        space = ProjectionSpace('Identity', self._dimension, self._dimension)
        return ProjectionMatrix(data.as_matrix(), space)


class PcaProjector[M: MatrixLike](Projector[M]):
    """This class projects a segment collection to a PCA space."""

    def __init__(self, model: PCA, selection: str):
        self._model = model
        self._selection = selection

    def __repr__(self) -> str: return f'PcaProjector({self._model})'

    @property
    def name(self) -> str:
        return f'pca_{self._selection}'

    @property
    def details(self) -> str:
        string = '- type: PCA\n'
        string += f'\t- selection: {self._selection}\n'
        string += f'\t- input dimension: {self.input_dimension}\n'
        string += f'\t- components: {self.output_dimension}\n'
        return string

    @property
    def params(self) -> dict:
        return {
            'type': 'PCA',
            'params': {
                'model': 'projector.pkl',
                'selection': self._selection
            }
        }

    def save(self, path: Path) -> None:
        with open(path, 'wb') as file:
            pkl.dump(self._model, file)

    @classmethod
    def load(cls, config: dict) -> 'PcaProjector[M]':
        model_path = config['params']['model']
        selection = config['params']['selection']
        with open(model_path, 'rb') as file:
            model = pkl.load(file)
        return cls(model, selection)

    @property
    def means(self): return self._model.mean_

    @property
    def coefficients(self) -> np.ndarray: return self._model.components_

    @property
    def input_dimension(self) -> int: return self._model.n_features_in_

    @property
    def output_dimension(self) -> int: return self._model.components_.shape[0]

    @property
    def explained_variance(self) -> float: return self._model.explained_variance_ratio_

    def project(self, data: M) -> ProjectionMatrix:
        space = ProjectionSpace('PCA', self.input_dimension, self.output_dimension, self._selection)
        return ProjectionMatrix(self._model.transform(data.as_matrix()), space)
