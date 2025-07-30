import abc
from typing import Protocol, Optional, Sequence, Callable

from sklearn.decomposition import PCA

from sompy import SOMFactory
from signalizer.model.analysis.categorizations import SomCategorizationSpace
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.wrappers import InstanceMatrix
from signalizer.processing.categorizers import SomCategorizer, Categorizer, InstanceCategorizer
from signalizer.processing.projectors import PcaProjector, Projector, InstanceProjector, IdentityProjector


class Trainer[I: MatrixLike, O](Protocol):

    @abc.abstractmethod
    def train(self, input_data: I) -> O: ...


class ListTrainer[I: MatrixLike, O]:
    """Trains a trainer object on a list of matrix-like objects."""

    def __init__(self, trainer: Trainer[I, O]):
        self._trainer = trainer

    def train(self, data: list[I]) -> O:
        """Trains a trainer object on the concatenated matrix-like objects."""
        return self._trainer.train(data[0].concat(data))


class InstanceListProjectorTrainer[R: TimeReferenced, I: MatrixLike]:
    """Trains a projector on the given training instances."""

    def __init__(self, trainer: ListTrainer[I, Projector[I]]):
        self._trainer = trainer

    def train(self, instances: list[InstanceMatrix[R, I]]) -> InstanceProjector[R, I]:
        # Train the projector on the concatenated data
        projector = self._trainer.train([instance.data for instance in instances])
        return InstanceProjector[R, I](projector)


class InstanceListCategorizerTrainer[R: TimeReferenced, I: MatrixLike]:
    """Trains a projector on the given training instances."""

    def __init__(self, trainer: ListTrainer[I, Categorizer[I]]):
        self._trainer = trainer

    def train(self, instances: list[InstanceMatrix[R, I]]) -> InstanceCategorizer[R, I]:
        # Train the projector on the concatenated data
        categorizer = self._trainer.train([instance.data for instance in instances])
        return InstanceCategorizer[R, I](categorizer)


class IdentityProjectorTrainer[I: MatrixLike](Trainer[I, IdentityProjector[I]]):

    def train(self, input_data: I) -> IdentityProjector[I]:
        return IdentityProjector[I](input_data.as_matrix().shape[1])


class PcaProjectorTrainer[I: MatrixLike]:

    def __init__(self, selector_value: Optional[float] | Optional[int] = None):

        if selector_value is None:
            self._selection_mode = 'all'
        else:
            if isinstance(selector_value, int):
                self._selection_mode = f'top_{selector_value}'
            elif isinstance(selector_value, float):
                self._selection_mode = f'variance_{selector_value}'

        self._selector_value = selector_value

    def train(self, input_data: I) -> PcaProjector[I]:

        model = PCA(
            n_components=self._selector_value,
            svd_solver='full'
        )
        model.fit(input_data.as_matrix())

        return PcaProjector[I](model, self._selection_mode)


class SomCategorizerTrainer[I: MatrixLike]:

    def __init__(self, map_size: Sequence[int]):
        self._space = SomCategorizationSpace(map_size)

    def train(self, input_data: I) -> SomCategorizer[I]:
        model = SOMFactory.build(
            input_data.as_matrix(),
            mapsize=self._space.map_size,
            normalization=None,
            initialization='pca'
        )
        model.train(n_job=1, train_rough_len=4, train_finetune_len=16, verbose=None)

        return SomCategorizer[I](model)
