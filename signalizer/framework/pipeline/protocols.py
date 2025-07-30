import abc
from pathlib import Path
from typing import Protocol, Iterable

from signalizer.model.wrappers import EntityData


class Pipeline[I, O](Protocol):
    """Interface for processing pipelines that processes a data wrapped in EntityData."""

    @abc.abstractmethod
    def run(self, data: I) -> O: ...


class ExtractorPipeline[I, O](Pipeline[I, O], Protocol):
    """Interface for pipelines that extract data from a data wrapped in EntityData."""

    @abc.abstractmethod
    def run_on_instances(self, data: I) -> O: ...


class SavablePipeline[I, O](Pipeline[I, O], Protocol):
    """Interface for pipelines that can be saved to a file."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def hash(self) -> str: ...

    @property
    @abc.abstractmethod
    def result_path(self) -> Path: ...

    @abc.abstractmethod
    def save(self, path: Path, name: str = None) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load[T](cls, path: Path) -> T: ...


class TrainingPipeline[I, O](Protocol):

    @property
    @abc.abstractmethod
    def pipeline(self) -> SavablePipeline[I, O]: ...

    @abc.abstractmethod
    def run(self, data: Iterable[EntityData[I]]) -> Iterable[EntityData[O]]: ...

    @abc.abstractmethod
    def with_instances(self, data: Iterable) -> Iterable[EntityData[O]]: ...


class PipelineBlock(Protocol):

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def details(self) -> str: ...

    @property
    @abc.abstractmethod
    def params(self) -> dict: ...

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load[T](cls, config: dict) -> T: ...


class EntityHandler[I](Protocol):
    """Interface for entity handlers that processes EntityData without return value"""

    @abc.abstractmethod
    def run(self, entity: EntityData[I]) -> None: ...
