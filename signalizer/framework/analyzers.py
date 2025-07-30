import abc
from pathlib import Path
from typing import Iterable, Protocol

from signalizer.framework.pipeline.protocols import Pipeline, EntityHandler, TrainingPipeline
from signalizer.model.wrappers import EntityData, Entity


class IndividualAnalyzer[I, O](Protocol):
    """Interface for individual entity processing pipelines"""

    @abc.abstractmethod
    def run(self, entities: Iterable[EntityData[I]]) -> Iterable[EntityData[O]]: ...

    @abc.abstractmethod
    def add_handlers(self, handlers: list[EntityHandler[O]]) -> 'IndividualAnalyzer[I, O]': ...


class IndividualProcessingAnalysis[I, O]:
    """IndividualAnalysis applies a pipeline on each entity individually, then performs
    the additional post-processing steps defined by a list of EntityHandlers."""

    def __init__(self, pipeline: Pipeline[I, O], handlers: list[EntityHandler[O]] = None):
        self._pipeline = pipeline
        self._handlers = handlers or []

    def run(self, entities: Iterable[EntityData[I]]) -> Iterable[EntityData[O]]:
        return [self._run_one(entity) for entity in entities]

    def add_handlers(self, handlers: list[EntityHandler[O]]) -> 'IndividualAnalyzer[I, O]':
        self._handlers.extend(handlers)
        return self

    def _run_one(self, entity: EntityData[I]) -> EntityData[O]:
        result = entity.with_data(self._pipeline.run(entity.data))
        for handler in self._handlers:
            handler.run(result)
        return result


class TrainableIndividualAnalysis[I, O]:
    """TrainableIndividualAnalysis runs a training pipeline on a list of entities,
    then performs the additional post-processing steps defined by a list of EntityHandlers."""

    def __init__(self, pipeline: TrainingPipeline[I, O], handlers: list[EntityHandler[O]] = None):
        self._pipeline = pipeline
        self._handlers = handlers or []

    def save_pipeline(self, path: Path) -> None:
        """Saves the pipeline to the given path."""
        if not path.exists():
            path.mkdir(parents=True)
        if not path.is_dir():
            raise ValueError("The provided path is not a directory.")

        trained_pipeline = self._pipeline.pipeline
        save_path = path / trained_pipeline.result_path
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        self._pipeline.pipeline.save(save_path)

    def add_handlers(self, handlers: list[EntityHandler[O]]) -> 'TrainableIndividualAnalysis[I, O]':
        self._handlers.extend(handlers)
        return self

    def run(self, entities: Iterable[EntityData[I]]) -> Iterable[EntityData[O]]:
        # run training and processing
        return self._pipeline.run(entities)

    def with_instances(self, entities: Iterable[EntityData[I]]) -> Iterable[EntityData[O]]:
        """Sets the instances for the training pipeline."""
        return self._pipeline.with_instances(entities)
