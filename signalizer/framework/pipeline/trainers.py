from datetime import datetime
from typing import Iterable

from signalizer.framework.pipeline.processors import RecordingInstanceProjectorCategorizerPipeline
from signalizer.framework.pipeline.protocols import SavablePipeline
from signalizer.model.analysis.categorizations import CategorizationMatrix
from signalizer.model.experiment.recordings import Recording
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.experiment.sequences import TimeReferencedVector
from signalizer.model.wrappers import InstanceMatrix, EntityData
from signalizer.processing.extractors import InstanceExtractor
from signalizer.processing.trainers import InstanceListProjectorTrainer, \
    InstanceListCategorizerTrainer


class ProjectedInstanceCategorizationTrainer[R: TimeReferencedVector, M: MatrixLike]:

    def __init__(self,
                 projector_trainer: InstanceListProjectorTrainer[R, M],
                 categorizer_trainer: InstanceListCategorizerTrainer[R, M]):

        self._projector_trainer = projector_trainer
        self._categorizer_trainer = categorizer_trainer

        self._extractor = None
        self._projector = None
        self._categorizer = None

        self._training_time = None
        self._training_entities = []

    def with_extractor(self, extractor: InstanceExtractor[R, M]) -> 'ProjectedInstanceCategorizationTrainer':
        """Sets the extractor for the training."""
        self._extractor = extractor
        return self

    def run(self, data: Iterable[EntityData[Recording]]) -> \
            Iterable[EntityData[InstanceMatrix[R, CategorizationMatrix]]]:
        """Runs the training pipeline from the beginning."""

        if self._extractor is None:
            raise ValueError("Extractor not set. Call 'with_extractor' method first.")

        self._training_entities = [entity.entity for entity in data]
        instances = [entity.with_data(self._extractor.extract(entity.data)) for entity in data]

        return self.with_instances(instances)

    def with_instances(self, entities: Iterable[EntityData[InstanceMatrix[R, M]]]) -> \
            Iterable[EntityData[InstanceMatrix[R, CategorizationMatrix]]]:
        """Runs the training pipeline with the provided instances."""

        if not self._training_entities:
            self._training_entities = [entity.entity for entity in entities]

        data = [entity.data for entity in entities]

        self._projector = self._projector_trainer.train(data)
        projections = [self._projector.project(instance) for instance in data]

        self._categorizer = self._categorizer_trainer.train(projections)
        categorizations = [self._categorizer.assign(instance) for instance in projections]

        self._training_time = datetime.now().strftime('%Y-%m-%d %H:%M')

        return [
            EntityData(entity, categorization)
            for entity, categorization
            in zip(self._training_entities, categorizations)
        ]

    @property
    def pipeline(self) -> SavablePipeline[Recording, InstanceMatrix[R, CategorizationMatrix]]:
        """Returns the trained pipeline."""

        if not self._training_time:
            raise ValueError("Pipeline not trained yet. Call 'run' method first.")

        return RecordingInstanceProjectorCategorizerPipeline(
            self._extractor,
            self._projector,
            self._categorizer,
            self._training_time,
            self._training_entities
        )
