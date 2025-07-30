from typing import Iterable

from signalizer.framework.protocols import Processor
from signalizer.model.experiment.signals import Signal
from signalizer.processing.segment_cutters import EventRelatedSegmentCutter
from signalizer.processing.signalprocessing.filters import Filter
from signalizer.processing.signalprocessing.resamplers import Resampler


class ProcessorFactory:

    @staticmethod
    def create[P: Processor](config: dict) -> P:
        if config['type'] == 'resampler':
            return Resampler.from_config(config)
        else:
            raise ValueError(f"Unknown processor type: {config['type']}")


class ProcessingSequence[T](Processor[T]):
    """Class that represents a sequence of processing steps. It applies each processor in the order they were added."""

    def __init__(self, processors: Iterable[Processor[T]]):
        self._processors = processors

    @property
    def name(self) -> str:
        return '_'.join([processor.name for processor in self._processors]) if self._processors else ''

    @property
    def details(self) -> str:
        return '\n'.join([processor.details for processor in self._processors]) if self._processors else ''

    @property
    def params(self) -> dict:
        return {'processors': [processor.params for processor in self._processors]} if self._processors else {}

    @classmethod
    def from_config(cls, config: dict) -> 'ProcessingSequence':
        processors = [ProcessorFactory.create(config) for config in config['processors']]
        return cls(processors)

    def process(self, item: T) -> T:
        for processor in self._processors:
            item = processor.process(item)
        return item


class SignalProcessingSequenceBuilder:
    """Builder class for creating a SignalProcessingSequence.
    It allows adding different types of processors to the sequence."""
    def __init__(self):
        self._processors = []

    def with_resampler(self, param: dict):
        self._processors.append(Resampler(**param))
        return self

    def with_filter(self, param: dict):
        self._processors.append(Filter.create(param))
        return self

    def build(self) -> ProcessingSequence[Signal]:
        return ProcessingSequence[Signal](self._processors)


class EventRelatedSegmentCutterFactory:
    """Factory for creating EventRelatedSegmentCutter instances based on a configuration dictionary."""
    @staticmethod
    def create(config: dict) -> EventRelatedSegmentCutter:
        return EventRelatedSegmentCutter.from_config(config)


