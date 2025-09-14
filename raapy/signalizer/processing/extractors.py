import abc
from pathlib import Path
from typing import Protocol

import yaml

from signalizer.framework.pipeline.protocols import PipelineBlock
from signalizer.model.experiment.recordings import Recording
from signalizer.model.experiment.segments import SegmentMatrix, MatrixLike
from signalizer.model.experiment.sequences import EventVector, TimeReferencedVector
from signalizer.model.wrappers import InstanceMatrix
from signalizer.processing.retrievers import SignalRetriever, EventRetriever
from signalizer.processing.segment_cutters import EventRelatedSegmentCutter


class InstanceExtractor[R: TimeReferencedVector, M: MatrixLike](PipelineBlock, Protocol):
    """Interface for extracting instances from a recording"""

    @abc.abstractmethod
    def extract(self, recording: Recording) -> InstanceMatrix[R, M]: ...

    @classmethod
    def load(cls, config: dict) -> 'InstanceExtractor':
        if config['type'] == 'event_related_segment_extractor':
            return EventRelatedSegmentExtractor.load(config)
        else:
            raise ValueError(f"Unknown extractor type: {config['type']}")


class EventRelatedSegmentExtractor(InstanceExtractor[EventVector, SegmentMatrix]):
    """EventRelatedSegmentExtractor extracts signal segments relative to events."""

    def __init__(self,
                 signal_retriever: SignalRetriever,
                 event_retriever: EventRetriever,
                 segment_cutter: EventRelatedSegmentCutter):

        self._signal_retriever = signal_retriever
        self._event_retriever = event_retriever
        self._segment_cutter = segment_cutter

    @property
    def name(self) -> str:
        return ' '.join([
            self._event_retriever.name,
            self._signal_retriever.name,
            self._segment_cutter.name
        ])

    @property
    def details(self) -> str:
        string = ''
        for name, retriever in [
            ('Signal', self._signal_retriever),
            ('Event', self._event_retriever),
            ('Segment cutter', self._segment_cutter)
        ]:
            string += f'- {name}\n' + '\n'.join([f'\t{row}' for row in retriever.details.split('\n')]) + '\n'
        return string

    @property
    def params(self) -> dict:
        return {
            'signal_retriever': self._signal_retriever.params,
            'event_retriever': self._event_retriever.params,
            'segment_cutter': self._segment_cutter.params,
            'type': 'event_related_segment_extractor'
        }

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, config: dict) -> 'EventRelatedSegmentExtractor':
        signal_extractor = SignalRetriever.from_config(config['signal_retriever'])
        event_extractor = EventRetriever.from_config(config['event_retriever'])
        segmenter = EventRelatedSegmentCutter.from_config(config['segment_cutter'])

        return cls(signal_extractor, event_extractor, segmenter)

    def extract(self, recording: Recording) -> InstanceMatrix[EventVector, SegmentMatrix]:
        signal = self._signal_retriever.extract(recording)
        events = self._event_retriever.extract(recording)
        return self._segment_cutter.cut(signal, events)
