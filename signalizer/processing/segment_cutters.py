from pathlib import Path

import numpy as np
import yaml

from signalizer.framework.pipeline.protocols import PipelineBlock
from signalizer.framework.protocols import SavableOperator
from signalizer.model.experiment.segments import SegmentMatrix
from signalizer.model.experiment.sequences import EventVector
from signalizer.model.experiment.signals import Signal, SignalMatrix
from signalizer.model.general.data import Interval
from signalizer.model.measurements import TimeUnit
from signalizer.model.temporal.data import TimeInterval
from signalizer.model.wrappers import InstanceMatrix


class EventRelatedSegmentCutter(SavableOperator):

    def __init__(self, window: TimeInterval):
        self._window = window

    def __repr__(self):
        return f'EventRelatedSegmentCutter({self._window})'

    @property
    def name(self) -> str:
        return f'window_{self._window.start}_{self._window.end}_{self._window.unit}'

    @property
    def details(self) -> str:
        return f'- cut window: {self._window}'

    @property
    def params(self) -> dict:
        config = {
            'cut_window': {
                'start': self._window.start,
                'end': self._window.end
            },
            'unit': str(self._window.unit)
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'EventRelatedSegmentCutter':
        interval = TimeInterval(
            Interval(config['cut_window']['start'], config['cut_window']['end']),
            TimeUnit.from_abbreviation(config['unit'])
        )
        return cls(interval)

    def cut(self, signal: Signal, events: EventVector) -> InstanceMatrix[EventVector, SegmentMatrix]:

        event_segment_pairs = []
        for event in events:
            if signal.is_valid_trigger(event.timestamp, self._window):
                event_segment_pairs.append((event, signal.triggered_portion(event.timestamp, self._window).values))

        valid_events = [event for event, _ in event_segment_pairs]
        valid_portions = [portion for _, portion in event_segment_pairs]

        segments = SegmentMatrix(
            SignalMatrix(
                self._generate_time(signal.si),
                np.stack(valid_portions, axis=1),
                signal.si, signal.time_unit, signal.value_unit
            )
        )

        return InstanceMatrix[EventVector, SegmentMatrix](EventVector.from_list(valid_events), segments)

    def _generate_time(self, si: float) -> np.ndarray:
        sample_count = int((self._window / si).length)
        return np.linspace(self._window.start, self._window.end, sample_count + 1)
