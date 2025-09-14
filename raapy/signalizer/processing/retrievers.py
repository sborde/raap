from typing import Optional

from signalizer.model.experiment.recordings import Recording
from signalizer.model.experiment.sequences import EventVector
from signalizer.model.experiment.signals import Signal
from signalizer.processing.factories import ProcessingSequence


class SignalRetriever:
    """SignalRetriever extracts a signal from a recording with possible subsequent preprocessing."""

    def __init__(self, signal_name: str, processing_sequence: Optional[ProcessingSequence[Signal]] = None):
        self._signal_name = signal_name
        self._processing_sequence = processing_sequence or ProcessingSequence[Signal]([])

    def __repr__(self):
        return f'SignalRetriever({self._signal_name}, {self._processing_sequence})'

    @property
    def name(self) -> str:
        if self._processing_sequence.name:
            return f'{self._signal_name}_{self._processing_sequence.name}'
        return f'{self._signal_name}'

    @property
    def details(self) -> str:
        string = f'- name: "{self._signal_name}"\n'
        if self._processing_sequence.name:
            string += f'- Processing sequence:\n'
            string += '\n'.join([f'\t{row}' for row in self._processing_sequence.details.split('\n')])
        return string

    @property
    def signal_name(self) -> str:
        return self._signal_name

    @property
    def params(self) -> dict:
        config: dict[str, str | dict] = {
            'signal_name': self._signal_name
        }

        processing_config = self._processing_sequence.params
        if processing_config:
            config['processing_sequence'] = processing_config

        return config

    @classmethod
    def from_config(cls, config: dict) -> 'SignalRetriever':
        signal_name = config['signal_name']

        processing_sequence = None
        if 'processing_sequence' in config:
            processing_sequence = ProcessingSequence.from_config(config['processing_sequence'])

        return cls(signal_name, processing_sequence)

    def extract(self, recording: Recording) -> Signal:
        signal = recording.get_waveform(self._signal_name)
        return self._processing_sequence.process(signal)


class EventRetriever:
    """EventRetriever extracts a trigger from a recording."""

    def __init__(self, event_name: str):
        self._event_name = event_name

    def __repr__(self):
        return f'EventRetriever({self._event_name})'

    @property
    def name(self) -> str:
        return f'{self._event_name}_related'

    @property
    def details(self) -> str:
        return f'- name: "{self._event_name}"'

    @property
    def event_name(self) -> str:
        return self._event_name

    @property
    def params(self) -> dict:
        config = {'event_name': self._event_name}
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'EventRetriever':
        event_name = config['event_name']
        return cls(event_name)

    def extract(self, recording: Recording) -> EventVector:
        return recording.get_event(self._event_name)
