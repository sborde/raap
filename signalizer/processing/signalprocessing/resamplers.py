from signalizer.framework.protocols import Processor
from signalizer.model.experiment.signals import Signal


class Resampler(Processor[Signal]):
    """Processing class that resamples the input signal to the defined new sampling rate."""

    def __init__(self, target_fs: int):
        self._target_fs = target_fs

    def __repr__(self):
        return f'Resampler({self._target_fs})'

    @property
    def name(self):
        return f'{self._target_fs}Hz'

    @property
    def details(self) -> str:
        return f'- Resampler:\n\t- sampling rate: {self._target_fs}Hz'

    @property
    def params(self) -> dict:
        params = {
            'type': 'resampler',
            'params': {
                'target_fs': self._target_fs
            }
        }
        return params

    @classmethod
    def from_config(cls, config: dict) -> 'Resampler':
        return cls(config['params']['target_fs'])

    def process(self, signal: Signal) -> Signal:
        return signal.resampled(self._target_fs)
