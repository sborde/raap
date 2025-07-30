import abc
from typing import Iterable

import numpy as np
import scipy.signal
from scipy.signal import butter

from signalizer.model.experiment.signals import Signal


class Filter(abc.ABC):
    """Abstract class for filtering signals. It defines the interface for all signal filters."""

    def __call__(self, signal: Signal) -> Signal:
        """Runs the filter on the given signal."""
        return self.run(signal)

    def run(self, signal: Signal) -> Signal:
        """Returns the filtered version of the given input time series"""

        cutoff = self._calculate_cutoff(signal.fs)
        ba = self._calculate_coefficient(cutoff)

        filtered_voltage = scipy.signal.filtfilt(ba[0], ba[1], np.nan_to_num(signal.values), axis=0)
        return Signal(signal.times, filtered_voltage, signal.si)

    @abc.abstractmethod
    def _calculate_cutoff(self, sampling_rate: float) -> np.ndarray: pass

    @abc.abstractmethod
    def _calculate_coefficient(self, cutoff: np.ndarray): pass

    @classmethod
    def create(cls, bands: float | Iterable[float], mode: str = None, order: int = 2, filter_type: str = None):

        filter_type = filter_type.lower() if filter_type else 'butterworth'
        mode = mode.lower() if mode else 'bandpass'

        if filter_type == 'butterworth':
            return ButterworthFilter(np.asarray(bands), order, mode)
        else:
            raise NotImplemented('Unknown filter type.')


class ButterworthFilter(Filter):

    def __init__(self, bands: np.ndarray, order: int, mode: str) -> None:
        self._bands = bands
        self._mode = mode
        self._order = order

    def __repr__(self) -> str:
        return f'ButterworthFilter({self._bands}, {self._order}, {self._mode})'

    def _calculate_cutoff(self, sampling_rate: float) -> np.ndarray:
        return self._bands / (sampling_rate / 2)

    def _calculate_coefficient(self, cutoff: np.ndarray):
        return butter(self._order, cutoff, btype=self._mode, output='ba')
