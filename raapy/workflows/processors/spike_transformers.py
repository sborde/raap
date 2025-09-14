

import abc
from typing import Protocol, Optional

import numpy as np

from scipy.signal import savgol_filter

from workflows.common.models.spike import SpikeShape


class SpikeTransformer(Protocol):
    """A protocol for spike transformers."""

    @abc.abstractmethod
    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """Transform a spike shape and return a new one or None."""
        pass


class DummySpikeTransformer:
    """A dummy spike transformer that returns the spike unchanged."""
    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """Return the spike unchanged."""
        return spike


class SpikeZeroAligner:
    """A class for aligning spikes to zero based on configuration."""
    def __init__(self, config: dict):
        self._config = config

    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """Align the spike to zero."""
        return align_spike_to_zero(spike, self._config)

def create_transformer(
    name: str,
    config: Optional[dict]  = None
) -> SpikeTransformer:
    if name == 'align':
        if config is None:
            raise ValueError(
                'Configuration must be provided ' +
                'for \'align\' transformer.'
            )
        return SpikeZeroAligner(config)
    else:
        return DummySpikeTransformer()


def find_start_index(v: np.ndarray, params: dict) -> Optional[int]:
    """
    Find the start index of a spike.

    It first smooths the signal, then calculates the first derivative.
    We consider threshold dV value as the given fraction of the maximum
    derivative value before the maximum. If the signal never reaches this
    threshold, it returns None. Else, it returns the first index where the
    derivative is at or above the threshold.
    """

    if 'smooth' in params and params['smooth']:
        v = savgol_filter(v, **params['smooth'])

    peak_i = np.argmax(v)
    dv = np.diff(v[:peak_i])
    if len(dv) == 0:
        # if there is no derivative, return None
        return None
    max_dv_i = np.argmax(dv)
    dv_ratio = params['start_threshold']
    threshold_dv = dv_ratio * dv[max_dv_i]

    below_index = np.where(dv[:max_dv_i] < threshold_dv)[0]

    return below_index[-1] + 1 if len(below_index) > 0 else None


def align_spike_to_zero(spike: SpikeShape, params: dict) -> SpikeShape:
    """
    Align the spike to zero time.
    The spike is shifted so that the peak is at zero time.
    """
    spike_peak_index = np.argmin(np.abs(spike.times - 0)).astype(int)
    start_i = find_start_index(spike.values[:spike_peak_index + 1], params)

    # if the start cannot be detected, return None
    if start_i is None:
        raise ValueError(
            f'Spike {spike.timestamp} start can\'t be detected.'
        )

    start_v = spike.values[start_i]

    return SpikeShape(
        spike.times,
        spike.values - start_v,
        spike.timestamp,
        spike.dt
    )
