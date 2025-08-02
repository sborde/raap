

import abc
import numpy as np
from typing import Protocol, Optional
from analyses.slowdown_descriptors import find_start_index
from data.spikes import SpikeShape

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