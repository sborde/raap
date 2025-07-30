

import abc
from typing import Protocol, Optional
from data.spikes import SpikeShape

from analyses.slowdown_descriptors import align_spike_to_zero


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