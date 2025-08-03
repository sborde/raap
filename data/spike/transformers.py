

import abc
import numpy as np

from typing import Protocol

from data.spike.detectors import SpikeStartFinder
from data.spikes import SpikeShape


class SpikeTransformer(Protocol):
    """A protocol for spike transformers."""

    @abc.abstractmethod
    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """Transform a spike shape and return a new one or None."""


class DummySpikeTransformer:
    """A dummy spike transformer that returns the spike unchanged."""
    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """Return the spike unchanged."""
        return spike


class SpikeStartAligner:
    """A class for aligning spikes to zero based on configuration."""
    def __init__(self, start_finder: SpikeStartFinder):
        self._find_start = start_finder

    def __call__(self, spike: SpikeShape) -> SpikeShape:
        """
        Align the spike to zero time.
        The spike is shifted so that the peak is at zero time.
        """
        start_i = self._find_start(spike)

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
