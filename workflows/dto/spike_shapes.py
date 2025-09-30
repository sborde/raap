


from typing import Optional
from data.spikes import SpikeShape


class PreparedSpikeShape:
    """A prepared spike shape with additional metadata."""
    def __init__(
        self,
        spike: SpikeShape,
        start: int,
        peak: int,
        noise_properties: Optional[dict[str, float]] = None
    ):
        self._spike = spike
        self._start = start
        self._peak = peak
        self.halfwidth_left: Optional[tuple[float, float]] = None
        self.halfwidth_right: Optional[tuple[float, float]] = None
        self._noise_properties = noise_properties or {}

    @property
    def spike(self) -> SpikeShape:
        """The original spike shape."""
        return self._spike

    @property
    def peak(self) -> int:
        """The peak point of the spike shape."""
        return self._peak

    @property
    def start(self) -> int:
        """The start point of the spike shape."""
        return self._start

    @property
    def noise_properties(self) -> dict[str, float]:
        """The noise properties of the spike shape."""
        return self._noise_properties
