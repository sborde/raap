



from scipy.signal import savgol_filter
from data.spikes import SpikeShape


class SavGolFilter:
    """
    A functor class for applying a Savitzky-Golay filter to a spike shape.

    It applies the filter with the given parameters to smooth the signal.
    """

    def __init__(self, window_length: int, polyorder: int):
        self._window_length = window_length
        self._polyorder = polyorder

    def __call__(self, spike: SpikeShape) -> SpikeShape:
        v = savgol_filter(spike.values, self._window_length, self._polyorder)
        return SpikeShape(
            spike.times,
            v,
            spike.timestamp,
            spike.dt
        )

