"""Functions to calculate rising phase features of individual spikes."""

from typing import Optional, Self
import numpy as np

from scipy.signal import savgol_filter, find_peaks

from workflows.common.models.spike import PlottableSpike, SpikeDerivative, SpikePhase, SpikeShape
from workflows.processors.spike_transformers import find_start_index


class SignalInterval:
    """
    A class to hold the start and end indices of a signal interval
    alongside with the sampling interval (dt).
    """

    def __init__(self, left_i: int, right_i: int, dt: float):
        self._left_i = int(left_i)
        self._right_i = int(right_i)
        self._dt = float(dt)

    @property
    def left_i(self) -> int:
        """The left index of the signal interval."""
        return self._left_i

    @property
    def right_i(self) -> int:
        """The right index of the signal interval."""
        return self._right_i

    @property
    def dt(self) -> float:
        """The sampling interval (dt) of the signal."""
        return self._dt

    @property
    def duration(self) -> float:
        """The duration of the signal interval."""
        return (self._right_i - self._left_i) * self._dt

    def is_within(self, index: int) -> bool:
        """Checks if the given index is within the signal interval."""
        return self._left_i <= index <= self._right_i

    def as_list(self) -> list[int]:
        return [self._left_i, self._right_i]


class NoiseProperties:
    """A class to hold the noise properties of a spike shape."""

    def __init__(
        self,
        raw: float = np.nan,
        smoothed: float = np.nan,
        noise: float = np.nan
    ):
        self._raw = raw
        self._smoothed = smoothed
        self._noise = noise

    @property
    def raw(self) -> float:
        """The power of the raw signal."""
        return self._raw

    @property
    def smoothed(self) -> float:
        """The power of the smoothed signal."""
        return self._smoothed

    @property
    def noise(self) -> float:
        """The power of the noise signal."""
        return self._noise

    def as_dict(self) -> dict:
        return {
            'raw_power': self._raw,
            'smoothed_power': self._smoothed,
            'noise_power': self._noise
        }


class LocalPeakFeatures:
    """A class to hold the features of a local peak in the spike derivative."""
    def __init__(self):
        self.dv_drop: float = np.nan
        self.rebound_time: float = np.nan
        self.relative_dv_drop: float = np.nan
        self.relative_rebound_time: float = np.nan

        self.left_peak: Optional[int] = None
        self.right_peak: Optional[int] = None
        self.dv_min_peak: Optional[float] = None

        self.valley_i: Optional[int] = None
        self.rebound_left_i: Optional[int] = None
        self.rebound_right_i: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            'dv_drop': self.dv_drop,
            'rebound_time': self.rebound_time,
            'relative_dv_drop': self.relative_dv_drop,
            'relative_rebound_time': self.relative_rebound_time,
            'left_peak': self.left_peak,
            'right_peak': self.right_peak,
            'dv_min_peak': self.dv_min_peak,
            'valley_i': self.valley_i,
            'rebound_i': [self.rebound_left_i, self.rebound_right_i]
        }


class DescribedSpikeShape(PlottableSpike):
    """A class to hold the features of a spike shape."""

    def __init__(
        self,
        spike: SpikeShape,
        start_i: int,
        search_interval: SignalInterval,
        noise_properties: Optional[NoiseProperties] = None,
        slowdown_features: Optional[LocalPeakFeatures] = None
    ):
        self._spike = spike

        self._start_i = start_i
        self._peak_i = spike.peak_i

        self._search_interval = search_interval
        self.noise_properties = noise_properties or NoiseProperties()
        self._slowdown_features = slowdown_features
        self.category: Optional[str] = None

    def __getattr__(self, name: str):
        if name in self.as_dict():
            return self.as_dict()[name]
        raise AttributeError(f"'DescribedSpikeShape' object has no attribute '{name}'")

    @property
    def spike(self) -> SpikeShape:
        """The spike shape."""
        return self._spike

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the spike shape.
        """
        return self._spike.times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the spike shape.
        """
        return self._spike.values

    @property
    def timestamp(self) -> float:
        """
        Get the timestamp of the spike shape.
        """
        return float(self._spike.timestamp)

    @property
    def dt(self) -> float:
        """
        Get the time step of the spike shape.
        """
        return self._spike.dt

    @property
    def derivative(self) -> SpikeDerivative:
        """The derivative of the spike shape."""
        return self._spike.get_derivative()

    @property
    def spike_phase(self) -> SpikePhase:
        """The phase of the spike shape."""
        return self._spike.get_phase()

    @property
    def start_i(self) -> int:
        """The start index of the spike rising phase."""
        return self._start_i

    @property
    def peak_i(self) -> int:
        """The time of the spike peak."""
        return self._peak_i

    @property
    def search_interval(self) -> SignalInterval:
        """The search interval for the rising phase features."""
        return self._search_interval

    @property
    def slowdown_features(self) -> LocalPeakFeatures:
        """The features of the local peak in the spike derivative."""
        if self._slowdown_features is None:
            raise ValueError('Slowdown features have not been calculated.')
        return self._slowdown_features

    @property
    def valley_time(self) -> float:
        """The time of the valley between the peaks."""
        if self.slowdown_features is None or self.slowdown_features.valley_i is None:
            return np.nan
        return float(self._spike.times[self.slowdown_features.valley_i])

    def get_phase(self) -> SpikePhase:
        """Get the phase of the spike shape."""
        return self._spike.get_phase()

    def with_slowdown_features(self, slowdown_features: LocalPeakFeatures) -> Self:
        """Returns a copy of the spike shape with the calculated slowdown features."""
        return type(self)(
            self._spike,
            self._start_i,
            self._search_interval,
            self.noise_properties,
            slowdown_features
        )

    def as_dict(self) -> dict:
        """Convert the features to a dictionary."""
        return {
            'timestamp': self.timestamp,
            'raw_power': self.noise_properties.raw,
            'noise_power': self.noise_properties.noise,
            'start_i': self.start_i,
            'peak_i': self.peak_i,
            'search_interval': self._search_interval.as_list(),
            **(self.slowdown_features.as_dict() if self.slowdown_features else {}),
            'valley_time': self.valley_time
        }

def find_search_interval(
    spike: SpikeShape,
    start_i: int,
    params: dict
) -> SignalInterval:
    """
    Find the search interval for the rising phase features of a spike.
    """
    search_start_i = find_search_start(spike, start_i, params['start'])
    search_end_i = find_search_end(spike, start_i, params['end'])
    return SignalInterval(search_start_i, search_end_i, spike.dt)

def calculate_rising_features(spike: SpikeShape, params: dict) -> Optional[DescribedSpikeShape]:
    """
    Calculate the features of the spike rising phase.
    """
    start_i = find_start_index(spike.values[:spike.peak_i + 1], params)

    # if the start cannot be detected, return None
    if start_i is None:
        return None

    if 'smooth' in params and params['smooth']:
        smoothed = smooth_spike(spike, params['smooth'])
        noise_properties = calculate_noise_properties(spike, smoothed)
        spike = smoothed

    search_interval = find_search_interval(spike, start_i, params['search_interval'])
    features = DescribedSpikeShape(
        spike,
        int(start_i),
        search_interval,
        noise_properties
    )

    # Calculate the rising phase duration
    dt = np.mean(np.diff(spike.times))

    peaks = find_dv_peaks(spike, search_interval)

    # if there are no secondary peaks, return default values
    if len(peaks) < 2:
        slowdown_features = LocalPeakFeatures()
        slowdown_features.relative_dv_drop = 0
        slowdown_features.relative_rebound_time = 0
        return features.with_slowdown_features(slowdown_features)

    v = spike.values
    peak_i = spike.peak_i

    derivative = spike.get_derivative()
    local_features: list[LocalPeakFeatures] = []
    global_peak = np.argmax(derivative.dv[:peak_i]).astype(int)
    for local_peak in peaks:
        if local_peak == global_peak:
            continue

        left_peak, right_peak = tuple(sorted([local_peak, global_peak]))
        dv_interpeak = derivative.dv[left_peak:right_peak + 1]
        dv_min_peak = min(dv_interpeak[0], dv_interpeak[-1])

        # Find the valley between the peaks
        dv_valley_i = np.argmin(dv_interpeak).astype(int)
        dv_valley = dv_interpeak[dv_valley_i]

        # Calculate the relative drop in dv
        dv_drop = dv_min_peak - dv_valley

        # find the first point left of the valley where dv reaches the min peak
        left_level_i = np.where(dv_interpeak <= dv_min_peak)[0][0]
        # find the first point right of the valley where dv reaches the min peak
        right_level_i = np.where(dv_interpeak <= dv_min_peak)[0][-1]

        # calculate the real rebound time
        rebound_time = (right_level_i - left_level_i) * dt
        if dv_interpeak[left_level_i] < dv_min_peak:
            alpha = (
                (dv_min_peak - dv_interpeak[left_level_i]) /
                (dv_interpeak[left_level_i-1] - dv_interpeak[left_level_i])
            )
            rebound_time += alpha * dt
        if dv_interpeak[right_level_i] < dv_min_peak:
            alpha = (
                (dv_min_peak - dv_interpeak[right_level_i]) /
                (dv_interpeak[right_level_i+1] - dv_interpeak[right_level_i])
            )
            rebound_time += (1 - alpha) * dt

        slowdown_features = LocalPeakFeatures()
        slowdown_features.dv_drop = float(dv_drop)
        slowdown_features.rebound_time = float(rebound_time)
        slowdown_features.relative_dv_drop = float(dv_drop / derivative.dv[global_peak])
        slowdown_features.relative_rebound_time = float(rebound_time / search_interval.duration)
        slowdown_features.dv_min_peak = float(dv_min_peak)
        slowdown_features.left_peak = int(left_peak)
        slowdown_features.right_peak = int(right_peak)
        slowdown_features.valley_i = int(dv_valley_i + left_peak)
        slowdown_features.rebound_left_i = int(left_level_i + left_peak)
        slowdown_features.rebound_right_i = int(right_level_i + left_peak)
        local_features.append(slowdown_features)

    selected_peak_features = sorted(local_features, key=lambda x: x.dv_drop)[-1]
    return features.with_slowdown_features(selected_peak_features)


def smooth_spike(spike: SpikeShape, params: dict) -> SpikeShape:
    """
    Smooth the spike shape using a Savitzky-Golay filter.
    """
    smoothed_values = savgol_filter(spike.values, **params)
    return SpikeShape(spike.times, smoothed_values, spike.timestamp, spike.dt)

def calculate_noise_properties(
    raw: SpikeShape,
    smoothed: SpikeShape
) -> NoiseProperties:
    """
    Calculate the noise properties of a spike shape.
    """
    return NoiseProperties(
        calculate_signal_power(raw.values),
        calculate_signal_power(smoothed.values),
        calculate_signal_power(raw.values - smoothed.values)
    )

def calculate_signal_power(signal: np.ndarray) -> float:
    """
    Calculate the power of a signal.
    """
    return float(np.mean(signal ** 2))

def find_search_start(spike: SpikeShape, start_i: int, params: dict) -> int:
    """
    Find the start index of the search for the rising phase features.
    """
    if params['mode'] == 'amplitude':
        return find_rise_start_index(
            spike, start_i, params['param']['threshold']
        )
    if params['mode'] == 'time':
        sample_count = count_proportional_sample_count(
            spike.peak_i, start_i, params['param']['ratio']
        )
        return start_i + sample_count
    return start_i

def find_search_end(spike: SpikeShape, start_i: int, params: dict) -> int:
    """
    Find the end index of the search for the rising phase features.
    """
    if params['mode'] == 'dv_max':
        return np.argmax(np.diff(spike.values)).astype(int)
    if params['mode'] == 'time':
        sample_count = count_proportional_sample_count(
            spike.peak_i, start_i, params['param']['ratio']
        )
        return spike.peak_i - sample_count
    return spike.peak_i

def count_proportional_sample_count(
    left_i: int,
    right_i: int,
    proportion: float
) -> int:
    """
    Returns the number of sample points of the given proportion of the distance
    between the left and right indices.
    """
    return np.round((right_i - left_i) * proportion).astype(int)

def find_rise_start_index(
    spike: SpikeShape,
    start_i: int,
    threshold: float
) -> int:
    """
    Find the start index of the rising phase of a spike.
    The rising phase is defined as the first point after the start index
    where the signal exceeds a certain proportion of its amplitude.
    """
    v = spike.values
    peak_i = spike.peak_i

    # align rising phase to the start
    rising_phase = v[start_i:peak_i + 1] - v[start_i]
    threshold_level = threshold * rising_phase[-1]

    # find the first index where the rising phase exceeds the threshold
    return np.where(rising_phase > threshold_level)[0][0] + start_i

def find_dv_peaks(
    spike: SpikeShape,
    search_interval: SignalInterval
) -> list[int]:
    """
    Find the peaks of the first derivative of a spike shape within a given search interval.
    """
    peaks, _ = find_peaks(spike.get_derivative().dv, prominence=(None, None))
    return [peak for peak in peaks if search_interval.is_within(peak)]
