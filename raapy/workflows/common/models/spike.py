"""
This module contains the classes for representing spike likes, spike shapes,
and spike rising parts.
"""
import abc

from typing import Self
from pathlib import Path

import numpy as np

from scipy.spatial.qhull import ConvexHull
from scipy.signal import savgol_filter

from signalizer.model.experiment.recordings import MatRecording
from signalizer.processing.extractors import EventRelatedSegmentExtractor
from signalizer.processing.factories import ProcessingSequence
from signalizer.processing.retrievers import SignalRetriever, EventRetriever
from signalizer.processing.segment_cutters import EventRelatedSegmentCutter
from signalizer.processing.signalprocessing.resamplers import Resampler
from signalizer.model.general.data import Interval
from signalizer.model.temporal.data import TimeInterval

from .series import SegmentMatrix, DataSeries, SeriesMatrix

class SpikeLikes(abc.ABC):
    """
    A class to represent a set of spike likes.
    """

    @property
    @abc.abstractmethod
    def timestamps(self) -> np.ndarray:
        """
        Get the timestamps of the spike likes.
        """

    @property
    @abc.abstractmethod
    def segments(self) -> SegmentMatrix:
        """
        Get the segments of the spike likes.
        """

    @property
    @abc.abstractmethod
    def start(self) -> int:
        """
        Get the starts of the spike likes.
        """

    @property
    @abc.abstractmethod
    def peak(self) -> int:
        """
        Get the peaks of the spike likes.
        """

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the spike likes.
        """
        return self.segments.times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the spike likes.
        """
        return self.segments.values

    def __len__(self) -> int:
        """
        Get the number of spike likes.
        """
        return len(self.segments)

    def get_phases(self) -> 'SpikePhases':
        """
        Get the phase plots of the spike likes.
        """
        derivatives = np.diff(self.values, axis=1)
        return SpikePhases(self.values[:, :-1], derivatives)


class SpikePhase:
    """
    A class to represent a spike phase.
    """

    def __init__(self, v: np.ndarray, dvdt: np.ndarray):
        self._v = v
        self._dvdt = dvdt

    @property
    def v(self) -> np.ndarray:
        """Get the voltage values of the spike phase."""
        return self._v

    @property
    def dvdt(self) -> np.ndarray:
        """Get the derivative of voltage with respect to time."""
        return self._dvdt


class AverageSpikeShape:
    """
    A class to represent the average spike shape.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray, dt: float):
        self.times = times
        self.values = values
        self.dt = dt

    def get_phase(self) -> SpikePhase:
        """Get the phase of the average spike shape."""
        dvdt = np.diff(self.values) / self.dt
        return SpikePhase(self.values[:-1], dvdt)


class SpikeShape:
    """
    A class to represent a spike shape.
    """

    def __init__(
        self,
        times: np.ndarray,
        values: np.ndarray,
        timestamp: float,
        dt: float
    ):
        self._times = times
        self._values = values
        self._timestamp = timestamp
        self._dt = dt

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the spike shape.
        """
        return self._times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the spike shape.
        """
        return self._values

    @property
    def timestamp(self) -> float:
        """
        Get the timestamp of the spike shape.
        """
        return self._timestamp

    @property
    def dt(self) -> float:
        """
        Get the time step of the spike shape.
        """
        return self._dt

    def get_phase(self) -> SpikePhase:
        """
        Get the phase of the spike shape.
        """
        dvdt = np.diff(self.values) / self.dt
        return SpikePhase(self.values[:-1], dvdt)

    @staticmethod
    def average(spikes: list['SpikeShape']) -> AverageSpikeShape:
        """
        Calculate the average spike shape from a list of spikes.
        """
        if not spikes:
            return AverageSpikeShape(
                np.array([]),
                np.array([]),
                dt=0.0
            )

        mean_values = np.mean([s.values for s in spikes], axis=0)
        return AverageSpikeShape(
            spikes[0].times,
            mean_values,
            spikes[0].dt
        )

class SpikeShapes(SpikeLikes):
    """
    A class to represent a set of spike shapes.
    """

    def __init__(
        self,
        times: np.ndarray,
        values: np.ndarray,
        timestamps: np.ndarray,
        dt: float
    ):
        self._segments = SegmentMatrix(times, values)
        self._timestamps = timestamps
        # self._start = self._find_start()
        self._peak = self._find_peak()
        self._dt = dt

    @property
    def timestamps(self) -> np.ndarray:
        """
        Get the timestamps of the spike shapes.
        """
        return self._timestamps

    @property
    def segments(self) -> SegmentMatrix:
        """
        Get the segments of the spike shapes.
        """
        return self._segments

    @property
    def start(self) -> int:
        """
        Find the start of the spike shapes.
        """
        return -1

    @property
    def peak(self) -> int:
        """
        Find the peak of the spike shapes.
        """
        return self._peak

    @property
    def dt(self) -> float:
        """
        Get the time step of the spike shapes.
        """
        return self._dt

    @property
    def concavity_indices(self) -> np.ndarray:
        """
        Get the concavity indices of the spike shapes.
        """
        rising_phases = self.get_rising_parts().get_phases()
        convex_hulls = rising_phases.get_convex_hulls()
        real_areas = rising_phases.areas
        convex_areas = np.array([hull.area for hull in convex_hulls])

        return 100 * (convex_areas - real_areas) / convex_areas

    def __len__(self) -> int:
        """
        Get the number of spike shapes.
        """
        return self.values.shape[0]

    def __getitem__(self, index: int) -> SpikeShape:
        """
        Get the spike shape at the given index.
        """
        return SpikeShape(
            self.times,
            self.values[index, :],
            self.timestamps[index],
            self.dt
        )

    def __iter__(self):
        """
        Iterate over the spike shapes.
        """
        for i in range(len(self)):
            yield SpikeShape(
                self.times,
                self.values[i, :],
                self.timestamps[i],
                self.dt
            )

    def start_to_peak_normalized(self) -> Self:
        """
        Normalize the spike shapes to the start to peak range.
        """

        normalized_values = []
        valid_timestamps = []
        for i in range(len(self)):
            v = self.values[i, :]
            left_v = v[:self.peak+1]
            dv = np.diff(left_v)
            max_dv_i = np.argmax(dv)
            threshold_dv = 0.1 * np.max(dv)
            if not np.any(dv[:max_dv_i] < threshold_dv):
                continue
            start_i = np.where(dv[:max_dv_i] < threshold_dv)[0][-1] + 1
            peak_i = self.peak
            normalized_values.append(
                (v - v[start_i]) / (v[peak_i] - v[start_i])
            )
            valid_timestamps.append(self.timestamps[i])
        
        return type(self)(
            self.times,
            np.stack(normalized_values, axis=0),
            np.array(valid_timestamps),
            self.dt
        )

    def get_half_spikes(self) -> Self:
        """
        Get the half spikes of the spike shapes.
        """
        return type(self)(
            self.times[:self.peak+1],
            self.values[:, :self.peak+1],
            self.timestamps,
            self.dt
        )

    def get_rising_parts(self) -> 'SpikeRisingParts':
        """
        Get the rising parts of the spike shapes.
        """
        start = self.start
        peak = self.peak

        return SpikeRisingParts(
            self.times[start:peak+1],
            self.values[:, start:peak+1],
            self.timestamps
        )

    def _find_start(self) -> int:
        """
        Find the start of the spike shapes.
        """
        # calculate the average shape
        avg_shape = np.mean(self.values, axis=0)
        # calculate the derivative of the average shape
        dv = np.diff(avg_shape)
        # find the index of the maximum derivative
        max_dv_i = np.argmax(dv)
        # find the threshold value
        threshold_v = 0.1 * np.max(avg_shape)
        # find the indices of the values above the threshold
        above_threshold_indices = np.where(dv[:max_dv_i] < threshold_v)[0]
        # return the index of the last value above the threshold
        return int(above_threshold_indices[-1] + 1)

    def _find_peak(self) -> int:
        """
        Find the peak of the spike shapes.
        """
        return int(np.argmin(np.abs(self.times - 0)))

    def apply_sg_filter(self, window_length: int = 5, polyorder: int = 2) -> Self:
        """
        Apply Savitzky-Golay filter to the spike shapes.
        
        Args:
            window_length: Length of the filter window (must be odd)
            poly_order: Order of the polynomial used to fit the samples
            
        Returns:
            New SpikeShapes instance with filtered values
        """
        filtered_values = np.array([
            savgol_filter(spike, window_length, polyorder)
            for spike in self.values
        ])

        return type(self)(
            self.times,
            filtered_values,
            self.timestamps,
            self.dt
        )


class SpikeRisingParts(SpikeLikes):
    """
    A class to represent a set of spike rising parts.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray, timestamps: np.ndarray):
        self._segments = SegmentMatrix(times, values)
        self._timestamps = timestamps

    @property
    def segments(self) -> SegmentMatrix:
        """
        Get the segments of the spike rising parts.
        """
        return self._segments

    @property
    def timestamps(self) -> np.ndarray:
        """
        Get the timestamps of the spike rising parts.
        """
        return self._timestamps

    @property
    def start(self) -> int:
        """
        Get the start of the spike rising parts.
        """
        return 0

    @property
    def peak(self) -> int:
        """
        Get the peak of the spike rising parts.
        """
        return len(self.segments) - 1

    def __getitem__(self, index: int) -> DataSeries:
        """
        Get the spike rising part at the given index.
        """
        return self._segments[index]


class SpikePhases:
    """
    A class to represent a set of spike phase plots.
    """

    def __init__(self, voltages: np.ndarray, derivatives: np.ndarray):
        self._series = SeriesMatrix(voltages, derivatives)

    @property
    def derivatives(self) -> np.ndarray:
        """
        Get the derivatives of the spike phases.
        """
        return self._series.y

    @property
    def voltages(self) -> np.ndarray:
        """
        Get the voltages of the spike phases.
        """
        return self._series.x

    @property
    def areas(self) -> np.ndarray:
        """
        Get the areas of the spike phases.
        """
        return np.array([series.area for series in self])

    def get_convex_hulls(self) -> list[DataSeries]:
        """
        Get the convex hulls of the spike phases.
        """
        convex_hulls = []
        for _, polygon in enumerate(self):
            point_set = np.stack([polygon.x, polygon.y], axis=1)
            convex_hull = ConvexHull(point_set)
            hull_vertices_indices = convex_hull.vertices
            convex_hulls.append(
                DataSeries(
                    point_set[hull_vertices_indices, 0],
                    point_set[hull_vertices_indices, 1]
                )
            )

        return convex_hulls

    def __len__(self) -> int:
        """
        Get the number of spike phases.
        """
        return len(self._series)

    def __getitem__(self, index: int) -> DataSeries:
        """
        Get the spike phase at the given index.
        """
        return self._series[index]


def extract_triggered_segments(
        recording: MatRecording,
        signal_name: str,
        event_name: str,
        cut_window,
        fs: int
) -> SpikeShapes:
    """
    Extracts the spikes from the recording.
    """

    spikes = EventRelatedSegmentExtractor(
        SignalRetriever(signal_name, ProcessingSequence([Resampler(fs)])),
        EventRetriever(event_name),
        EventRelatedSegmentCutter(TimeInterval(Interval.as_interval(cut_window)))
    ).extract(recording)

    return SpikeShapes(
        spikes.data.times,
        spikes.data.signals.as_matrix(),
        np.array([event.timestamp.time for event in spikes.reference]),
        spikes.data.signals.si
    )

def load_spike_shapes(input_path: Path, config: dict) -> SpikeShapes:
    """Load spike shapes from a recording file."""
    recording = MatRecording(input_path)
    return extract_triggered_segments(recording, **config)
