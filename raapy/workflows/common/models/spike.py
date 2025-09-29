"""
This module contains the classes for representing spike likes, spike shapes,
and spike rising parts.
"""
import abc

from typing import Optional, Protocol, Self
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
    def times(self) -> np.ndarray:
        """
        Get the times of the spike likes.
        """

    @property
    @abc.abstractmethod
    def values(self) -> np.ndarray:
        """
        Get the values of the spike likes.
        """

    def __len__(self) -> int:
        """
        Get the number of spike likes.
        """
        return self.values.shape[1]


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


class SpikeDerivative:
    """
    A class to represent the first derivative of a spike.
    """

    def __init__(self, times: np.ndarray, dv: np.ndarray):
        self._times = times
        self._dv = dv

    @property
    def times(self) -> np.ndarray:
        """Get the times of the spike derivative."""
        return self._times

    @property
    def dv(self) -> np.ndarray:
        """Get the derivative values of the spike derivative."""
        return self._dv


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


class PlottableSpike(Protocol):
    """A protocol for objects that can be plotted as spikes."""
    
    @property
    @abc.abstractmethod
    def times(self) -> np.ndarray:
        """Get the time points of the spike."""

    @property
    @abc.abstractmethod
    def values(self) -> np.ndarray:
        """Get the values of the spike."""

    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Get the time step of the spike."""

    @abc.abstractmethod
    def get_phase(self) -> SpikePhase:
        """Get the phase representation of the spike."""


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
        self._peak_i = int(np.argmin(np.abs(times - 0)))

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

    @property
    def peak_i(self) -> int:
        """
        Get the index of the peak of the spike shape.
        """
        return self._peak_i

    def get_derivative(self) -> SpikeDerivative:
        """
        Get the derivative of the spike shape.
        """
        return SpikeDerivative(self.times[:-1], np.diff(self.values))

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
        self._times = times
        self._values = values
        self._timestamps = timestamps
        self._dt = dt

    def __len__(self) -> int:
        """
        Get the number of spike shapes.
        """
        return self._values.shape[1]

    def __iter__(self):
        """
        Iterate over the spike shapes.
        """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> SpikeShape:
        """
        Get the spike shape at the given index.
        """
        return SpikeShape(
            self.times,
            self.values[:, index],
            self.timestamps[index],
            self.dt
        )

    @property
    def timestamps(self) -> np.ndarray:
        """
        Get the timestamps of the spike shapes.
        """
        return self._timestamps

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the spike shapes.
        """
        return self._times

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the spike shapes.
        """
        return self._values

    @property
    def dt(self) -> float:
        """
        Get the time step of the spike shapes.
        """
        return self._dt


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
        spikes.data.signals.as_matrix().T,
        np.array([event.timestamp.time for event in spikes.reference]),
        spikes.data.signals.si
    )

def load_spike_shapes(path: Path, params: dict) -> SpikeShapes:
    """Load spike shapes from a recording file."""
    recording = MatRecording(path)
    return extract_triggered_segments(recording, **params)
