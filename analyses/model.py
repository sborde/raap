"""A module for classes that hold data for analysis."""

import abc

from dataclasses import dataclass
from typing import Optional, Self, Sequence, Protocol
import numpy as np
from pandas import DataFrame

from signalizer.model.experiment.signals import Signal
from signalizer.model.experiment.recordings import MatRecording

IntervalEnds = tuple[np.ndarray, np.ndarray]


@dataclass
class Interval:
    """Class representing a time interval."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Get the duration of the interval."""
        return self.end - self.start

    def shift_relative(self, left: float = 0, right: float = 0) -> Self:
        """Extend the interval by relative amounts."""
        return type(self)(
            self.start + left * self.duration,
            self.end + right * self.duration
        )

    def normalize_by(self, spikes: np.ndarray) -> np.ndarray:
        """Normalize spike times by the given interval."""
        return (spikes - self.start) / self.duration

    def contained(self, events: np.ndarray) -> np.ndarray:
        """Get the events contained within the interval."""
        return events[(events >= self.start) & (events <= self.end)]


class IntervalLike(Protocol):
    """Protocol for interval-like objects."""
    @property
    @abc.abstractmethod
    def start(self) -> float:
        """Get the start of the interval."""

    @property
    @abc.abstractmethod
    def end(self) -> float:
        """Get the end of the interval."""


class RecordingWithSpikes:
    """This class stores a raw recording and its associated spike data."""

    def __init__(self, recording: MatRecording, spikes: DataFrame):
        self.recording = recording
        self.spikes = spikes

    @property
    def n_spikes(self) -> int:
        """Returns the number of spikes in the recording."""
        return self.spikes.shape[0]

    def mark_episode_events(
        self, name: str
    ) -> np.ndarray:
        """
        Mark all events that fall within the specified episode.
        """
        if name == 'all':
            return np.ones(self.n_spikes, dtype=bool)
        return mark_events_in_interval(self.spikes, self.get_episode_ends(name))

    def get_spikes(self, spike_type: Optional[str]) -> np.ndarray:
        """
        Get the spike times as a numpy array.
        """
        if spike_type is None:
            return self.spikes['timestamp'].to_numpy()
        return select_spikes(self.spikes, spike_type)

    def get_signal(
        self,
        name: str,
        fs: Optional[int] = None
    ) -> Signal:
        """
        Get the signal from the recording by name, optionally resampled to fs.
        """
        signal = self.recording.get_waveform(name)
        if fs is not None:
            signal = signal.resampled(fs)
        return signal

    def get_markers(self, name: str) -> np.ndarray:
        """
        Get marker times of the event given by its name.
        """
        events = self.recording.get_event(name)
        markers = np.array([event.time for event in events])
        return markers

    def get_episode_ends(self, name: str) -> IntervalEnds:
        """Returns the start and end times of all episodes with the given name."""
        episodes = self.recording.get_episode(name)
        starts, ends = [], []
        for episode in episodes:
            starts.append(episode.start)
            ends.append(episode.end)
        return (np.array(starts), np.array(ends))

    def get_episode_intervals(self, name: str) -> list[Interval]:
        """Returns a list of Interval objects for all episodes with the given name."""
        starts, ends = self.get_episode_ends(name)
        return [Interval(start, end) for start, end in zip(starts, ends)]

    def get_oscillation_cycles(
        self,
        oscillation_name: str,
        marker_name: str
    ) -> IntervalEnds:
        """
        Returns the start and end times of all cycles of the specified oscillation
        within the episodes of the given name.
        """
        episodes = self.recording.get_episode(oscillation_name)
        markers = self.get_markers(marker_name)
        starts, ends  = [], []
        for episode in episodes:
            is_within = (markers >= episode.start) & (markers <= episode.end)
            markers_within = markers[is_within]
            starts.extend(markers_within[:-1])
            ends.extend(markers_within[1:])
        return (np.array(starts), np.array(ends))

def mark_events_around_trigger(
    events: DataFrame,
    triggers: np.ndarray,
    window: Sequence[float]
) -> np.ndarray:
    """
    Marks events that fall within a specified window around each trigger.
    """
    # Create a mask for all events
    mask = np.zeros(len(events), dtype=bool)
    for trigger in triggers:
        start = trigger + window[0]
        end = trigger + window[1]
        mask |= ((events['timestamp'] >= start) & (events['timestamp'] <= end))
    return mask

def mark_events_in_interval(events: DataFrame, intervals: IntervalEnds) -> np.ndarray:
    """
    Returns a binary mask that marks which events fall within any of the 
    given intervals passed as an input argument.
    """
    timestamps = events['timestamp'].values
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, end in zip(*intervals):
        mask |= ((timestamps >= start) & (timestamps <= end))
    return mask

def get_events_in_intervals(events: DataFrame, intervals: IntervalEnds) -> DataFrame:
    """
    Returns a subset of the given DataFrame where the timestamp
    value falls within any of the specified intervals.
    """
    mask = mark_events_in_interval(events, intervals)
    return events[mask]

def select_spikes(spikes: DataFrame, spike_type: str) -> np.ndarray:
    """Select spikes of the given type."""
    flag = spikes['spike_type'] == spike_type
    return spikes[flag]['timestamp'].to_numpy()
