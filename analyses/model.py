
from typing import Sequence
import numpy as np
from pandas import DataFrame

from signalizer.model.experiment.recordings import MatRecording

IntervalEnds = tuple[np.ndarray, np.ndarray]

class RecordingWithSpikes:
    """This class stores a raw recording and its associated spike data."""

    def __init__(self, recording: MatRecording, spikes: DataFrame):
        self.recording = recording
        self.spikes = spikes

    @property
    def n_spikes(self) -> int:
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

    def get_markers(self, name: str) -> np.ndarray:
        """
        Get marker times of the event given by its name.
        """
        events = self.recording.get_event(name)
        markers = np.array([event.timestamp.time for event in events])
        return markers

    def get_episode_ends(self, name: str) -> IntervalEnds:
        """Returns the start and end times of all episodes with the given name."""
        episodes = self.recording.get_episode(name)
        return (
            np.array([episode.interval.start for episode in episodes]),
            np.array([episode.interval.end for episode in episodes]),
        )

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