"""Module for classes representing recordings."""

from dataclasses import dataclass

from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from tglab.util.hdf import get_group, get_float, get_vector, list_keys
from .signals import Signal
# from .interval import Episode, Interval


@dataclass
class Interval:
    """Represents a time interval with a start and end time."""
    start: float
    end: float


@dataclass
class Episode:
    """Represents an episode consisting of a name and a list of intervals."""
    name: str
    intervals: list[Interval]


ChanneltypeChecker = Callable[[h5py.Group, str], bool]

def signal_names(path: str) -> list[str]:
    """List all waveform names in the recording."""
    return channel_names(path, check=_issignal)

def marker_names(path: str) -> list[str]:
    """List all marker names in the recording."""
    return channel_names(path, check=_ismarker)

def interval_names(path: str) -> list[str]:
    """List all interval names in the recording."""
    starts, ends = set(), set()
    for name in marker_names(path):
        if name.endswith('_st'):
            starts.add(name[:-3])
        elif name.endswith('_end'):
            ends.add(name[:-4])
    return sorted(starts.intersection(ends))

def channel_names(
    path: str | Path,
    check: ChanneltypeChecker | None = None
) -> list[str]:
    """
    List all channel names in the recording matching the selector.
    If no selector is provided, all types of channel names are returned.
    """
    with h5py.File(path, 'r') as file:
        names = list(file.keys())
        if check is None:
            return names
        return list(filter(lambda name: check(file, name), names))

def get_signal(path: str | Path, name: str) -> Signal:
    """Returns a signal by its name."""
    try:
        with h5py.File(str(path), 'r') as file:
            channel = get_group(file, name)
            t = get_vector(channel, 'times')
            v = get_vector(channel, 'values')
            si = get_float(channel, 'interval')
    except KeyError as e:
        raise MissingSignalException(name) from e
    return Signal(t, v, si)

def get_markers(path: str | Path, name: str) -> np.ndarray:
    """Returns a ndarray of marker timestamps."""
    try:
        with h5py.File(str(path), 'r') as file:
            channel = get_group(file, name)
            times = get_vector(channel, 'times')
            return times
    except KeyError as e:
        raise MissingMarkerException(name) from e

def get_intervals(path: str | Path, name: str) -> list[Interval]:
    """Returns a list of intervals of the given name."""
    try:
        starts = get_markers(path, f'{name}_st')
        ends = get_markers(path, f'{name}_end')
    except MissingMarkerException as e:
        raise MissingIntervalException(name) from e
    return [Interval(start, end) for start, end in zip(starts, ends)]

def get_episode(path: str | Path, name: str) -> Episode:
    """Returns an episode of the given name."""
    intervals = get_intervals(path, name)
    return Episode(name, intervals)

def get_event(path: str | Path, name: str) -> np.ndarray:
    """Returns a ndarray of event times."""
    return get_markers(path, name)

def _ismarker(file: h5py.Group, name: str) -> bool:
    """
    Check if the given name corresponds
    to a marker channel in the file.
    """
    keys = list_keys(file[name])
    return 'times' in keys and 'values' not in keys

def _issignal(file: h5py.Group, name: str) -> bool:
    """
    Check if the given name corresponds
    to a signal channel in the file.
    """
    keys = list_keys(file[name])
    return 'times' in keys and 'values' in keys and 'codes' not in keys


class MissingChannelException(Exception):
    "Raised when a required channel is missing from the recording."
    def __init__(self, channel_name: str, channel_type: str):
        super().__init__(
            f" {channel_type.capitalize()} '{channel_name}' is missing."
        )


class MissingSignalException(MissingChannelException):
    "Raised when a required signal is missing from the recording."
    def __init__(self, signal_name: str):
        super().__init__(signal_name, 'signal')


class MissingMarkerException(MissingChannelException):
    "Raised when a required marker is missing from the recording."
    def __init__(self, marker_name: str):
        super().__init__(marker_name, 'marker')


class MissingIntervalException(MissingChannelException):
    "Raised when a required interval is missing from the recording."
    def __init__(self, interval_name: str):
        super().__init__(interval_name, 'interval')
