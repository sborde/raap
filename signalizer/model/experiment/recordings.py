import abc
from pathlib import Path

import h5py

import hdf5storage
from signalizer.model.experiment.sequences import EventVector, EpisodeSequence
from signalizer.model.general.sequences import IntervalSequence
from signalizer.model.experiment.signals import Signal
from signalizer.model.temporal.sequences import TimeStampSequence, TimeIntervalSequence


class Recording(abc.ABC):

    @abc.abstractmethod
    def list_waveforms(self) -> list[str]: pass

    @abc.abstractmethod
    def list_markers(self) -> list[str]: pass

    @abc.abstractmethod
    def get_waveform(self, name: str) -> Signal: pass

    @abc.abstractmethod
    def get_episode(self, name: str) -> EpisodeSequence: pass

    @abc.abstractmethod
    def get_event(self, name: str) -> EventVector: pass

    @abc.abstractmethod
    def get_title(self, name: str) -> str: pass


class MatRecording(Recording):

    def __init__(self, path: Path):
        self._path = path

    def __repr__(self) -> str:
        return f'MatRecording({self._path})'

    def list_waveforms(self) -> list[str]:
        with h5py.File(str(self._path), 'r') as file:
            return [name for name in file.keys() if self._is_signal(file[name])]

    def list_markers(self) -> list[str]:
        with h5py.File(str(self._path), 'r') as file:
            return [name for name in file.keys() if self._is_marker(file[name])]

    def get_waveform(self, name: str) -> Signal:
        channel = hdf5storage.loadmat(str(self._path))[name]
        return Signal(channel['times'][0].squeeze(), channel['values'][0].squeeze())

    def get_episode(self, name: str) -> EpisodeSequence:
        start_channel = hdf5storage.loadmat(str(self._path), variable_names=[f'{name}_st'])[f'{name}_st']
        end_channel = hdf5storage.loadmat(str(self._path), variable_names=[f'{name}_end'])[f'{name}_end']
        intervals = IntervalSequence(start_channel['times'][0][:, 0], end_channel['times'][0][:, 0])
        return EpisodeSequence(TimeIntervalSequence(intervals), name)

    def get_event(self, name: str) -> EventVector:
        channel = hdf5storage.loadmat(str(self._path), variable_names=[name])[name]
        timestamps = TimeStampSequence(channel['times'][0][:, 0])
        return EventVector(timestamps, name)

    def get_title(self, name: str) -> str:
        channel = hdf5storage.loadmat(str(self._path), variable_names=[name])
        return channel['title'][0]

    @staticmethod
    def _is_marker(group: h5py.Group) -> bool:
        keys = group.keys()
        return 'times' in keys and 'values' not in keys

    @staticmethod
    def _is_signal(group: h5py.Group) -> bool:
        keys = group.keys()
        return 'times' in keys and 'values' in keys and 'codes' not in keys
