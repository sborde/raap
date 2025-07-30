import abc
from pathlib import Path
from typing import Callable

import h5py

from signalizer.model.experiment.recordings import Recording, MatRecording
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.wrappers import InstanceMatrix
from loader_register import loaders


class Loader[T](abc.ABC):

    def __call__(self, dir_path: Path, entity_label: str) -> T:
        return self.load(dir_path, entity_label)

    @abc.abstractmethod
    def load(self, dir_path: Path, entity_label: str) -> T: pass

    @classmethod
    def create(cls, config: dict) -> 'Loader':
        if config['type'] == 'recording':
            if config['format'] == 'mat':
                return MatRecordingLoader()
            elif config['format'] == 'hdf5':
                return HDFLoader(config['loader'])
        else:
            return HDFLoader.create(config['params'])


class MatRecordingLoader(Loader[Recording]):

    def load(self, dir_path: Path, entity_label: str) -> Recording:
        return MatRecording(dir_path / f'{entity_label}.mat')


class HDFLoader[T](Loader[T]):
    """Loader any HDFExportable object as HDF5 files"""

    def __init__(self, loader: Callable[[h5py.Group], T], suffix: str = ''):
        self._loader = loader
        self._suffix = '_' + suffix

    def load(self, dir_path: Path, entity_label: str) -> T:
        with h5py.File(dir_path / f'{entity_label + self._suffix}.h5', 'r') as file:
            return self._loader(file)

    @classmethod
    def create(cls, config: dict) -> 'HDFLoader':
        if config['type'] in loaders:
            return HDFLoader(
                loaders[config['type']], config['suffix']
            )
        elif config['type'] == 'instance_vector':
            return HDFLoader(
                InstanceSequenceLoader.create(config['params']), config['suffix']
            )
        else:
            raise ValueError(f"Unknown loader type: {config['type']}")


class InstanceSequenceLoader[R: TimeReferenced, D: MatrixLike]:
    """Loads the analysis instance from HDF5 file"""

    def __init__(self, reference_loader: Callable[[h5py.Group], R], data_loader: Callable[[h5py.Group], D]):
        self._reference_loader = reference_loader
        self._data_loader = data_loader

    def __call__(self, group: h5py.Group) -> InstanceMatrix[R, D]:
        reference = self._reference_loader(group)
        data = self._data_loader(group)
        return InstanceMatrix[R, D](reference, data)

    @classmethod
    def create(cls, config: dict) -> 'InstanceSequenceLoader':
        if config['reference'] not in loaders:
            raise ValueError(f"Unknown reference loader: {config['reference']}")
        if config['data'] not in loaders:
            raise ValueError(f"Unknown data loader: {config['data']}")
        return InstanceSequenceLoader(loaders[config['reference']], loaders[config['data']])
