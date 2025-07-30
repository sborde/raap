import abc
from pathlib import Path

import h5py

from signalizer.model.wrappers import EntityData
from signalizer.framework.pipeline.protocols import EntityHandler
from signalizer.io.protocols import HDFExportable


class EntitySaver[T](EntityHandler[T], abc.ABC):

    @abc.abstractmethod
    def save(self, wrapper: EntityData[T]): pass

    def run(self, wrapper: EntityData[T]) -> None:
        self.save(wrapper)

    @classmethod
    def from_config(cls, config: dict) -> 'EntitySaver':
        """Create an EntitySaver from a configuration dictionary."""
        if config['format'] == 'hdf5':
            return SingleHDFEntitySaver(Path(config['path']), config['suffix'], config.get('extension', 'h5'))
        raise ValueError(f"Unsupported format: {config['format']}")


class SingleHDFEntitySaver[T: HDFExportable](EntitySaver[T]):
    """Saver class that saves the entity to a file in HDF5 format.
    The resulting file name is the entity name with the given suffix."""
    def __init__(self, path: Path, suffix: str, extension: str = 'h5'):
        if path.exists() and not path.is_dir():
            raise ValueError("SingleHDFEntitySaver's argument points to an existing file.")

        self._path = path
        if not path.exists():
            path.mkdir(parents=True)

        self._suffix = f'_{suffix}.{extension}'

    def save(self, entity: EntityData[T]):
        file_name = entity.label + self._suffix
        with h5py.File(self._path / file_name, 'w') as file:
            entity.data.save(file)
