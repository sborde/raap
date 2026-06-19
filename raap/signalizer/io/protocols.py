import abc
from typing import Protocol

import h5py


class HDFExportable(Protocol):
    """Interface for objects that can be saved to or loaded from HDF5 file."""
    @abc.abstractmethod
    def save(self, group: h5py.Group): ...
