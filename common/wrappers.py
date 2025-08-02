
import abc
from pathlib import Path
from typing import Protocol

class CellInfo:
    """
    A class that holds information about a cell. Name of the cell and its
    type are mandatory.
    """
    def __init__(self, name: str, cell_type: str):
        self._name = name
        self._cell_type = cell_type

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._cell_type


class CellWrapperLike[T](Protocol):
    """
    A protocol that defines the interface for a cell wrapper.
    It should contain the cell's information and data.
    """
    @property
    @abc.abstractmethod
    def info(self) -> CellInfo:
        """Get the information about the cell."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the cell."""

    @property
    @abc.abstractmethod
    def cell_type(self) -> str:
        """Get the type of the cell."""

    @property
    @abc.abstractmethod
    def data(self) -> T:
        """Get the data of the cell."""


class CellWrapper[T]:
    """A wrapper class for a cell's data and its information."""
    def __init__(self, cell_info: CellInfo, data: T):
        self._cell_info = cell_info
        self._data = data

    @property
    def info(self) -> CellInfo:
        """Get the information about the cell."""
        return self._cell_info

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._cell_info.name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._cell_info.cell_type

    @property
    def data(self) -> T:
        """Get the data of the cell."""
        return self._data


class CellPath:
    """
    A class that holds the path information for a cell.
    """
    def __init__(self, path: Path, name: str, cell_type: str):
        self._info = CellInfo(name, cell_type)
        self._path = path

    @property
    def info(self) -> CellInfo:
        """Get the information about the cell."""
        return self._info

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._info.name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._info.cell_type

    @property
    def data(self) -> Path:
        """Get the path of the cell."""
        return self._path

    @property
    def path(self) -> Path:
        """Get the path of the cell."""
        return self._path
