"""
This module defines classes that contains cell-related data
and cell's associated metadata.
"""

from pathlib import Path

class CellInfo:
    """
    A class that holds information about a cell. Name of the cell and its
    type are mandatory.
    """
    def __init__(self, name: str, group: str):
        self._name = name
        self._group = group

    @property
    def name(self) -> str:
        """Returns the cell's name."""
        return self._name

    @property
    def group(self) -> str:
        """Returns the cell's group."""
        return self._group


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
    def group(self) -> str:
        """Get the group of the cell."""
        return self._cell_info.group

    @property
    def data(self) -> T:
        """Get the data of the cell."""
        return self._data

    def with_data[N](self, new_data: N) -> 'CellWrapper[N]':
        """Create a new CellWrapper with the same info but different data."""
        return CellWrapper(self._cell_info, new_data)


class CellPath:
    """
    A class that holds the path information for a cell.
    """
    def __init__(self, info: CellInfo, path: Path):
        self._info = info
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
    def group(self) -> str:
        """Get the group of the cell."""
        return self._info.group

    @property
    def path(self) -> Path:
        """Get the path of the cell."""
        return self._path
