"""
This module defines classes that contains group-related data
and group's associated metadata.
"""


class GroupInfo:
    """
    A class that holds information about a group of cells.
    It contains the name of the group and the members of the group.
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the group."""
        return self._name


class GroupWrapper[T]:
    """A wrapper class for a group of cells' data and its information."""
    def __init__(self, group_info: GroupInfo, members: list[T]):
        self._group_info = group_info
        self._members = members

    @property
    def info(self) -> GroupInfo:
        """Get the information about the group."""
        return self._group_info

    @property
    def name(self) -> str:
        """Get the name of the group."""
        return self._group_info.name

    @property
    def members(self) -> list[T]:
        """Get the members of the group."""
        return self._members

    @property
    def n_members(self) -> int:
        """Get the number of members in the group."""
        return len(self._members)
