

from collections import namedtuple

import numpy as np
from pandas import DataFrame

from common.data_utils import CellWrapper

ResultGroup = namedtuple("ResultGroup", ["name", "cells"])
GroupMemberNames = namedtuple("GroupMemberNames", ["name", "cells"])

def get_grouping(
    grouping: DataFrame,
    results: list[CellWrapper]
) -> list[ResultGroup]:
    """
    Returns the CellInfo objects grouped by the input table.
    """
    return [
        collect_members(results, collect_member_names(grouping, int(group_id)))
        for group_id in grouping['group'].unique()
        if not np.isnan(group_id)
    ]

def collect_member_names(
    grouping: DataFrame,
    group_id: int
) -> GroupMemberNames:
    """
    Collects the member names for a specific group.
    """
    return GroupMemberNames(
        name=group_id,
        cells=grouping[grouping['group'] == group_id]['name'].tolist()
    )

def collect_members(
    results: list[CellWrapper],
    group_members: GroupMemberNames
) -> ResultGroup:
    """
    Collects the members of a group based on the cell names.
    """
    return ResultGroup(
        group_members.name,
        [result for result in results if result.info.name in group_members.cells]
    )