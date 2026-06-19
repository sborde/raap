"""This module contains functions for episode part comparison analysis."""

from pathlib import Path
import numpy as np
import pandas  as pd

from analyses.model import RecordingWithSpikes
from common.data_utils import CellWrapper
from workflows.run_util.io import ResultPaths


def process_cell(
    input_file: CellWrapper[ResultPaths],
    episode_name: str,
    episode_part_count: int,
    spike_types: list[str],
    individual_path: Path
) -> pd.DataFrame:
    """Analyzes a cell's episode partitioning"""
    cell_data = input_file.data.load()

    episodes = collect_episode_intervals(cell_data, episode_name)
    episode_spikes = collect_spikes_of_episode(
        cell_data.spikes, episodes
    )
    partitioned_spikes = divide_spikes_to_episode_parts(
        episode_spikes, episode_part_count
    )
    spike_type_counts = count_spike_type_per_episode_part(
        partitioned_spikes,
        spike_types,
        value_type='prob'
    )
    spike_type_counts.insert(0, 'episode', episode_name)
    spike_type_counts.insert(0, 'group', input_file.cell_type)
    spike_type_counts.insert(0, 'name', input_file.name)

    save_path = individual_path / input_file.cell_type
    save_path.mkdir(exist_ok=True, parents=True)
    spike_type_counts.to_excel(
        f'{save_path}/{input_file.name}_episode_partition_values.xlsx',
        index=False
    )
    return spike_type_counts

def aggregate_cell(
    data: pd.DataFrame,
    episode_part_count: int,
    spike_types: list[str],
) -> pd.DataFrame:
    """Aggregates result within cell"""
    part_aggregates = []
    for part_i in range(episode_part_count):
        part_rows = data[data['part_id'] == part_i]
        part_aggregate: dict = {
            'name': data.iloc[0]['name'],
            'group': data.iloc[0]['group'],
            'part_id': part_i
        }
        for spike_type in spike_types:
            part_aggregate[f'{spike_type}_mean'] = np.mean(part_rows[spike_type])
            part_aggregate[f'{spike_type}_std'] = np.std(part_rows[spike_type])
        part_aggregates.append(part_aggregate)
    return pd.DataFrame(part_aggregates)

def flatten_aggregate_df(
    data: pd.DataFrame,
    spike_types: list[str],
) -> pd.DataFrame:
    """Returns a flatten version of part aggregate data frame"""
    flat_rows = []
    for _, row in data.iterrows():
        row_header = {
            'name': row['name'],
            'group': row['group'],
            'part_id': row['part_id'],
        }
        for spike_type in spike_types:
            flat_rows.append(row_header | {
                'spike_type': spike_type,
                'value': row[f'{spike_type}_mean']
            })
    return pd.DataFrame(flat_rows)

def collect_episode_intervals(
    data: RecordingWithSpikes,
    name: str
) -> list[tuple[float, float]]:
    """Returns the intervals of the specified episodes"""
    interval_ends = data.get_episode_ends(name)
    return [
        (float(start), float(end)) for (start, end)
        in zip(interval_ends[0], interval_ends[1])
    ]

def collect_spikes_of_episode(
    spikes: pd.DataFrame,
    intervals: list[tuple[float, float]]
) -> list[pd.DataFrame]:
    """Returns spikes within each interval with normalized timestamps"""
    return [
        get_spikes_within_normalized(spikes, start, end)
        for start, end in intervals
    ]

def divide_spikes_to_episode_parts(
    episode_spikes: list[pd.DataFrame],
    n_parts: int
) -> list[list[pd.DataFrame]]:
    """Return episode spikes divided to the given number of parts"""
    edges = np.linspace(0, 1, n_parts + 1)
    return [
        partition_episode_spikes(spikes, edges)
        for spikes in episode_spikes
    ]

def partition_episode_spikes(
    spikes: pd.DataFrame,
    edges: np.ndarray
) -> list[pd.DataFrame]:
    """Returns spikes of an episode divided by edges"""
    return [
        get_spikes_within(spikes, start=start, end=end)
        for start, end in zip(edges[:-1], edges[1:])
    ]

def get_spikes_within_normalized(
    spikes: pd.DataFrame,
    start: float,
    end: float
) -> pd.DataFrame:
    """Returns spikes that fall within the interval, normalized by duration"""
    subset = get_spikes_within(spikes, start=start, end=end)
    subset.loc[:, 'timestamp'] = (subset['timestamp'] - start) / (end - start)
    return subset

def get_spikes_within(
    spikes: pd.DataFrame,
    start: float,
    end: float
) -> pd.DataFrame:
    """Returns spikes that fall within the interval"""
    return spikes[
        (start <= spikes['timestamp']) & (spikes['timestamp'] < end)
    ].copy()

def count_spike_type_per_episode_part(
    episodes_partitions: list[list[pd.DataFrame]],
    spike_types: list[str],
    value_type: str = 'count'
) -> pd.DataFrame:
    """Returns the count of each spike type per episode part"""
    results = []
    for episode_i, episode_partitions in enumerate(episodes_partitions):
        episode_results = []
        for part_i, part_spikes in enumerate(episode_partitions):
            part_result = {
                'episode_id': episode_i,
                'part_id': part_i,
            }
            part_counts = {
                spike_type: np.sum(part_spikes['spike_type'] == spike_type)
                for spike_type in spike_types
            }
            episode_results.append(part_result | part_counts)
        episode_df = pd.DataFrame(episode_results)
        if value_type == 'prob':
            for spike_type in spike_types:
                episode_df[spike_type] = episode_df[spike_type] / episode_df[spike_type].sum()
        results.append(episode_df)
    return pd.concat(results, axis=0)
