
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from common.data_utils import CellInfo, CellWrapper
from triggered_firing.plot_utils import create_spike_palette_for_celltype


def run_grouped_analysis(config: dict):
    """Performs the groupwise analysis"""
    paths = create_outputs(config['output']['path'])
    for group in get_grouping(config['grouping']['path']):
        process_group(config, paths, group)

def create_outputs(base_dir: str) -> dict[str, Path]:
    """
    Creates the output paths for the grouping analysis.
    """
    output_base = Path(base_dir) / 'grouping'
    output_paths = {
        name: output_base / name
        for name in ['figures', 'aggregated']
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_paths

def process_group(config: dict, paths: dict[str, Path], group: dict):
    """
    Performs the processing steps for a single group defined by the group
    properties.
    """
    data = [
        load_data(config['output']['path'], config['output']['suffix'], cell) 
        for cell in group['members']
    ]
    n_trials = np.array([cell.data.shape[1] - 2 for cell in data])

    conditions = config['params'].get('marker_condition') or []
    data = [validate_by_condition(cell, conditions) for cell in data]

    n_valid_trials = np.array([cell.data.shape[1] - 2 for cell in data])
    mean_prob = np.mean(n_valid_trials / n_trials)

    suffix = config['output']['suffix']
    aggregated_data =  pd.concat(
        [aggregate_cell(cell) for cell in data],
        ignore_index=True
    )
    aggregated_data.to_csv(
        paths['aggregated'] / f'{group['id']}_mean_{suffix}.csv'
    )

    fig = plot_grouped_result(aggregated_data, config)
    plt.title(
        '\n'.join([
            f"Mean activity during {config['params']['episode']} cycle",
            f"P({'&'.join(conditions)})={mean_prob:.4f}"  if conditions else ""
        ])
    )
    fig.savefig(
        paths['figures'] / f'{group['id']}_mean_{suffix}.png'
    )
    plt.close(fig)

def load_members(params: dict, members: list[CellInfo]) -> list[CellWrapper[pd.DataFrame]]:
    """
    Load the data for the specified members.
    """
    return [
        load_data(params['path'], params['suffix'], member) 
        for member in members
    ]

def validate_by_condition(
    cell_data: CellWrapper[pd.DataFrame], conditions: list[str]
) -> CellWrapper[pd.DataFrame]:
    """
    Validates the cell data against the specified marker conditions.
    """
    data_frame = cell_data.data
    n_trials = data_frame.shape[1]
    valid_flag = np.ones(n_trials, dtype=bool)
    for i in  range(2, n_trials):
        for condition in conditions:
            if data_frame[data_frame['spike_type'] == condition].iloc[:, i].sum() == 0:
                valid_flag[i] = False
                break

    print(
        f"{cell_data.info.name} " +
        f"all triggers: {n_trials - 2}, " +
        f"valid triggers: {valid_flag[2:].sum()}"
    )

    return CellWrapper(
        cell_data.info,
        data_frame.loc[:, valid_flag]
    )

def get_grouping(path: str) -> list[dict]:
    """
    Returns the CellInfo objects grouped according to the input table.
    """
    assignment = pd.read_excel(path)
    group_ids = assignment['group'].unique()
    return [
        make_group(assignment, group_id)
        for group_id in group_ids
        if not np.isnan(group_id)
    ]

def make_group(assignment: pd.DataFrame, group_id: str) -> dict:
    """
    Create a group dictionary from the cell DataFrame and group name.
    """
    members = assignment[assignment['group'] == group_id]
    infos = [
        CellInfo(row['name'], row['cell_type'])
        for _, row in members.iterrows()
    ]
    return {'id': int(group_id), 'members': infos}

def load_data(
    path: str,
    suffix: str,
    info: CellInfo
) -> CellWrapper[pd.DataFrame]:
    """
    Load the data for the specified cell.
    """
    cell_data = pd.read_csv((
        Path(path) /
        'statistics' /
        info.cell_type /
        f'{info.name}_{suffix}.csv'
    ))
    return CellWrapper(info, cell_data)

def aggregate_cell(
    data: CellWrapper[pd.DataFrame]
) -> pd.DataFrame:
    """
    Aggregate the data for the specified cell.
    """
    average = data.data[['centers', 'spike_type']].copy()
    average['cell_name'] = data.info.name
    average['cell_type'] = data.info.cell_type
    average['mean'] = data.data.iloc[:, 2:].mean(axis=1, skipna=True)
    return average

def plot_grouped_result(aggregated_data: pd.DataFrame, config: dict) -> Figure:
    """
    Plot the results of the grouped analysis.
    """
    cell_types = aggregated_data['cell_type'].unique()
    if len(cell_types) > 1:
        palette = config['palette']['spike_type']
    else:
        palette = create_spike_palette_for_celltype(
            cell_types[0],
            config['plot']['coloring_mode'],
            config['palette']
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    if config['plot']['group']['superimposed']:
        for _, group in aggregated_data.groupby('cell_name'):
            sns.lineplot(
                data=group,
                x='centers',
                y='mean',
                hue='spike_type',
                palette=palette,
                alpha=0.5
            )
    spike_type_average = aggregated_data.groupby(
        by=['centers', 'spike_type']
    )['mean'].mean().reset_index()
    sns.lineplot(
        data=spike_type_average,
        x='centers',
        y='mean',
        hue='spike_type',
        palette=palette,
        linewidth=2
    )

    handles, labels = ax.get_legend_handles_labels()
    handle_dict = {}
    for handle, label in zip(handles, labels):
        if label not in handle_dict:
            handle_dict[label] = handle

    # Get unique handles and labels
    unique_handles = list(handle_dict.values())
    unique_labels = list(handle_dict.keys())

    ax.legend(
        unique_handles,
        unique_labels,
        title='Spike type',
        loc='upper right',
        bbox_to_anchor=(1.0, 1),
        borderaxespad=0.
    )

    return fig
