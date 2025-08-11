
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from common.data_utils import CellInfo, CellWrapper
from triggered_firing.plot_utils import create_spike_palette_for_celltype

def convert_to_cell_info(data: pd.DataFrame) -> list[CellInfo]:
    """
    Convert a DataFrame to a list of CellInfo objects.
    """
    return [
        CellInfo(row['name'], row['cell_type'])
        for _, row in data.iterrows()
    ]

def make_group(cells: pd.DataFrame, group_name: str) -> dict:
    """
    Create a group dictionary from the cell DataFrame and group name.
    """
    members = cells[cells['group'] == group_name]
    return {
        'group_name': int(group_name),
        'members': convert_to_cell_info(members)
    }

def get_grouping(data: pd.DataFrame) -> list[dict]:
    """
    Returns the CellInfo objects grouped by the input table.
    """
    return [
        make_group(data, group_name)
        for group_name in data['group'].unique()
        if not np.isnan(group_name)
    ]

def load_cell_data(
    params: dict,
    info: CellInfo
) -> CellWrapper[pd.DataFrame]:
    """
    Load the data for the specified cell.
    """
    cell_data = pd.read_csv((
        Path(params['path']) /
        'statistics' /
        info.cell_type /
        f'{info.name}_{params["suffix"]}.csv'
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

def aggregate_group(
    members: list[CellWrapper[pd.DataFrame]]
) -> pd.DataFrame:
    """
    Aggregate the data for the specified group members.
    """
    cell_averages = [
        aggregate_cell(member)
        for member in members
    ]
    return pd.concat(cell_averages, ignore_index=True)

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
    for _, group in aggregated_data.groupby('cell_name'):
        sns.lineplot(
            data=group,
            x='centers',
            y='mean',
            hue='spike_type',
            palette=palette,
            alpha=0.5
        )
    spike_type_average = aggregated_data.groupby(by=['centers', 'spike_type'])['mean'].mean().reset_index()
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

def run_grouped_analysis(config: dict):
    grouping_dir = Path(config['output']['path']) / 'grouping'
    grouping_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = grouping_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir = grouping_dir / 'aggregated'
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    groups = get_grouping(pd.read_excel(config['grouping']['path']))
    for group in groups:
        group_data = [
            load_cell_data(config['output'], member)
            for member in group['members']
        ]
        aggregated_data = aggregate_group(group_data)
        aggregated_data.to_csv(
            aggregated_dir /
            (
                f'{group["group_name"]}' +
                '_mean' +
                f'_{config["output"]["suffix"]}' +
                '.csv'
            )
        )

        fig = plot_grouped_result(aggregated_data, config)
        fig.savefig(
            figure_dir /
            (
                f'{group["group_name"]}' +
                '_mean' + 
                f'_{config["output"]["suffix"]}' +
                '.png'
            )
        )
        plt.close(fig)