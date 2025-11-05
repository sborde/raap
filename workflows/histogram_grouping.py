"""A module for functions that groups and aggregates histogram results."""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from common.data_utils import CellInfo, CellWrapper
from param_classes.workflows.histogram_grouping import HistogramGroupingWorkflowConfig
from param_classes.workflows.input_output import GroupInfoParams
from workflows.normalized_interval import EpisodeRelatedResults

GroupedResults = tuple[str, list[CellWrapper[EpisodeRelatedResults]]]

def run_grouped_analysis(
    config: HistogramGroupingWorkflowConfig,
    cell_results: list[CellWrapper[EpisodeRelatedResults]]
):
    """Performs the groupwise analysis"""
    processor = HistogramGroupingWorkflow(config)
    groups = create_groups(cell_results, config.grouping)
    for group in groups:
        processor(group)

def create_groups(
    cell_results: list[CellWrapper[EpisodeRelatedResults]],
    params: GroupInfoParams
) -> list[GroupedResults]:
    """
    Returns the CellInfo objects grouped according to the input table.
    """
    table = pd.read_excel(params.path)
    group_ids = get_selected_group_names(
        table, params.column_name, params.groups
    )
    return [
        make_group(
            cell_results,
            group_id, 
            list_member_names(
                table,
                group_id,
                params.column_name
            )
        ) for group_id in group_ids
    ]

def get_selected_group_names(
    table: pd.DataFrame, column_name: str, group_names: list[str]
):
    """
    Returns the CellInfo objects grouped according to the input table,
    filtered by the selected groups.
    """
    group_ids = table[column_name].unique()
    if group_names:
        return [gid for gid in group_ids if gid in set(group_names)]
    return [gid for gid in group_ids if not np.isnan(gid)]

def list_member_names(
    table: pd.DataFrame, group_id: str, column_name: str
) -> list[str]:
    """
    Returns the list of member names for the specified group.
    """
    members = table[table[column_name] == group_id]
    return [row['name'] for _, row in members.iterrows()]

def make_group(
    cell_results: list[CellWrapper],
    group_id: str,
    cell_names: list[str]
) -> GroupedResults:
    """
    Create a group dictionary from the cell DataFrame and group name.
    """
    members = [
        cell for cell in cell_results
        if cell.info.name in set(cell_names)
    ]
    return (group_id, members)


class HistogramGroupingWorkflow:
    """
    Workflow for grouping and aggregating histogram results.
    """
    def __init__(self, config: HistogramGroupingWorkflowConfig):
        self._palette = config.palette
        self._paths = None
        if config.output.path:
            self._paths = {
                name: Path(config.output.path) / dir_name
                for name, dir_name in zip(
                    ['results', 'figures', 'stats'],
                    ['01_results', '02_figures', '03_statistics']
                )
            }
            for path in self._paths.values():
                path.mkdir(parents=True, exist_ok=True)

    def __call__(self, group: GroupedResults):
        """
        Performs the processing steps for a single group defined by the group
        properties.
        """
        group_id, members = group
        
        data = [
            load_data(config['output']['path'], config['output']['suffix'], cell) 
            for cell in members
        ]
        n_trials_per_cell = np.array([cell.data.shape[1] - 2 for cell in data])

        conditions = config['params'].get('marker_condition') or []
        data = [validate_by_condition(cell, conditions) for cell in data]

        n_valid_trials = np.array([cell.data.shape[1] - 2 for cell in data])
        mean_prob = np.mean(n_valid_trials / n_trials_per_cell)

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



def validate_by_condition(
    data: pd.DataFrame, conditions: list[str]
) -> pd.DataFrame:
    """
    Validates the cell data against the specified marker conditions.
    """
    n_trials = data.shape[1]
    valid_flag = np.ones(n_trials, dtype=bool)
    for i in  range(2, n_trials):
        for condition in conditions:
            if data[data['spike_type'] == condition].iloc[:, i].sum() == 0:
                valid_flag[i] = False
                break
    return data.loc[:, valid_flag]

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
