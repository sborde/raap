"""
This module contains the workflow definition of the cellwise episode description.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from pandas import DataFrame
from scipy.stats import wilcoxon, kruskal

from matplotlib.figure import Figure

from common.data_utils import CellWrapper
from analyses.model import (
    RecordingWithSpikes
)
from param_classes.workflows.episode_analysis import EpisodeAnalysisWorkflowParams
from workflows.run_util.io import (
    ResultPaths,
    RecordingWithSpikes
)

def run_analysis(
    cells: list[CellWrapper[ResultPaths]],
    params: EpisodeAnalysisWorkflowParams
) -> list[CellWrapper[DataFrame]]:
    """Run the spike type vs. sleep episode relationship analysis."""
    analysis = EpisodeFiringBySpikeType(params)
    return [analysis.run(input_file) for input_file in tqdm(cells)]

def create_groups(cells: list[CellWrapper]) -> dict[str, list[CellWrapper]]:
    """Create groups based on cell types."""
    groups = {}
    for cell in cells:
        group_name = cell.info.cell_type
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(cell)
    return groups

def run_comparison(
    groups: dict[str, list[CellWrapper[DataFrame]]],
    params: EpisodeAnalysisWorkflowParams
):
    """Run the group comparison of spike type vs. sleep episode relationship."""
    comparison = EpisodeAnalysisGroupedWorkflow(params)
    for name, members in tqdm(groups.items(), desc='Process groups'):
        comparison.run(members, name)

class EpisodeFiringBySpikeType:
    """
    An analysis class for the analysis of the relationship 
    between spike types and sleep episodes or sections.
    """

    def __init__(self, params: EpisodeAnalysisWorkflowParams):
        if params.output.path:
            self._output_dir = (
                Path(params.output.path) /
                '01_results' /
                'Individual'
            )
            self._output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._output_dir = None

        self._palette = params.palette
        self._params = params.analysis

    def run(self, cell: CellWrapper[ResultPaths]) -> CellWrapper[DataFrame]:
        """
        Run the spike type distribution analysis.
        """
        # Load data
        data = cell.data.load()

        # Iterate over episodes
        results = []
        for episode in self._params.episodes:
            try:
                results.append(self._process_episode(data, episode))
            except KeyError:
                if self._params.verbose:
                    print(
                        f"[INFO] Episode '{episode}' is missing " +
                        f"in {cell.info.name}."
                    )
                continue
        result = pd.concat(results, ignore_index=True)

        if self._output_dir:
            self._save_results(CellWrapper(cell.info, result))

        return CellWrapper(cell.info, result)

    def _process_episode(self, data: RecordingWithSpikes, name: str) -> DataFrame:
        """
        Calculates the mean firing rate of each spike type
        during each interval of the given type of episode.
        """
        spikes = data.spikes.copy()
        timestamps = spikes['timestamp'].values
        starts, ends = data.get_episode_ends(name)
        results =  []
        for start, end in zip(starts, ends):
            duration = end - start
            is_within = (timestamps >= start) & (timestamps <= end)
            result = {
                'episode': name,
                'start': start,
                'end': end,
                'duration': duration
            }
            for spike_type in self._params.spike_types:
                result[spike_type] = np.sum(
                    is_within & (spikes['spike_type'] == spike_type)
                )
            results.append(result)
        return DataFrame(results)

    def _save_results(self, cell: CellWrapper[DataFrame]) -> None:
        """
        Save the results of the analysis.
        """
        if not self._output_dir:
            return
        result_dir = self._output_dir / cell.info.cell_type
        result_dir.mkdir(parents=True, exist_ok=True)
        cell.data.to_excel(
            result_dir /
            f"{cell.info.name}_{self._params.suffix}.xlsx",
            index=False
        )


class EpisodeAnalysisGroupedWorkflow:
    """
    Workflow for comparing spike type firing between episodes across groups.
    """

    def __init__(self, params: EpisodeAnalysisWorkflowParams):
        if params.output.path:
            self._output_dirs = self._setup_output_dirs(params.output.path)
        else:
            self._output_dirs = None
        self._params = params
        self._spike_types = params.analysis.spike_types
        self._episodes = params.analysis.episodes
        if params.grouping.aggregation == 'mean':
            self._aggregate = np.nanmean
        elif params.grouping.aggregation == 'median':
            self._aggregate = np.nanmedian
        else:
            raise ValueError("Invalid aggregation method.")

    def run(self, group: list[CellWrapper[DataFrame]], group_name: str):
        """Runs the group analysis on a given group of cells."""
        group_results = self._aggregate_cells(group)
        fig = self._plot_results(group_results, group_name)
        self._save_results(group_results, group_name)
        self._save_figure(fig, group_name)

    def _aggregate_cells(self, group: list[CellWrapper[DataFrame]]):
        """Runs the group analysis on a given group of cells."""

        cellwise_aggregates = []
        for cell in group:
            info = cell.info
            frame = cell.data
            cell_dict = {'name': info.name, 'group': info.cell_type}
            episodes_aggregates = []
            for episode in self._episodes:
                subset = frame[frame['episode'] == episode]
                if subset.shape[0] == 0:
                    episode_result = {
                        spike_type: np.nan
                        for spike_type in self._spike_types
                    }
                else:
                    episode_result = {
                        spike_type: self._aggregate(
                            subset[spike_type] / subset['duration']
                        )
                        for spike_type in self._spike_types
                    }
                episodes_aggregates.append(
                    cell_dict | {'episode': episode} | episode_result
                )
            cellwise_aggregates.append(pd.DataFrame(episodes_aggregates))
        group_results = pd.concat(cellwise_aggregates)
        group_results['episode'] = pd.Categorical(
            group_results['episode'], categories=self._episodes, ordered=True
        )
        return group_results

    def _plot_results(
        self, group_results: DataFrame, group_name: str
    ) -> Figure:
        """Plot the results of the group analysis."""
        n_spike_types = len(self._spike_types)
        fig, axes = plt.subplots(1, n_spike_types, figsize=(8, 6), sharey=True)
        for ax, spike_type in zip(axes, self._spike_types):
            spike_color = self._get_spike_color(spike_type, group_name)
            sns.boxplot(
                x='episode',
                y=spike_type,
                fill=False,
                fliersize=0,
                color='#ADADAD',
                data=group_results[~group_results[spike_type].isna()],
                ax=ax
            )
            for name in group_results['name'].unique():
                subset = group_results[group_results['name'] == name]
                ax.plot(
                    subset['episode'],
                    subset[spike_type],
                    color=spike_color,
                    markersize=10,
                    marker='o',
                    linestyle='-',
                    label=name,
                    linewidth=2,
                )

            ax.set_title(spike_type)
            ax.set_ylabel("Mean firing rate (Hz)")
            ax.set_xlabel("Episode")
            # if len(self._episodes) == 2:
            #     res = wilcoxon(
            #         group_results[group_results['episode'] == self._episodes[0]][spike_type],
            #         group_results[group_results['episode'] == self._episodes[1]][spike_type]
            #     )
            #     stat_result = (
            #         f"{self._episodes[0]} vs. {self._episodes[1]}, " +
            #         f"p={res.pvalue:.3f}"  # type: ignore
            #     )
            # else:
            #     _, kruskal_p = kruskal(
            #         *[
            #             group_results[group_results['episode'] == ep][spike_type].dropna()
            #             for ep in self._episodes
            #         ]
            #     )
            #     stat_result = f"Kruskal-Wallis p={kruskal_p:.3f}"

            # res = wilcoxon(
            #     group_results[group_results['episode'] == 'sw'][spike_type],
            #     group_results[group_results['episode'] == 'theta'][spike_type]
            # )

        plt.suptitle(group_name)
        plt.tight_layout()
        return fig

    def _save_figure(self, fig: Figure, group_name: str) -> None:
        """Save the figure of the group analysis."""
        if self._params.output.figures.save and self._output_dirs:
            figure_dir = self._output_dirs['figures']
            figure_name = f'{group_name}_{self._params.analysis.suffix}'
            if self._params.output.figures.as_png:
                fig.savefig(
                    figure_dir / f'{figure_name}.png',
                    dpi=self._params.output.figures.dpi,
                    format='png'
                )
            if self._params.output.figures.as_svg:
                fig.savefig(
                    figure_dir / f'{figure_name}.svg',
                    format='svg'
                )
            plt.close(fig)

    def _get_spike_color(self, spike_type: str, cell_type: str) -> str:
        """Get the color for a given spike type."""
        coloring = self._params.palette['spike_type'][spike_type]
        # return the defined color
        if coloring.startswith('#'):
            return coloring
        # return cell type based color
        if coloring == 'celltype':
            return self._params.palette['celltype'].get(
                cell_type, '#000000'
            )
        return '#000000'  # Default to black

    def _save_results(self, results: DataFrame, group_name: str) -> None:
        """Save the cellwise aggregate of a group."""
        if not self._output_dirs:
            return
        results.to_excel(
            self._output_dirs['results'] /
            f'{group_name}_cellwise_aggregate.xlsx', 
            index=False
        )

    def _setup_output_dirs(self, output_dir: str) -> dict[str, Path]:
        """
        Prepare the output directories for saving results.
        """
        dirs = {}
        for i, dir_name in enumerate(['results', 'figures', 'stats']):
            dirs[dir_name] = Path(output_dir) / f'{i+1:02d}_{dir_name}' / 'Grouped'
            dirs[dir_name].mkdir(parents=True, exist_ok=True)
        return dirs