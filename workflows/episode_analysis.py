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
from statsmodels.stats.multitest import multipletests

from matplotlib.figure import Figure

from common.data_utils import CellWrapper, GroupInfo, GroupWrapper
from analyses.model import (
    RecordingWithSpikes
)
from param_classes.workflows.episode_analysis import EpisodeAnalysisWorkflowParams, GroupwiseComparisonParams
from workflows.run_util.io import (
    ResultPaths,
    RecordingWithSpikes
)
from workflows.run_util.plotting import CategoricalDistribution, create_stacked_area_chart

def describe_episodes(
    cells: list[CellWrapper[ResultPaths]],
    params: EpisodeAnalysisWorkflowParams
) -> list[CellWrapper[DataFrame]]:
    """Run the spike type vs. sleep episode relationship analysis."""
    analysis = EpisodeFiringBySpikeType(params)
    return [analysis.run(input_file) for input_file in tqdm(cells)]

def create_groups(cells: list[CellWrapper]) -> list[GroupWrapper[CellWrapper]]:
    """Create groups based on cell types."""
    groups = {}
    for cell in cells:
        group_name = cell.info.cell_type
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(cell)
    return [GroupWrapper(GroupInfo(name), members) for name, members in groups.items()]

def run_comparison(
    groups: list[GroupWrapper[CellWrapper[DataFrame]]],
    params: GroupwiseComparisonParams
):
    """Run the group comparison of spike type vs. sleep episode relationship."""
    comparison = BetweenEpisodeComparisonWorkflow(params)
    for group in tqdm(groups, desc='Process groups'):
        members = [
            CellWrapper(member.info, pd.read_excel(member.data))
            for member in group.members
        ]
        comparison.run(GroupWrapper(group.info, members))

def run_spike_type_comparison(
    groups: list[GroupWrapper[CellWrapper[DataFrame]]],
    params: GroupwiseComparisonParams
):
    """Run the group comparison of spike type vs. sleep episode relationship."""
    comparison = BetweenSpikeTypeInEpisodeComparisonWorkflow(params)
    for group in tqdm(groups, desc='Process groups'):
        members = [
            CellWrapper(member.info, pd.read_excel(member.data))
            for member in group.members
        ]
        comparison.run(GroupWrapper(group.info, members))

def plot_episode_durations(
    groups: list[GroupWrapper[CellWrapper[DataFrame]]],
    params: GroupwiseComparisonParams
):
    """Creates and saves episode duration plots for each group."""
    plotting = EpisodeDurationPlotter(params)
    for group in tqdm(groups, desc='Plot groups'):
        members = [
            CellWrapper(member.info, pd.read_excel(member.data))
            for member in group.members
        ]
        plotting.run(GroupWrapper(group.info, members))

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

        self._params = params.analysis
        self._interval_type = self._params.interval_type

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

NamedCategorical = tuple[str, CategoricalDistribution]

class EpisodeDurationPlotter:
    """
    A class for plotting the durations of episodes per cell.
    """

    def __init__(self, params: GroupwiseComparisonParams):
        self._output_dir = (
            Path(params.data.results.path) /
            '02_figures' /
            'Individual'
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._suffix = params.analysis.suffix
        self._episodes = params.analysis.episodes
        self._colors = [
            params.palette['episode'][episode] for episode in self._episodes
        ]
        self._figure_params = params.output.figures

    def run(self, group: GroupWrapper[CellWrapper[DataFrame]]) -> None:
        """Run the episode duration plotting."""
        plot_data = [
            self._summarize_episodes(member) for member in group.members
        ]
        fig = self._create_plot(plot_data, group.name)
        self._save_figure(fig, group.name)

    def _create_plot(self, data: list[NamedCategorical], group_name: str) -> Figure:
        """Create the episode duration plot."""
        fig, ax = plt.subplots(figsize=(len(data) * 2, 4))
        create_stacked_area_chart( data, ax, {'palette': self._colors})
        ax.set_ylabel("Duration (s)")
        ax.set_xlabel("Cells")
        ax.set_title(group_name)
        plt.tight_layout()
        return fig

    def _summarize_episodes(self, cell: CellWrapper[DataFrame]) -> NamedCategorical:
        """Summarize the episodes in the given DataFrame."""
        values = np.array([
            self._aggregate_episode(cell.data, episode)
            for episode in self._episodes
        ])
        return (
            cell.name,
            CategoricalDistribution(values, self._episodes, self._suffix)
        )

    def _aggregate_episode(self, frame: DataFrame, episode: str) -> float:
        """Aggregate the results of a single episode."""
        return np.sum(frame[frame['episode'] == episode]['duration'])

    def _save_figure(self, fig: Figure, group_name: str) -> None:
        """Save the figure of the group analysis."""
        figure_name = f'{group_name}_cellwise_{self._suffix}_durations'
        if self._figure_params.as_png:
            fig.savefig(
                self._output_dir / f'{figure_name}.png',
                dpi=self._figure_params.dpi,
                format='png'
            )
        if self._figure_params.as_svg:
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(
                    self._output_dir / f'{figure_name}.svg',
                    format='svg'
                )
        plt.close(fig)


class BetweenEpisodeComparisonWorkflow:
    """
    A workflow class that performs statistical comparison of firing rates of 
    different spike types between sleep episodes.
    """

    def __init__(self, params: GroupwiseComparisonParams):
        self._output_dirs = self._setup_output_dirs(params.data.results.path)
        self._params = params
        self._spike_types = params.analysis.spike_types
        self._episodes = params.analysis.episodes
        if params.grouping.aggregation == 'mean':
            self._aggregate = np.nanmean
        elif params.grouping.aggregation == 'median':
            self._aggregate = np.nanmedian
        else:
            raise ValueError("Invalid aggregation method.")

    def run(self, group: GroupWrapper[CellWrapper[DataFrame]]):
        """Runs the group analysis on a given group of cells."""
        group_results = self._aggregate_cells(group.members)
        fig = self._plot_results(group_results, group.name)
        stat_results = self._perform_statistical_comparison(group_results)
        self._save_statistics(stat_results, group.name)
        self._save_results(group_results, group.name)
        self._save_figure(fig, group.name)

    def _aggregate_cells(self, group: list[CellWrapper[DataFrame]]) -> DataFrame:
        """Runs the group analysis on a given group of cells."""
        # TODO Move this to a separate class
        cellwise_aggregates = []
        for cell in group:
            heading = {
                'name': cell.info.name, 'group': cell.info.cell_type
            }
            episodes_aggregates = [
                heading | self._aggregate_episode(cell.data, episode)
                for episode in self._episodes
            ]
            cellwise_aggregates.append(pd.DataFrame(episodes_aggregates))

        group_results = pd.concat(cellwise_aggregates)
        group_results['episode'] = pd.Categorical(
            group_results['episode'], categories=self._episodes, ordered=True
        )
        return group_results

    def _aggregate_episode(self, frame: DataFrame, episode: str) -> dict:
        """Aggregate the results of a single episode."""
        frame = frame[frame['episode'] == episode]
        if frame.shape[0] == 0:
            aggregates = {spike_type: np.nan for spike_type in self._spike_types}
        else:
            aggregates = {
                spike_type: self._aggregate(frame[spike_type] / frame['duration'])
                for spike_type in self._spike_types
            }
        return {'episode': episode, **aggregates}

    def _perform_statistical_comparison(self, results: DataFrame) -> DataFrame:
        """Perform statistical comparison between episodes."""
        stat_results = []
        if len(self._episodes) == 2:
            for spike_type in self._spike_types:
                values  = [[], []]
                for name in results['name'].unique():
                    subset = results[results['name'] == name]
                    for i, episode in enumerate(self._episodes):
                        values[i].append(
                            subset[subset['episode'] == episode][spike_type].values
                        )
                res = wilcoxon(values[0], values[1])
                stat_results.append({
                    'spike_type': spike_type,
                    'episode_1': self._episodes[0],
                    'episode_2': self._episodes[1],
                    'difference': np.nanmedian(
                        np.array(values[0]) - np.array(values[1])
                    ),
                    'statistic': res.statistic,  # type: ignore
                    'p_value': res.pvalue  # type: ignore
                })
        else:
            for spike_type in self._spike_types:
                _, kruskal_p = kruskal(
                    *[
                        results[results['episode'] == ep][spike_type].dropna()
                        for ep in self._episodes
                    ]
                )
                stat_results.append({
                    'spike_type': spike_type,
                    'kruskal_p': kruskal_p,
                    **{
                        ep: np.nanmedian(
                            results[results['episode'] == ep][spike_type]
                        ) for ep in self._episodes
                    }
                })
        return DataFrame(stat_results)

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
                with plt.rc_context({'svg.fonttype': 'none'}):
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

    def _save_statistics(self, stats: DataFrame, group_name: str) -> None:
        """Save the statistical comparison results of a group."""
        if not self._output_dirs:
            return
        stats.to_excel(
            self._output_dirs['stats'] /
            f'{group_name}_statistical_comparison.xlsx', 
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


class BetweenSpikeTypeInEpisodeComparisonWorkflow:
    """
    A workflow class that performs statistical comparison of firing rates between
    different spike types within sleep episodes. Only those episode intervals are
    considered where both spike types are present.
    """

    def __init__(self, params: GroupwiseComparisonParams):
        self._output_dirs = self._setup_output_dirs(params.data.results.path)
        self._params = params
        self._spike_types = params.analysis.spike_types
        self._episodes = params.analysis.episodes
        self._condition = params.grouping.condition
        if params.grouping.aggregation == 'mean':
            self._aggregate = np.nanmean
        elif params.grouping.aggregation == 'median':
            self._aggregate = np.nanmedian
        else:
            raise ValueError("Invalid aggregation method.")

    def run(self, group: GroupWrapper[CellWrapper[DataFrame]]):
        """Runs the group analysis on a given group of cells."""
        group_results = self._aggregate_cells(group.members)
        self._save_results(group_results, group.name)

        fig = self._plot_results(group_results, group.name)
        self._save_figure(fig, group.name)

        stat_results = self._perform_statistical_comparison(group_results)
        self._save_statistics(stat_results, group.name)

    def _aggregate_cells(self, group: list[CellWrapper[DataFrame]]) -> DataFrame:
        """Runs the group analysis on a given group of cells."""
        # TODO Move this to a separate class
        cellwise_aggregates = []
        for cell in group:
            heading = {
                'name': cell.info.name, 'group': cell.info.cell_type
            }
            episodes_aggregates = [
                heading | self._aggregate_episode(cell.data, episode)
                for episode in self._episodes
            ]
            cellwise_aggregates.append(pd.DataFrame(episodes_aggregates))

        group_results = pd.concat(cellwise_aggregates)
        group_results['episode'] = pd.Categorical(
            group_results['episode'], categories=self._episodes, ordered=True
        )
        return group_results

    def _aggregate_episode(self, frame: DataFrame, episode: str) -> dict:
        """Aggregate the results of a single episode."""
        frame = frame[frame['episode'] == episode]
        if frame.shape[0] == 0:
            aggregates = {
                spike_type: np.nan for spike_type in self._spike_types
            } | {'ratio': np.nan}
        else:
            selected = np.ones(frame.shape[0]).astype(bool)
            for spike_type in self._condition:
                selected &= (frame[spike_type] > 0)

            frame = frame[selected]
            if frame.shape[0] == 0:
                aggregates = {
                    spike_type: np.nan for spike_type in self._spike_types
                } | {'ratio': 0.0}
            else:
                aggregates = {
                    spike_type: self._aggregate(frame[spike_type] / frame['duration'])
                    for spike_type in self._spike_types
                } | {'ratio': np.mean(selected)}
        return {'episode': episode, **aggregates}

    def _perform_statistical_comparison(self, results: DataFrame) -> DataFrame:
        """Perform statistical comparison between episodes."""
        stat_results = []
        print(results)
        for i, episode in enumerate(self._episodes):
            subset = results[results['episode'] == episode]
            values = [subset[spike_type].values for spike_type in self._spike_types]
            res = wilcoxon(values[0], values[1])

            stat_results.append({
                'episode': episode,
                'spike_type_1': self._spike_types[0],
                'spike_type_2': self._spike_types[1],
                'difference': np.nanmedian(
                    np.array(values[0]) - np.array(values[1])
                ),
                'statistic': res.statistic,  # type: ignore
                'p_value': res.pvalue  # type: ignore
            })

        return DataFrame(stat_results)

    def _plot_results(
        self, results: DataFrame, group_name: str
    ) -> Figure:
        """Plot the results of the group analysis."""
        n_episodes = len(self._episodes)
        fig, axes = plt.subplots(1, n_episodes, figsize=(8, 6), sharey=True)
        for ax, episode in zip(axes, self._episodes):
            subset = results.loc[
                results['episode'] == episode,
                ['name', 'ratio'] + self._spike_types
            ]
            long_df = pd.melt(
                subset,
                id_vars=['name', 'ratio'],
                value_vars=self._spike_types,
                var_name='spike_type',
                value_name='frequency'
            )
            long_df['spike_type'] = pd.Categorical(
                long_df['spike_type'],
                categories=self._spike_types,
                ordered=True
            )
            episode_color = self._get_episode_color(episode, group_name)
            sns.boxplot(
                x='spike_type',
                y='frequency',
                fill=False,
                fliersize=0,
                color='#ADADAD',
                data=long_df,
                ax=ax
            )
            for name in long_df['name'].unique():
                cell_subset = long_df[long_df['name'] == name]
                ax.plot(
                    cell_subset['spike_type'],
                    cell_subset['frequency'],
                    color=episode_color,
                    markersize=10,
                    marker='o',
                    linestyle='-',
                    label=name,
                    linewidth=2,
                )

            mean_ratio = (
                np.nanmean(subset['ratio'])
                if np.any(~np.isnan(subset['ratio']))
                else np.nan
            )
            ax.set_title(f"{episode} ratio: {mean_ratio:.2f}")
            ax.set_ylabel("Mean firing rate (Hz)")
            ax.set_xlabel("Spike type")

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
                with plt.rc_context({'svg.fonttype': 'none'}):
                    fig.savefig(
                        figure_dir / f'{figure_name}.svg',
                        format='svg'
                    )
            plt.close(fig)

    def _get_episode_color(self, episode: str, cell_type: str) -> str:
        """Get the color for a given episode."""
        coloring = self._params.palette['episode'][episode]
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

    def _save_statistics(self, stats: DataFrame, group_name: str) -> None:
        """Save the statistical comparison results of a group."""
        if not self._output_dirs:
            return
        stats.to_excel(
            self._output_dirs['stats'] /
            f'{group_name}_statistical_comparison.xlsx', 
            index=False
        )

    def _save_table(self, table: DataFrame, path: str) -> None:
        """Save the statistical comparison results of a group."""
        if not self._output_dirs:
            return
        table.to_excel(path, index=False)

    def _setup_output_dirs(self, output_dir: str) -> dict[str, Path]:
        """
        Prepare the output directories for saving results.
        """
        dirs = {}
        for i, dir_name in enumerate(['results', 'figures', 'stats']):
            dirs[dir_name] = Path(output_dir) / f'{i+1:02d}_{dir_name}' / 'Grouped'
            dirs[dir_name].mkdir(parents=True, exist_ok=True)
        return dirs
