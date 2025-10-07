

from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from common.data_utils import CellInfo, CellWrapper
from analyses.model import (
    RecordingWithSpikes,
    mark_events_around_trigger,
    mark_events_in_interval
)
from workflows.run_util.grouping import get_grouping
from workflows.run_util.io import (
    ResultPaths,
    RecordingWithSpikes,
    setup_result_dir
)
from workflows.run_util.plotting import CategoricalDistribution, create_stacked_area_chart


class EpisodeFiringBySpikeType:
    """
    An analysis class for the analysis of the relationship 
    between spike types and sleep episodes or sections.
    """

    def __init__(self, config: dict):
        self._output_dirs = setup_result_dir(config['output']['path'])
        self._episodes = config['params']['episodes']
        self._spike_types = config['params']['spike_types']
        self._plot_params = config['plot']
        self._palette = config['palette']
        if config['grouping']['do']:
            self._grouping = pd.read_excel(config['grouping']['path'])

    def run(self, cell: CellWrapper[ResultPaths]) -> CellWrapper[DataFrame]:
        """
        Run the spike type distribution analysis.
        """
        # Load data
        data = cell.data.load()
        # Calculate firing rate during episodes
        result_list = []
        for episode in self._episodes:
            try:
                result_list.append(self._process_episode(data, episode))
            except KeyError:
                print(
                    f"[INFO] Episode '{episode}' is missing in {cell.info.name}."
                )
                continue
        result = pd.concat(result_list, ignore_index=True)
        result.to_excel(
            self._output_dirs['results'] /
            f"{cell.info.name}_episode_firing_rate_by_spiketype.xlsx",
            index=False
        )
        return CellWrapper(cell.info, result)

    def run_grouped(self, results: list[CellWrapper[DataFrame]]):
        """
        Aggregate the results from multiple cells into a single DataFrame.
        """
        groups = get_grouping(self._grouping, results)
        for group in groups:
            aggregate = pd.concat([
                member.data.groupby(['episode']).mean().reset_index()
                for member in group.cells
            ], ignore_index=True)

            self._plot_results_grouped(group.name, aggregate)

            if not self._output_dirs['stats']:
                raise RuntimeError("Output directory not set.")
            group_dir = self._output_dirs['stats'] / 'grouped'
            group_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            aggregate.to_csv(
                group_dir / f"group_{group.name:02d}_spike_episode_stats.csv",
                index=False
            )

    def _process_episode(self, data: RecordingWithSpikes, name: str) -> DataFrame:
        """
        Calculates the mean firing rate of each spike type
        during each interval of the given type of episode.
        """
        spikes = data.spikes.copy()
        timestamps = spikes['timestamp'].values
        starts, ends = data.get_episode_ends(name)
        for start, end in zip(starts, ends):
            duration = end - start
            is_within = (timestamps >= start) & (timestamps <= end)
            result = {
                'episode': name,
                'start': start,
                'end': end,
                'duration': duration
            }
            for spike_type in self._spike_types:
                result[spike_type] = np.sum(
                    is_within & (spikes['spike_type'] == spike_type)
                ) / duration
        return DataFrame(result, index=[0])

    def _plot_results_individual(self, info: CellInfo, stats: DataFrame):
        """
        Plots the ratios of spike types across different sleep episodes.
        """
        if not self._output_dirs['figures']:
            raise RuntimeError("Output directory not set.")
        group_dir = self._output_dirs['figures'] / info.cell_type
        group_dir.mkdir(parents=True, exist_ok=True)
        self._make_plot(stats, group_dir.as_posix(), info.name)

    def _plot_results_grouped(self, group_name: str, stats: DataFrame):
        """
        Plots the grouped results for each spike type.
        """
        if not self._output_dirs['figures']:
            raise RuntimeError("Output directory not set.")
        group_dir = self._output_dirs['figures'] / 'grouped'
        group_dir.mkdir(parents=True, exist_ok=True)
        self._make_plot(stats, group_dir.as_posix(), group_name)

    def _make_plot(self, statistics: DataFrame, path: str, title: str):
        stats = statistics.copy()
        for state_group_name in stats['group'].unique():
            group_stats = stats[stats['group'] == state_group_name]
            state_type = group_stats['type'].iloc[0]

            if state_type == 'trigger':
                group_name = group_stats['group'].iloc[0]
                states = [
                    f'yes_{group_name}',
                    f'no_{group_name}'
                ]
                palette = [
                    self._palette['trigger'][group_name],
                    self._palette['trigger']['none']
                ]
                stats['state'] = pd.Categorical(
                    stats['state'],
                    categories=states,
                    ordered=True
                )
            else:
                states = []
                palette = []
                for episode_group in self._episodes:
                    if episode_group['name'] == state_group_name:
                        for episode_name in episode_group['episodes'] + ['none']:
                            states.append(episode_name)
                            palette.append(self._palette['episode'][episode_name])
                        break

            plot_data = []
            for spike_type in self._spike_types:
                values = []
                spike_stats = group_stats[group_stats['spike_type'] == spike_type]
                for state in states:
                    state_value = spike_stats[spike_stats['state'] == state]['ratio'].values
                    values.append(state_value)
                plot_data.append((
                    spike_type,
                    CategoricalDistribution(
                        np.array(values),
                        states,
                        state_group_name
                    )
                ))
            fig, ax = plt.subplots(figsize=self._plot_params['figsize'])
            create_stacked_area_chart(
                plot_data,
                ax,
                params={
                    'figsize': (10, 6),
                    'palette': palette
                }
            )
            plt.xlabel('Spike Type')
            plt.ylabel('Ratio')
            plt.tight_layout()
            plt.title(title)

            fig.savefig(
                Path(path) /
                f"{title}_spiketype_{state_group_name}_ratio.png",
                dpi=100
            )
            plt.close(fig)
