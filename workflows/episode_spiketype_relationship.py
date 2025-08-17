

from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from common.data_utils import CellInfo, CellWrapper
from analyses.model import (
    RecordingWithSpikes,
    mark_events_around_trigger
)
from workflows.run_util.grouping import get_grouping
from workflows.run_util.io import (
    ResultPaths,
    RecordingWithSpikes
)
from workflows.run_util.plotting import CategoricalDistribution, create_stacked_area_chart


class SpikeEpisodeProbabilityAnalysis:
    """
    An analysis class for the analysis of the relationship 
    between spike types and sleep episodes or sections.
    """

    def __init__(self, config: dict):
        self._output_path = config['output']['path']

        self._episode_groups = config['params']['episode_groups']
        self._trigger_windows = config['params'].get('trigger_windows', [])
        self._spike_types = config['params']['spike_types']

        self._plot_params = config['plot']
        self._palette = config['palette']

        self._grouping = pd.read_excel(
            config['grouping']['path']
        )

        self._output_dirs = {}
        self._prepare_output_dirs()

    def _prepare_output_dirs(self):
        """
        Prepare the output directories for saving results.
        """
        output_path = Path(self._output_path)
        for i, dir_type in enumerate(['results', 'figures', 'stats']):
            self._output_dirs[dir_type] = output_path / f"{i+1:02d}_{dir_type}"
            self._output_dirs[dir_type].mkdir(parents=True, exist_ok=True)

    def run(self, cell: CellWrapper[ResultPaths]) -> CellWrapper[DataFrame]:
        """
        Run the spike type distribution analysis.
        """
        # Load data
        data = cell.data.load()
        
        # Calculate spike type probabilities for marked episodes
        for episode_group in self._episode_groups:
            data = self._process_episode_group(data, episode_group)

        # Calculate spike type probabilities for triggered sections
        for trigger_window in self._trigger_windows:
            data = self._process_trigger_windows(data, trigger_window)

        stats = self._calculate_statistics(data.spikes)

        self._save_stats(cell.info, stats)
        self._save_assignments(cell.info, data.spikes)
        self._plot_results_individual(cell.info, stats)

        return CellWrapper(cell.info, stats)

    def run_grouped(self, results: list[CellWrapper[DataFrame]]):
        """
        Aggregate the results from multiple cells into a single DataFrame.
        """
        groups = get_grouping(self._grouping, results)
        for group in groups:
            pooled = pd.concat([member.data for member in group.cells])
            aggregate = pooled.groupby([
                'type', 'group', 'spike_type', 'state'
            ], observed=True).mean().reset_index()

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


    def _process_episode_group(
        self,
        data: RecordingWithSpikes,
        group: dict
    ) -> RecordingWithSpikes:
        """
        Classifies each spike according to its membership in any of the 
        episode of the specified episode group. Membership of the spikes
        outside of the interval of interest are left empty. Spikes within 
        the specified interval, that belong to none of the episodes are 
        classified as 'none'.
        """
        # Create the column for the episode group
        spikes = data.spikes.copy()
        spikes[group['name']] = None

        # Find events in the sample space
        sample_space_name = group.get('parent', 'all')
        is_analzed = data.mark_episode_events(sample_space_name)
        spikes.loc[is_analzed, group['name']] = 'none'

        # Find events in each episode
        episode_flags = {
            episode_name: is_analzed & data.mark_episode_events(episode_name)
            for episode_name in group['episodes']
        }

        # Count summary for each spike
        for spike_type in self._spike_types:
            has_spike_type = spikes['spike_type'] == spike_type
            for episode_name, is_in_episode in episode_flags.items():
                is_selected = has_spike_type & is_in_episode
                spikes.loc[is_selected, group['name']] = episode_name

        return RecordingWithSpikes(data.recording, spikes)

    def _process_trigger_windows(
        self,
        data: RecordingWithSpikes,
        params: dict
    ) -> RecordingWithSpikes:
        """
        Classifies each spike according to its membership in any of the
        trigger windows specified in the trigger dictionary.
        """
        cut_window = params['window']
        if params['time_unit'] == 'ms':
            cut_window = [x / 1000 for x in cut_window]
        marker_name = params['name']

        # Initialize the grouping column
        spikes = data.spikes.copy()
        spikes[marker_name] = f'no_{marker_name}'

        # Create trigger windows for each marker
        markers = data.get_markers(marker_name)

        # Mark spikes according to window membership
        flag = mark_events_around_trigger(spikes, markers, cut_window)
        spikes.loc[flag, marker_name] = f'yes_{marker_name}'

        return RecordingWithSpikes(data.recording, spikes)

    def _calculate_statistics(self, data: DataFrame) -> DataFrame:
        """
        Calculates the spike distribution across different episode or trigger
        types for each spike type.
        """
        results = []
        for episode_group in self._episode_groups:
            group_name = episode_group['name']
            in_group = data[group_name].notnull()
            for spike_type in self._spike_types:
                has_spike_type = (data['spike_type'] == spike_type) & in_group
                for episode_name in episode_group['episodes'] + ['none']:
                    in_episode = (data[group_name] == episode_name) & has_spike_type
                    ratio = (
                        in_episode.sum() / has_spike_type.sum()
                        if has_spike_type.sum() > 0 else 0
                    )
                    results.append({
                        'type': 'episode',
                        'group': group_name,
                        'spike_type': spike_type,
                        'state': episode_name,
                        'ratio': ratio
                    })

        for trigger in self._trigger_windows:
            for spike_type in self._spike_types:
                has_spike_type = data['spike_type'] == spike_type
                for state_name in [f'yes_{trigger["name"]}', f'no_{trigger["name"]}']:
                    in_state = (data[trigger["name"]] == state_name) & has_spike_type
                    ratio = (
                        in_state.sum() / has_spike_type.sum()
                        if has_spike_type.sum() > 0 else 0
                    )
                    results.append({
                        'type': 'trigger',
                        'group': trigger['name'],
                        'spike_type': spike_type,
                        'state': state_name,
                        'ratio': ratio
                    })
        return DataFrame(results)

    def _save_assignments(self, info: CellInfo, data: DataFrame):
        """
        Save the spike assignments to the output path.
        """
        if not self._output_dirs['results']:
            raise RuntimeError("Output directory not set.")
        group_dir = self._output_dirs['results'] / info.cell_type
        group_dir.mkdir(parents=True, exist_ok=True)

        # Select columns to write
        output_table = data[
            ['timestamp', 'spike_type'] +
            [group['name'] for group in self._episode_groups] +
            [trigger['name'] for trigger in self._trigger_windows]
        ]

        # Save to file
        output_table.to_csv(
            group_dir / 
            f'{info.name}_episode_membership.csv',
            index=False
        )

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
                for episode_group in self._episode_groups:
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

    def _save_stats(self, info: CellInfo, stats: DataFrame):
        """
        Save the spike statistics to the output path.
        """
        if not self._output_dirs['stats']:
            raise RuntimeError("Output directory not set.")
        group_dir = self._output_dirs['stats'] / info.cell_type
        group_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        stats.to_csv(
            group_dir / f"{info.name}_spike_episode_stats.csv",
            index=False
        )
