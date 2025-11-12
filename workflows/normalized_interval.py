

import functools
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from pandas import DataFrame

from analyses.model import RecordingWithSpikes, Interval
from common.data_utils import CellWrapper
from param_classes.workflows.normalized_interval_firing import (
    NormalizedIntervalFiringWorkflowParams
)
from triggered_firing.calculators import MultiTrialHistogram, select_spikes

from workflows.run_util.io import ResultPaths
from workflows.result_classes import EventSequenceList, IntervalRelatedEvents, FiringAnalysisResults

SpikeTypeGroupedResults = dict[str, FiringAnalysisResults]

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: NormalizedIntervalFiringWorkflowParams
) -> list[CellWrapper[SpikeTypeGroupedResults]]:
    """Run the normalized inter RA-AP firing workflow on the given input files."""
    analysis = NormalizedIntervalFiringWorkflow(params)
    return [analysis(input_file) for input_file in input_files]

def run_grouped_analysis(
    cell_results: list[CellWrapper[SpikeTypeGroupedResults]],
    group_name: str,
    params: NormalizedIntervalFiringWorkflowParams
) -> tuple[str, DataFrame]:
    """Run the cell-wise aggregation workflow on the given input files."""
    aggregator = NormalizedIntervalGroupAggregatorWorkflow(params)
    aggregation = aggregator(cell_results, group_name)
    return (group_name, aggregation)


class NormalizedIntervalFiringWorkflow:
    """
    Workflow for analyzing normalized inter RA-AP interval firing of AIS-AP.
    """
    def __init__(self, config: NormalizedIntervalFiringWorkflowParams):
        self._palette = config.palette

        params = config.analysis
        self._episode_name = params.episode_name
        self._spike_types = params.spike_types
        self._smooth_size = params.smooth_size
        self._cut_window = Interval(
            params.bin_range[0], params.bin_range[1]
        )
        self._bins = np.arange(
            params.bin_range[0],
            params.bin_range[1] + params.bin_size,
            params.bin_size
        )
        self._centers = np.mean(
            [self._bins[:-1], self._bins[1:]], axis=0
        )
        self._bin_size = params.bin_size
        self._value_unit = params.values
        self._ylim = params.ylim
        self._xlim = params.bin_range

        self._make_histogram = functools.partial(
            MultiTrialHistogram,
            centers=self._centers,
            data_unit='',
            value_unit=self._value_unit
        )
        self._extend_interval = functools.partial(
            Interval.shift_relative,
            left=self._cut_window.start,
            right=self._cut_window.end - 1.0
        )

        self._condition = params.interval_condition

        self._show_raster = params.show_raster
        self._raster_size = params.raster_size
        self._raster_band_color = params.raster_band_color

        self._paths = None
        if config.output.path:
            self._paths = {
                name: Path(config.output.path) / dir_name
                for name, dir_name in zip(
                    ['results', 'figures'], ['01_results', '02_figures']
                )
            }
            for path in self._paths.values():
                path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, cell: CellWrapper[ResultPaths]
    ) -> CellWrapper[SpikeTypeGroupedResults]:
        """Process a single input file and return the results."""
        results = self._calculate_in_episode(
            cell.data.load(), self._episode_name
        )
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate_in_episode(
        self, cell_data: RecordingWithSpikes, episode_name: str
    ) -> SpikeTypeGroupedResults:
        """Calculate normalized inter RA-AP interval firing for a single episode."""
        intervals = cell_data.get_episode_intervals(episode_name)
        return {
            spike_type: self._calculate_on_subset(
                select_spikes(cell_data.spikes, spike_type), intervals
            ) for spike_type in self._spike_types
        }

    def _calculate_on_subset(
        self, spikes: np.ndarray, intervals: list[Interval]
    ) -> FiringAnalysisResults:
        """Calculate firing rate in the given sequence of intervals."""
        values = []
        events = []
        for interval in intervals:
            cut_events = self._collect_normalized_events(spikes, interval)
            values.append(self._count_in_interval(cut_events))
            events.append(cut_events.sequence)
        return FiringAnalysisResults(
            self._make_histogram(values=np.stack(values, axis=1)),
            EventSequenceList(events)
        )

    def _collect_normalized_events(
        self, spikes: np.ndarray, interval: Interval
    ) -> IntervalRelatedEvents:
        """Collect normalized spike times within the given interval."""
        cutting_interval = self._extend_interval(interval)
        spikes_in_interval = cutting_interval.contained(spikes)
        normalized_spikes = interval.normalize_by(spikes_in_interval)
        return IntervalRelatedEvents(interval, normalized_spikes)

    def _count_in_interval(
        self, spikes_within: IntervalRelatedEvents
    ) -> np.ndarray:
        """Count the number of spikes within the given interval."""
        counts = np.histogram(spikes_within.sequence, bins=self._bins)[0]
        if self._value_unit == 'freq':
            return counts / (spikes_within.interval.duration * self._bin_size)
        if self._value_unit == 'prob':
            total_spikes = len(spikes_within.sequence)
            if total_spikes > 0:
                return counts / total_spikes
        return counts

    def _plot(
        self, cell_result: CellWrapper[SpikeTypeGroupedResults]
    ):
        """Plot the results."""

        spike_colors = self._palette['spike_type'] | {
            'AIS-AP': self._palette['cell_type'][cell_result.info.cell_type]
        }

        if self._show_raster:
            fig, axes = plt.subplots(
                2, 1, figsize=(8, 6),
                constrained_layout=True,
                sharex=True
            )

            raster_offset = self._raster_size / 2
            raster_ax = axes[0]
            hist_ax = axes[1]

            for i, events in enumerate(cell_result.data['RA-AP'].sequences.sequences):
                if len(events) == 0:
                    continue
                raster_ax.fill_between(
                    self._xlim,
                    [i - raster_offset, i - raster_offset],
                    [i + raster_offset, i + raster_offset],
                    color=self._raster_band_color,
                    alpha=0.1,
                    edgecolor='none'
                )

            for spike_type, result in cell_result.data.items():
                spike_color = spike_colors[spike_type]
                for i, events in enumerate(result.sequences.sequences):
                    raster_ax.vlines(
                        events,
                        i - raster_offset,
                        i + raster_offset,
                        color=spike_color
                    )
            raster_ax.set_ylim((
                -raster_offset,
                raster_offset + len(cell_result.data['RA-AP'].sequences.sequences)
            ))
            raster_ax.set_yticks([])
            raster_ax.set_ylabel('')
        else:
            fig, hist_ax = plt.subplots(1, 1, figsize=(8, 3))

        n_all_trials = list(cell_result.data.items())[0][1].histograms.n_trials
        valid_trial_flag = np.ones(n_all_trials, dtype=bool)
        for condition_spike_type in self._condition:
            spike_type_result = cell_result.data[condition_spike_type]
            for i, sequence in enumerate(spike_type_result.sequences.sequences):
                if np.sum((0 <= sequence) & (sequence <= 1)) == 0:
                    valid_trial_flag[i] = False

        for spike_type, result in cell_result.data.items():
            spike_color = spike_colors[spike_type]

            hist: MultiTrialHistogram = result.histograms
            centers = hist.centers
            values = hist.values[:, valid_trial_flag]
            mean_values = np.mean(values, axis=1)
            sd_values = np.std(values, axis=1)
            sem_values = sd_values / np.sqrt(values.shape[1])
            hist_ax.fill_between(
                centers,
                mean_values - sem_values,
                mean_values + sem_values,
                color=spike_color,
                alpha=0.3
            )
            hist_ax.plot(
                centers,
                mean_values,
                '-',
                color=spike_color,
                label=spike_type
            )

        if self._value_unit == 'count':
            hist_ax.set_ylabel('Avg spike count')
        elif self._value_unit == 'freq':
            hist_ax.set_ylabel('Firing (Hz)')
        elif self._value_unit == 'prob':
            hist_ax.set_ylabel('Spike distribution')
        hist_ax.set_ylim(self._ylim)

        hist_ax.set_xlabel('Normalized time')
        hist_ax.set_xlim(self._xlim)

        hist_ax.legend()

        hist_ax.set_title(
            f'n={np.sum(valid_trial_flag)}/{n_all_trials}'
        )
        plt.suptitle(
            f'{cell_result.info.name} Normalized {self._episode_name} firing'
        )
        if self._paths:
            fig_path = (
                self._paths['figures'] /
                f'{cell_result.info.name}_normalized_{self._episode_name}_firing'
            )
            fig.savefig(
                f'{fig_path}.png', format='png', dpi=300
            )
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(
                    f'{fig_path}.svg', format='svg'
                )
            plt.close(fig)
        else:
            plt.show()

    def _save_result(self, cell_result: CellWrapper[SpikeTypeGroupedResults]):
        """Save the results to file."""
        if not self._paths:
            return
        result_path = (
            self._paths['results'] /
            f'{cell_result.info.name}_normalized_{self._episode_name}_firing.xlsx'
        )
        pool_frame = convert_episode_result_to_frame(cell_result.data)
        pool_frame.to_excel(result_path, index=False)


class NormalizedIntervalGroupAggregatorWorkflow:
    """Class to aggregate peri-trigger time histogram results across multiple cells."""

    def __init__(self, params: NormalizedIntervalFiringWorkflowParams):
        self._palette = params.palette
        self._spike_types = params.analysis.spike_types
        self._interval_condition = params.analysis.interval_condition
        self._xlim = params.analysis.bin_range
        self._ylim = params.analysis.ylim
        self._suffix = (
            'normalized' +
            f'_{params.analysis.episode_name}' +
            '_mean'
        )
        self._paths = {
            'results': Path(params.output.path) / '01_results' / 'grouped',
            'cellwise_figures': Path(params.output.path) / '02_figures' / 'individual',
            'groupwise_figures': Path(params.output.path) / '02_figures' / 'groupwise'
        }
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        input_files: list[CellWrapper[SpikeTypeGroupedResults]],
        group_name: str
    ) -> DataFrame:
        """Aggregate results from multiple input files."""
        aggregation = self._aggregate_per_cell(input_files)
        aggregation.to_excel(
            self._paths['results'] / f'{group_name}_{self._suffix}_histograms.xlsx',
            index=False
        )
        groupwise_aggregate = self._groupwise_aggregate(aggregation)
        self._plot_grouped_detection(groupwise_aggregate, group_name)
        return aggregation

    def _aggregate_per_cell(
        self,
        cells: list[CellWrapper[SpikeTypeGroupedResults]]
    ) -> DataFrame:
        """Aggregate values within a single cell."""
        cell_frames = [
            CellWrapper(
                cell_result.info,
                convert_episode_result_to_frame(cell_result.data)
            ) for cell_result in cells
        ]
        values = {}
        for cell_frame in cell_frames:
            cell_values = self._get_values(cell_frame.data)
            cellwise_mean = np.mean(cell_values, axis=1)
            values[cell_frame.name] = cellwise_mean
        return DataFrame(
            {
                'spike_type': cell_frames[0].data['spike_type'],
                'centers': cell_frames[0].data['centers'],
                **values
            }
        )

    def _get_values(self, frame: DataFrame) -> np.ndarray:
        """Filter the DataFrame based on the trigger condition."""
        centers = frame['centers']
        all_values = frame.iloc[:, 2:].to_numpy()
        if not self._interval_condition:
            return all_values
        spike_type_vector = frame['spike_type']
        flag = np.ones(all_values.shape[1], dtype=bool)
        for condition in self._interval_condition:
            mask = (
                (spike_type_vector == condition) &
                (0 <= centers) & (centers <= 1)
            )
            values = all_values[mask, :]
            flag &= (np.sum(values, axis=0) > 0)
        return all_values[:, flag]

    def _groupwise_aggregate(self, cells: DataFrame) -> DataFrame:
        """Aggregate values"""
        values = cells.iloc[:, 2:].to_numpy()
        mean_values = np.mean(values, axis=1)
        sd_values = np.std(values, axis=1)
        sem_values = sd_values / np.sqrt(values.shape[1])
        return DataFrame({
            'spike_types': cells['spike_type'],
            'centers': cells['centers'],
            'mean': mean_values,
            'sem': sem_values
        })

    def _plot_grouped_detection(
        self,
        grouped_data: DataFrame,
        group_name: str
    ):
        """Plots the group average histogram per spike-type and the detected points."""

        spike_colors = self._palette['spike_type'] | {
            'AIS-AP': self._palette['cell_type'][group_name]
        }

        fig, ax = plt.subplots(
            1, 1, figsize=(8, 3), constrained_layout=True
        )
        for spike_type in self._spike_types:
            spike_data = grouped_data[
                grouped_data['spike_types'] == spike_type
            ]
            color = spike_colors[spike_type]

            ax.fill_between(
                spike_data['centers'],
                spike_data['mean'] - spike_data['sem'],
                spike_data['mean'] + spike_data['sem'],
                color=color,
                alpha=0.3
            )
            ax.plot(
                spike_data['centers'],
                spike_data['mean'],
                color=color,
                label=spike_type
            )

        ax.set_ylim(self._ylim)
        ax.set_xlim(self._xlim)
        ax.set_xlabel('Normalized time')
        ax.legend()
        plt.suptitle(group_name)

        figure_name = f'{group_name}_{self._suffix}'
        fig.savefig(
            self._paths['groupwise_figures'] / f'{figure_name}.png',
            dpi=100,
            format='png'
        )
        with plt.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(
                self._paths['groupwise_figures'] / f'{figure_name}.svg',
                format='svg'
            )
        plt.close(fig)


def convert_episode_result_to_frame(
    results: SpikeTypeGroupedResults
) -> pd.DataFrame:
    """Convert the episode result to a DataFrame."""
    frames = []
    for spike_type, result in results.items():
        hist: MultiTrialHistogram = result.histograms
        column_names = [f'interval_{i+1}' for i in range(hist.n_trials)]
        df = pd.DataFrame(
            hist.values,
            columns=column_names
        )
        df.insert(0, 'centers', hist.centers)
        df.insert(0, 'spike_type', spike_type)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
