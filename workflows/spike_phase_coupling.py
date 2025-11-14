

import functools
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from pandas import DataFrame

from scipy.ndimage import gaussian_filter1d

from analyses.model import RecordingWithSpikes, Interval
from common.data_utils import CellWrapper
from param_classes.workflows.spike_phase_coupling import (
    SpikePhaseCouplingWorkflowParams
)
from triggered_firing.calculators import MultiTrialHistogram, select_spikes

from workflows.run_util.io import ResultPaths
from workflows.result_classes import EventSequenceList, IntervalRelatedEvents, FiringAnalysisResults

SpikeTypeGroupedResults = dict[str, FiringAnalysisResults]

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: SpikePhaseCouplingWorkflowParams
) -> list[CellWrapper[SpikeTypeGroupedResults]]:
    """Run the normalized inter RA-AP firing workflow on the given input files."""
    analysis = SpikePhaseCouplingWorkflow(params)
    return [analysis(input_file) for input_file in input_files]

def run_grouped_analysis(
    cell_results: list[CellWrapper[SpikeTypeGroupedResults]],
    group_name: str,
    params: SpikePhaseCouplingWorkflowParams
) -> tuple[str, DataFrame]:
    """Run the cell-wise aggregation workflow on the given input files."""
    aggregator = SpikePhaseCouplingGroupAggregatorWorkflow(params)
    aggregation = aggregator(cell_results, group_name)
    return (group_name, aggregation)


class SpikePhaseCouplingWorkflow:
    """
    Workflow for analyzing normalized inter RA-AP interval firing of AIS-AP.
    """
    def __init__(self, config: SpikePhaseCouplingWorkflowParams):

        params = config.analysis

        self._episode_name = params.episode_name
        self._marker_name = params.marker_name
        self._spike_types = params.spike_types

        self._smooth = None
        if params.smooth_size > 0.0:
            self._smooth = functools.partial(
                gaussian_filter1d,
                sigma=params.smooth_size,
                axis=0
            )

        edges = np.linspace(0, 360, params.num_bins + 1)
        centers = np.mean([edges[:-1], edges[1:]], axis=0)
        self._num_bins = params.num_bins
        self._value_unit = params.values

        self._calculate_histogram = functools.partial(np.histogram, bins=edges)

        self._make_histogram = functools.partial(
            MultiTrialHistogram,
            centers=centers,
            data_unit='',
            value_unit=self._value_unit
        )

        self._condition = params.interval_condition

        self._palette = config.palette
        self._ylim = params.ylim
        self._xlim = params.xlim
        self._show_raster = params.show_raster
        self._raster_size = params.raster_size
        self._raster_band_color = params.raster_band_color
        self._y_label = {
            'count': 'Avg spike count',
            'freq': 'Firing (Hz)',
            'prob': 'Spike distribution'
        }[self._value_unit]


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
            cell.data.load(), self._episode_name, self._marker_name
        )
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate_in_episode(
        self,
        cell_data: RecordingWithSpikes,
        episode_name: str,
        marker_name: str
    ) -> SpikeTypeGroupedResults:
        """Calculate normalized inter RA-AP interval firing for a single episode."""
        cycles = self._collect_cycles(cell_data, episode_name, marker_name)
        return {
            spike_type: self._calculate_on_subset(
                select_spikes(cell_data.spikes, spike_type), cycles
            ) for spike_type in self._spike_types
        }

    def _calculate_on_subset(
        self, spikes: np.ndarray, intervals: list[Interval]
    ) -> FiringAnalysisResults:
        """Calculate firing rate in the given sequence of intervals."""
        values = []
        events = []
        for interval in intervals:
            interval_spikes = interval.contained(spikes)
            normalized_spikes = interval.normalize_by(interval_spikes) * 360.0
            cut_events = IntervalRelatedEvents(interval, normalized_spikes)
            values.append(self._count_in_interval(cut_events))
            events.append(cut_events.sequence)
        return FiringAnalysisResults(
            self._make_histogram(values=np.stack(values, axis=1)),
            EventSequenceList(events)
        )

    def _collect_cycles(
        self,
        cell_data: RecordingWithSpikes,
        episode_name: str,
        marker_name: str
    ) -> list[Interval]:
        """Collect oscillation cycles based on the episode and marker names."""
        intervals = cell_data.get_episode_intervals(episode_name)
        markers = cell_data.get_markers(marker_name)
        cycles = []
        for interval in intervals:
            interval_markers = interval.contained(markers)
            for start, end in zip(interval_markers[:-1], interval_markers[1:]):
                cycles.append(Interval(start, end))
        return cycles

    def _count_in_interval(
        self, spikes_within: IntervalRelatedEvents
    ) -> np.ndarray:
        """Count the number of spikes within the given interval."""
        counts, _ = self._calculate_histogram(spikes_within.sequence)

        if self._value_unit == 'freq':
            counts = counts / (spikes_within.interval.duration / self._num_bins)
        elif self._value_unit == 'prob':
            total_spikes = len(spikes_within.sequence)
            if total_spikes > 0:
                counts = counts / total_spikes

        counts = counts.astype(float)
        if self._smooth is not None:
            counts = self._smooth(counts)

        return counts

    def _mark_valid_trials(
        self,
        result: SpikeTypeGroupedResults
    ) -> np.ndarray:
        """Mark valid trials based on the trigger condition."""
        n_trials = list(result.items())[0][1].histograms.n_trials
        flag = np.ones(n_trials, dtype=bool)
        for condition in self._condition:
            sequences = result[condition].sequences.sequences
            for i, sequence in enumerate(sequences):
                if len(sequence) == 0:
                    flag[i] = False
        return flag

    def _repeat_circular_events(
        self,
        events: np.ndarray
    ) -> np.ndarray:
        """Repeat the events for circular x axis plotting."""
        repeat_events = []

        if self._xlim[0] < 0:
            before_events = events[events > (360 + self._xlim[0])]
            repeat_events.append(before_events - 360)

        middle_events = events[
            (self._xlim[0] <= events) & (events <= self._xlim[1])
        ]
        repeat_events.append(middle_events)

        if self._xlim[1] > 360:
            after_events = events[events < (self._xlim[1] - 360)]
            repeat_events.append(after_events + 360)

        return np.concatenate(repeat_events, axis=0)

    def _repeat_circular_data(
        self,
        histogram: MultiTrialHistogram,
    ) -> MultiTrialHistogram:
        """Repeat the histogram data for circular x axis plotting."""
        centers = histogram.centers
        values = histogram.values

        repeat_centers = []
        repeat_values = []

        if self._xlim[0] < 0:
            before_flag = (360 + self._xlim[0]) < centers
            repeat_centers.append(centers[before_flag] - 360)
            repeat_values.append(values[before_flag, :])

        middle_flag = (self._xlim[0] <= centers) & (centers <= self._xlim[1])
        repeat_centers.append(centers[middle_flag])
        repeat_values.append(values[middle_flag, :])

        if self._xlim[1] > 360:
            after_flag = centers < (self._xlim[1] - 360)
            repeat_centers.append(centers[after_flag] + 360)
            repeat_values.append(values[after_flag, :])


        return MultiTrialHistogram(
            centers=np.concatenate(repeat_centers, axis=0),
            values=np.concatenate(repeat_values, axis=0),
            data_unit=histogram.data_unit,
            value_unit=histogram.value_unit
        )

    def _plot(self, cell_result: CellWrapper[SpikeTypeGroupedResults]):
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

            n_trials = len(cell_result.data['RA-AP'].sequences.sequences)

            raster_offset = (self._raster_size * n_trials) / 2
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
                    circular_events = self._repeat_circular_events(events)
                    raster_ax.vlines(
                        circular_events,
                        i - raster_offset,
                        i + raster_offset,
                        color=spike_color
                    )
            raster_ax.set_ylim((
                -raster_offset, raster_offset + n_trials
            ))
            raster_ax.set_yticks([])
            raster_ax.set_ylabel('')
        else:
            fig, hist_ax = plt.subplots(1, 1, figsize=(8, 3))

        valid_trial_flag = self._mark_valid_trials(cell_result.data)
        n_all_trials = valid_trial_flag.shape[0]
        n_valid_trials = np.sum(valid_trial_flag)
        for spike_type, result in cell_result.data.items():
            spike_color = spike_colors[spike_type]
            plot_histogram = self._repeat_circular_data(result.histograms)

            hist: MultiTrialHistogram = plot_histogram
            centers = hist.centers
            values = hist.values[:, valid_trial_flag]

            mean_values = np.mean(values, axis=1)
            std_values = np.std(values, axis=1)
            sem_values = std_values / np.sqrt(values.shape[1])


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

        hist_ax.set_ylabel(self._y_label)
        hist_ax.set_ylim(self._ylim)

        hist_ax.set_xlabel('Cycle phase')
        hist_ax.set_xlim(self._xlim)

        hist_ax.legend()

        hist_ax.set_title(
            f'n={n_valid_trials}/{n_all_trials}'
        )
        plt.suptitle(
            f'{cell_result.info.name} Spike-{self._episode_name} cycle coupling'
        )
        if self._paths:
            fig_path = (
                self._paths['figures'] /
                f'{cell_result.info.name}_spike_{self._episode_name}_cycle_coupling'
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


class SpikePhaseCouplingGroupAggregatorWorkflow:
    """Class to aggregate peri-trigger time histogram results across multiple cells."""

    def __init__(self, params: SpikePhaseCouplingWorkflowParams):
        self._palette = params.palette
        self._spike_types = params.analysis.spike_types
        self._interval_condition = params.analysis.interval_condition
        self._xlim = params.analysis.xlim
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
