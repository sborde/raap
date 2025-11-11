

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from matplotlib import pyplot as plt
import numpy as np

from pandas import DataFrame

from analyses.model import RecordingWithSpikes, Interval
from common.data_utils import CellWrapper
from param_classes.workflows.normalized_inter_raap import (
    InterRaapSpikeCountFiringWorkflowParams,
    NormalizedInterRaapFiringWorkflowParams
)
from triggered_firing.calculators import Histogram, MultiTrialHistogram
from workflows.run_util.io import ResultPaths

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: NormalizedInterRaapFiringWorkflowParams
):
    """Run the normalized inter RA-AP firing workflow on the given input files."""
    analysis = NormalizedInterRaapFiringWorkflow(params)
    for cell in input_files:
        analysis(cell)
    # [analysis(input_file) for input_file in input_files]

def run_inter_raap_spike_count_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: InterRaapSpikeCountFiringWorkflowParams
):
    """Run the inter RA-AP spike count firing workflow on the given input files."""
    analysis = InterRaapSpikeCountFiringWorkflow(params)
    for cell in input_files:
        analysis(cell)


class IntervalLike(Protocol):
    """Protocol for interval-like objects."""
    @property
    @abc.abstractmethod
    def start(self) -> float:
        """Get the start of the interval."""

    @property
    @abc.abstractmethod
    def end(self) -> float:
        """Get the end of the interval."""

@dataclass
class IntervalRelatedEvents:
    """Class representing events related to an interval."""
    interval: Interval
    events: np.ndarray


class NormalizedInterRaapFiringWorkflow:
    """
    Workflow for analyzing normalized inter RA-AP interval firing of AIS-AP.
    """
    def __init__(self, params: NormalizedInterRaapFiringWorkflowParams):
        self._palette = params.palette
        self._histogram_params = params.analysis.histogram
        self._smooth_size = params.analysis.smooth_size
        self._cut_window = Interval(
            params.analysis.histogram.bin_range[0],
            params.analysis.histogram.bin_range[1]
        )
        self._bins = np.arange(
            self._histogram_params.bin_range[0],
            self._histogram_params.bin_range[1] + self._histogram_params.bin_size,
            self._histogram_params.bin_size
        )
        self._centers = np.mean(
            [self._bins[:-1], self._bins[1:]], axis=0
        )
        self._bin_size = self._histogram_params.bin_size
        self._results_dir = Path(params.output.path) / '01_results'
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = Path(params.output.path) / '02_figures'
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, cell: CellWrapper[ResultPaths]):
        """Process a single input file and return the results."""
        cell_data = cell.data.load()
        results = self._calculate(cell_data.spikes)
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        # self._save_result(cell_result)

    def _plot(self, cell_result: CellWrapper[tuple[MultiTrialHistogram, list[np.ndarray]]]) -> None:
        """Plot the results."""
        cell_color = self._palette['cell_type'][cell_result.info.cell_type]
        fig, axes  = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        hist, norm_events = cell_result.data
        for i, events in enumerate(norm_events):
            axes[0].vlines(
                events,
                i-0.25,
                i+0.25,
                color=cell_color
            )
        axes[0].set_title(cell_result.info.name)

        mean_hist = np.mean(hist.values, axis=1)
        sem_hist = np.std(hist.values, axis=1) / np.sqrt(hist.values.shape[1])
        axes[1].fill_between(
            hist.centers,
            mean_hist - sem_hist,
            mean_hist + sem_hist,
            color=cell_color,
            alpha=0.3
        )
        axes[1].plot(hist.centers, mean_hist, '-', color=cell_color)
        plt.show()

    def _calculate(
        self, spike_data: DataFrame
    ) -> tuple[MultiTrialHistogram, list[np.ndarray]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        type_vector = spike_data['spike_type'].to_numpy()
        timestamps = spike_data['timestamp'].to_numpy()
        interval_events = self._collect(
            timestamps[type_vector == 'RA-AP'],
            timestamps[type_vector == 'AIS-AP']
        )

        hist_values, normalized_events = [], []
        for interval, events in interval_events:
            norm_events = (events - interval.start) / interval.duration
            values, _ = np.histogram(norm_events, bins=self._bins)

            value_unit = 'count'
            if self._histogram_params.value_unit == 'prob':
                if np.sum(values) > 0:
                    values = values / np.sum(values)
                value_unit = 'prob'
            if self._histogram_params.value_unit == 'freq':
                values = values / (interval.duration * self._bin_size)
                value_unit = 'Hz'

            hist_values.append(values)
            normalized_events.append(norm_events)

        histograms = MultiTrialHistogram(
            centers=self._centers,
            values=np.stack(hist_values, axis=1),
            data_unit='',
            value_unit=value_unit
        )

        return histograms, normalized_events

    def _collect(
        self, markers: np.ndarray, spikes: np.ndarray
    ) -> list[tuple[Interval, np.ndarray]]:
        """Collect spikes within the given intervals."""
        intervals = convert_markers_to_intervals(markers)
        cut_intervals = [
            self._get_cut_interval(interval) for interval in intervals
        ]
        return [
            (interval, cut_intervals.contained(spikes))
            for (interval, cut_intervals) in zip(intervals, cut_intervals)
        ]

    def _get_cut_interval(
        self, interval: Interval
    ) -> Interval:
        """Get the cut interval based on histogram parameters."""
        cut_start = interval.start + self._cut_window.start * interval.duration
        cut_end = interval.start + self._cut_window.end * interval.duration
        return Interval(cut_start, cut_end)


class InterRaapSpikeCountFiringWorkflow:
    """
    Workflow for counting the AIS-AP spikes in inter-RA-AP intervals.
    """
    def __init__(self, params: InterRaapSpikeCountFiringWorkflowParams):
        self._palette = params.palette
        self._episode_names = params.analysis.episodes

        self._results_dir = Path(params.output.path) / '01_results'
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = Path(params.output.path) / '02_figures'
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, cell: CellWrapper[ResultPaths]):
        """Process a single input file and return the results."""
        data = cell.data.load()
        episode_histograms = self._calculate(data)
        cell_result = CellWrapper(cell.info, episode_histograms)
        self._plot(cell_result)
        # self._save_result(cell_result)

    def _calculate(
        self, cell_data: RecordingWithSpikes
    ) -> list[tuple[str, Histogram]]:
        histograms = []
        for episode_name in self._episode_names:
            try:
                episodes = list(cell_data.recording.get_episode(episode_name))
                histogram = self._calculate_for_episode(
                    cell_data.spikes, episodes
                )
                histograms.append((episode_name, histogram))
            except KeyError as e:
                print(f"Episode '{episode_name}' is missing, skip. {e}")
        return histograms

    def _calculate_for_episode(
        self, spike_data: DataFrame, episodes: Sequence[IntervalLike]
    ) -> Histogram:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        results = []
        for episode in episodes:
            episode_spikes = get_spikes_in_interval(episode, spike_data)
            type_vector = episode_spikes['spike_type'].to_numpy()
            timestamps = episode_spikes['timestamp'].to_numpy()
            interspike_events = self._collect(
                timestamps[type_vector == 'RA-AP'],
                timestamps[type_vector == 'AIS-AP']
            )
            results += interspike_events

        results = sorted(results, key=lambda x: x.interval.duration)

        histograms = Histogram(
            centers=np.array([r.interval.duration for r in results]),
            values=np.array([[len(r.events)] for r in results]),
            data_unit='s',
            value_unit='count'
        )

        return histograms

    def _collect(
        self, markers: np.ndarray, events: np.ndarray
    ) -> list[IntervalRelatedEvents]:
        """Collect spikes within the given intervals."""
        intervals = convert_markers_to_intervals(markers)
        return collect_event_subsequences(intervals, events)

    def _plot(self, cell_result: CellWrapper[list[tuple[str, Histogram]]]) -> None:
        """Plot the results."""
        fig, ax  = plt.subplots(len(cell_result.data), 1, figsize=(8, 3))
        histograms = cell_result.data
        for i, (episode_name, histogram) in enumerate(histograms):
            color = self._palette['episode'][episode_name]
            ax[i].plot(
                histogram.centers,
                histogram.values,
                '-',
                color=color,
                label=episode_name
            )
            ax[i].legend([], [], frameon=False)
            ax[i].set_title(episode_name)
        plt.suptitle(cell_result.info.name)
        plt.show()

def collect_spikes_in_episode(
    intervals: Sequence[IntervalLike],
    spike_data: DataFrame
) -> DataFrame:
    """Collect spikes within the given episodes."""
    flag = np.zeros(len(spike_data), dtype=bool)
    for interval in intervals:
        flag |= (
            (spike_data['timestamp'] >= interval.start) &
            (spike_data['timestamp'] <= interval.end)
        )
    return spike_data[flag]

def get_spikes_in_interval(
    interval: IntervalLike, spike_data: DataFrame
) -> DataFrame:
    """Get spikes within the given interval."""
    return spike_data[
        (spike_data['timestamp'] >= interval.start) &
        (spike_data['timestamp'] <= interval.end)
    ]

def collect_event_subsequences(
    intervals: list[Interval],
    events: np.ndarray
) -> list[IntervalRelatedEvents]:
    """Collect event subsequences within the given intervals."""
    return [
        IntervalRelatedEvents(interval, interval.contained(events))
        for interval in intervals
    ]

def convert_segmented_markers_to_intervals(
    segmented_markers: list[np.ndarray]
) -> list[Interval]:
    """Convert segmented spike times to inter-spike intervals."""
    intervals = []
    for markers in segmented_markers:
        intervals.extend(convert_markers_to_intervals(markers))
    return intervals

def convert_markers_to_intervals(markers: np.ndarray) -> list[Interval]:
    """Convert spike times to inter-spike intervals."""
    return [
        Interval(markers[i], markers[i + 1])
        for i in range(len(markers) - 1)
    ]
