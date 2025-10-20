

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from pandas import DataFrame

from common.data_utils import CellWrapper
from param_classes.workflows.normalized_inter_raap import (
    NormalizedInterRaapFiringWorkflowParams
)
from triggered_firing.calculators import Histogram
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


class NormalizedInterRaapFiringWorkflow:
    """
    Workflow for analyzing normalized inter RA-AP interval firing of AIS-AP.
    """
    def __init__(self, params: NormalizedInterRaapFiringWorkflowParams):
        self._palette = params.palette
        self._histogram_params = params.analysis.histogram
        self._smooth_size = params.analysis.smooth_size
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

    def _plot(self, cell_result: CellWrapper[tuple[Histogram, list[np.ndarray]]]) -> None:
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
    ) -> tuple[Histogram, list[np.ndarray]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        type_vector = spike_data['spike_type'].to_numpy()
        timestamps = spike_data['timestamp'].to_numpy()
        interval_events = self._collect(
            timestamps[type_vector == 'RA-AP'],
            timestamps[type_vector == 'AIS-AP']
        )

        hist_values, normalized_events = [], []
        for interval, events in interval_events:
            duration = interval[1] - interval[0]
            norm_events = (events - interval[0]) / duration
            values, _ = np.histogram(norm_events, bins=self._bins)

            value_unit = 'count'
            if self._histogram_params.value_unit == 'prob':
                if np.sum(values) > 0:
                    values = values / np.sum(values)
                value_unit = 'prob'
            if self._histogram_params.value_unit == 'freq':
                values = values / (duration * self._bin_size)
                value_unit = 'Hz'

            hist_values.append(values)
            normalized_events.append(norm_events)

        histograms = Histogram(
            centers=self._centers,
            values=np.stack(hist_values, axis=1),
            data_unit='',
            value_unit=value_unit
        )

        return histograms, normalized_events

    def _collect(
        self, markers: np.ndarray, spikes: np.ndarray
    ) -> list[tuple[tuple[float, float], np.ndarray]]:
        """Collect spikes within the given intervals."""
        intervals = self._convert_to_intervals(markers)
        cut_intervals = [
            self._get_cut_interval(interval) for interval in intervals
        ]
        return [
            (interval, spikes[(spikes >= start) & (spikes <= end)])
            for (interval, (start, end)) in zip(intervals, cut_intervals)
        ]

    def _convert_to_intervals(self, spike_times: np.ndarray) -> np.ndarray:
        """Convert spike times to inter-spike intervals."""
        return np.stack([spike_times[:-1], spike_times[1:]], axis=1)

    def _get_cut_interval(
        self, interval: tuple[float, float]
    ) -> tuple[float, float]:
        """Get the cut interval based on histogram parameters."""
        start, end = interval
        duration = end - start
        cut_start = start + self._histogram_params.bin_range[0] * duration
        cut_end = start + self._histogram_params.bin_range[1] * duration
        return cut_start, cut_end

