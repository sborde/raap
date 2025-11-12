"""This module contains classes and function to run spike autocorrelation analysis workflows."""

import functools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

from analyses.model import (
    RecordingWithSpikes,
    Interval
)
from common.data_utils import CellWrapper
from param_classes.workflows.spike_ac_params import SpikeACAnalysisParamsWorkflowParams
from triggered_firing.calculators import (
    MultiTrialHistogram
)

from workflows.normalized_interval import select_spikes
from workflows.result_classes import EventSequenceList, FiringAnalysisResults
from workflows.run_util.io import ResultPaths
from workflows.spike_ac_analysis import (
    make_boundedline_histogram,
    make_raster_plot,
    save_figure,
    select_during_episode
)


def run_analysis(
    params: SpikeACAnalysisParamsWorkflowParams,
    input_files: list[CellWrapper[ResultPaths]]
) -> list[CellWrapper[DataFrame]]:
    """Run the spike autocorrelation analysis workflow on the given input files."""
    analysis = SpikeCorrelationAnalysis(params)
    return [analysis(input_file) for input_file in input_files]


class SpikeCorrelationAnalysis:
    """Class to perform spike autocorrelation analysis for individual spike types."""
    def __init__(self, config: SpikeACAnalysisParamsWorkflowParams):
        params = config.analysis
        self._episode_names = params.episodes
        self._spike_types = params.spike_types

        self._smooth = None
        if params.smooth_size > 0.0:
            self._smooth = functools.partial(
                gaussian_filter1d,
                sigma=params.smooth_size,
                axis=0
            )

        self._bin_size = params.bin_size
        self._bins = np.arange(
            params.bin_range[0],
            params.bin_range[1] + params.bin_size,
            params.bin_size
        )
        self._centers = np.mean(
            [self._bins[:-1], self._bins[1:]], axis=0
        )
        self._value_unit = params.values
        self._calculate_histogram = functools.partial(
            np.histogram,
            bins=self._bins
        )
        self._make_histogram = functools.partial(
            MultiTrialHistogram,
            centers=self._centers,
            data_unit='ms',
            value_unit=self._value_unit
        )

        self._to_svg = params.to_svg
        self._palette = config.palette
        self._show_raster = params.show_raster
        self._raster_size = params.raster_size
        self._ylim = params.ylim
        self._xlim = params.bin_range

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

    @property
    def _y_label(self) -> str:
        """Get the y label for the plots based on the value unit."""
        if self._value_unit == 'count':
            return 'Avg spike count'
        if self._value_unit == 'freq':
            return 'Firing (Hz)'
        if self._value_unit == 'prob':
            return 'Spike distribution'
        return ''

    def __call__(self, cell: CellWrapper[ResultPaths]) -> CellWrapper:
        """Process a single input file and return the results."""
        cell_data = cell.data.load()
        results = self._calculate(cell_data)
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate(
        self, cell_data: RecordingWithSpikes
    ) -> dict[str, dict[str, dict[str, FiringAnalysisResults]]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        results = {}
        for episode_name in self._episode_names:
            try:
                results[episode_name] = self._calculate_episode_correlations(
                    cell_data.spikes,
                    cell_data.get_episode_intervals(episode_name)
                )
            except KeyError:
                continue

        return results

    def _calculate_episode_correlations(
        self, spikes: DataFrame, intervals: list[Interval]
    ) -> dict[str, dict[str, FiringAnalysisResults]]:
        """Calculate the spike cross-correlation histograms during the given episode."""
        result = {}
        for trigger_type in self._spike_types:
            result[trigger_type] = {}
            triggers = select_during_episode(
                select_spikes(spikes, trigger_type), intervals
            )
            for event_type in self._spike_types:
                if trigger_type == event_type:
                    continue
                events = select_during_episode(
                    select_spikes(spikes, event_type), intervals
                )
                result[trigger_type][event_type] = self._calculate_cc(
                    triggers, events
                )
        return result

    def _calculate_cc(
        self,
        triggers: np.ndarray,
        events: np.ndarray
    ) -> FiringAnalysisResults:
        """Calculate the event cross-correlation histogram for the given triggers and events."""
        if len(triggers) == 0 or len(events) == 0:
            return FiringAnalysisResults(
                self._make_histogram(values=np.zeros_like(self._centers)),
                sequences=EventSequenceList([])
            )

        hist_counts, sequences = [], []
        for trigger in triggers:
            subsequence = (events - trigger) * 1000
            counts, _  = self._calculate_histogram(subsequence)
            if self._value_unit == 'freq':
                counts = counts / self._bin_size * 1000
            elif self._value_unit == 'prob':
                if sum(counts) > 0:
                    counts = counts / sum(counts)

            counts = counts.astype(float)
            if self._smooth is not None:
                counts = self._smooth(counts)

            hist_counts.append(counts)
            sequences.append(subsequence)

        values = np.stack(hist_counts, axis=1)

        return FiringAnalysisResults(
            self._make_histogram(values=values),
            sequences=EventSequenceList(sequences)
        )

    def _plot(
        self,
        cell_result: CellWrapper[dict[str, dict[str, dict[str, FiringAnalysisResults]]]]
    ):
        """Plot the results."""
        cell_name = cell_result.info.name
        spike_colors = self._palette['spike_type'] | {
            'AIS-AP': self._palette['cell_type'][cell_result.info.cell_type]
        }

        for episode_name, episode_results in cell_result.data.items():
            for trigger_name, trigger_results in episode_results.items():
                for spike_type, firing_result in trigger_results.items():
                    if self._show_raster:
                        fig, axes  = plt.subplots(
                            2, 1, figsize=(8, 6),
                            constrained_layout=True,
                            sharex=True
                        )
                        hist_ax = axes[1]
                        make_raster_plot(
                            sequences=firing_result.sequences.sequences,
                            ax=axes[0],
                            color=spike_colors[spike_type],
                            marker_ratio=self._raster_size,
                            xlim=self._xlim
                        )
                        axes[0].set_xlabel('')
                    else:
                        fig, hist_ax  = plt.subplots(1, 1, figsize=(8, 3))

                    histogram = firing_result.histograms
                    make_boundedline_histogram(
                        histogram,
                        hist_ax,
                        color=spike_colors[spike_type]
                    )
                    hist_ax.set_xlim(self._xlim)
                    hist_ax.set_ylim(self._ylim)

                    hist_ax.set_title(f'n={histogram.n_trials}')
                    hist_ax.set_ylabel(self._y_label)
                    hist_ax.set_xlabel('Time (ms)')
                    plt.suptitle(
                        f'{cell_result.info.name} - {episode_name} ' +
                        f'{trigger_name}-{spike_type} CrossCorr'
                    )

                    if self._paths:
                        save_figure(
                            self._paths['figures'],
                            fig,
                            (
                                f'{cell_name}_{episode_name}_' +
                                f'{trigger_name}_{spike_type}_cc'
                            ),
                            self._to_svg
                        )
                    else:
                        plt.show()

    def _save_result(
        self,
        cell_result: CellWrapper[dict[str, dict[str, dict[str, FiringAnalysisResults]]]]
    ):
        """Save the results to file."""
        if not self._paths:
            return

        cell_name = cell_result.info.name
        for episode_name in self._episode_names:
            if episode_name not in cell_result.data:
                continue
            episode_result = cell_result.data[episode_name]
            for trigger_name in self._spike_types:
                if trigger_name not in episode_result:
                    continue
                trigger_result = episode_result[trigger_name]
                for spike_type in self._spike_types:
                    if spike_type not in trigger_result:
                        continue
                    save_name = (
                        f'{cell_name}_{episode_name}_' +
                        f'{trigger_name}_{spike_type}_cc'
                    )
                    histograms = trigger_result[spike_type].histograms
                    col_names = [
                        f'trigger_{i + 1}' for i in range(histograms.n_trials)
                    ]
                    df = DataFrame(
                        histograms.values,
                        columns=col_names
                    )
                    df.insert(0, 'centers', histograms.centers)

                    df.to_excel(
                        f'{self._paths["results"]}/{save_name}.xlsx',
                        index=False
                    )
