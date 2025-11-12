"""This module contains classes and function to run spike autocorrelation analysis workflows."""

from dataclasses import dataclass
import functools
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

from analyses.model import RecordingWithSpikes
from common.data_utils import CellWrapper
from param_classes.workflows.spike_ac_params import SpikeACAnalysisParamsWorkflowParams
from triggered_firing.calculators import (
    Histogram,
    MultiTrialHistogram,
    select_spikes
)

from workflows.normalized_inter_raap import Interval
from workflows.run_util.io import ResultPaths
from workflows.result_classes import EventSequenceList, FiringAnalysisResults

@dataclass
class CategorizedResult:
    """Dataclass to hold categorized analysis results."""
    histogram: Histogram
    spike_type: str = ''
    episode_name: str = ''

def run_analysis(
    params: SpikeACAnalysisParamsWorkflowParams,
    input_files: list[CellWrapper[ResultPaths]]
) -> list[CellWrapper[DataFrame]]:
    """Run the spike autocorrelation analysis workflow on the given input files."""
    analysis = SpikeACAnalysis(params)
    return [analysis(input_file) for input_file in input_files]


class SpikeACAnalysis:
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
        data = cell.data.load()
        results = self._calculate(data)
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate(
        self, cell_data: RecordingWithSpikes
    ) -> dict[str, dict[str, FiringAnalysisResults]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        results = {}
        for episode_name in self._episode_names:
            try:
                results[episode_name] = self._calculate_episode_ac(
                    cell_data.spikes,
                    cell_data.get_episode_intervals(episode_name)
                )
            except KeyError:
                continue

        return results

    def _calculate_episode_ac(
        self, spikes: DataFrame, intervals: list[Interval]
    ) -> dict[str, FiringAnalysisResults]:
        """Calculate the spike autocorrelation histograms during the given episode."""
        return {
            spike_type: self._calculate_ac(
                select_during_episode(
                    select_spikes(spikes, spike_type), intervals
                )
            ) for spike_type in self._spike_types
        }

    def _calculate_ac(self, spikes: np.ndarray) -> FiringAnalysisResults:
        """Calculate the autocorrelation histogram for the given spikes."""

        if len(spikes) == 0:
            return FiringAnalysisResults(
                self._make_histogram(values=np.zeros_like(self._centers)),
                sequences=EventSequenceList([])
            )

        hist_counts, sequences = [], []
        for i, trigger in enumerate(spikes):
            other_spikes = (np.delete(spikes, i) - trigger) * 1000
            counts, _  = self._calculate_histogram(other_spikes)
            if self._value_unit == 'freq':
                counts = counts / self._bin_size * 1000
            elif self._value_unit == 'prob':
                if sum(counts) > 0:
                    counts = counts / sum(counts)

            counts = counts.astype(float)
            if self._smooth is not None:
                counts = self._smooth(counts)

            hist_counts.append(counts)
            sequences.append(other_spikes)
        values = np.stack(hist_counts, axis=1)
        return FiringAnalysisResults(
            self._make_histogram(values=values),
            sequences=EventSequenceList(sequences)
        )

    def _plot(
        self,
        cell_result: CellWrapper[dict[str, dict[str, FiringAnalysisResults]]]
    ):
        """Plot the results."""
        cell_name = cell_result.info.name
        spike_colors = self._palette['spike_type'] | {
            'AIS-AP': self._palette['cell_type'][cell_result.info.cell_type]
        }

        for episode_name, episode_results in cell_result.data.items():
            for spike_type, firing_result in episode_results.items():
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
                        marker_size=self._raster_size,
                        xlim=self._xlim
                    )
                    axes[0].set_xticks([])
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
                    f'{cell_result.info.name} - {episode_name} {spike_type} AC'
                )

                if self._paths:
                    save_figure(
                        self._paths['figures'],
                        fig,
                        f'{cell_name}_{episode_name}_{spike_type}_ac'
                    )
                else:
                    plt.show()

    def _save_result(
        self,
        cell_result: CellWrapper[dict[str, dict[str, FiringAnalysisResults]]]
    ):
        """Save the results to file."""
        if not self._paths:
            return

        cell_name = cell_result.info.name
        for episode_name, episode_results in cell_result.data.items():
            for spike_type, firing_result in episode_results.items():
                save_name = f'{cell_name}_{episode_name}_{spike_type}_ac'
                histograms = firing_result.histograms
                col_names = [
                    f'trigger_{i + 1}' for i in range(histograms.n_trials)
                ]
                df = DataFrame(histograms.values, columns=col_names)
                df.insert(0, 'centers', histograms.centers)
                df.to_excel(
                    f'{self._paths["results"]}/{save_name}.xlsx', index=False
                )

def make_raster_plot(
    sequences: list[np.ndarray],
    ax: Axes,
    color: str,
    marker_size: float,
    xlim: tuple[float, float]
):
    """Create a raster plot figure."""
    raster_offset = marker_size / 2
    for i, events in enumerate(sequences):
        ax.vlines(
            events,
            i - raster_offset,
            i + raster_offset,
            color=color
        )
    ax.set_xlim(xlim)
    ax.set_ylim((-raster_offset, raster_offset + len(sequences)))
    ax.set_yticks([])
    ax.set_ylabel('')

def make_boundedline_histogram(
    histogram: MultiTrialHistogram,
    ax: Axes,
    color: str
):
    """Draw a multi trial histogram with a bounded line."""
    centers = histogram.centers
    values = histogram.values
    mean_values = np.mean(values, axis=1)
    sd_values = np.std(values, axis=1)
    sem_values = sd_values / np.sqrt(values.shape[1])
    ax.fill_between(
        centers,
        mean_values - sem_values,
        mean_values + sem_values,
        color=color,
        alpha=0.3
    )
    ax.plot(
        centers,
        mean_values,
        '-',
        color=color
    )

def select_during_episode(
    spikes: np.ndarray, intervals: list[Interval]
) -> np.ndarray:
    """Select spikes that occur during any interval of the given episode."""
    subsequences = [interval.contained(spikes) for interval in intervals]
    return np.concatenate(subsequences, axis=0)

def save_figure(path: Path, fig: Figure, name: str):
    """Save the given figure to the specified path in PNG and SVG formats."""
    fig.savefig(f'{path}/{name}.png', format='png', dpi=300, bbox_inches='tight')
    # with plt.rc_context({'svg.fonttype': 'none'}):
    #     fig.savefig(f'{path}/{name}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
