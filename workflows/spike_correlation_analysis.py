"""This module contains classes and function to run spike autocorrelation analysis workflows."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

from analyses.model import RecordingWithSpikes
from common.data_utils import CellWrapper
from param_classes.workflows.spike_ac_params import SpikeACAnalysisParamsWorkflowParams
from triggered_firing.calculators import (
    Histogram
)

from workflows.normalized_inter_raap import Interval
from workflows.normalized_interval import select_spikes
from workflows.run_util.io import ResultPaths
from workflows.spike_ac_analysis import save_figure, select_during_episode


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
        self._episodes = params.episodes
        self._spike_types = params.spike_types
        self._smooth_size = params.smooth_size
        self._cut_window = Interval(
            params.bin_range[0], params.bin_range[1]
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
        self._palette = config.palette

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

    def __call__(self, cell: CellWrapper[ResultPaths]) -> CellWrapper:
        """Process a single input file and return the results."""
        cell_data = cell.data.load()
        results = self._calculate(cell_data)
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    @property
    def _y_label(self) -> str:
        """Get the y label for the plots based on the value unit."""
        if self._value_unit == 'count':
            return 'Avg spike count'
        elif self._value_unit == 'freq':
            return 'Firing (Hz)'
        elif self._value_unit == 'prob':
            return 'Spike distribution'
        return ''

    def _calculate(
        self, cell_data: RecordingWithSpikes
    ) -> dict[str, dict[str, dict[str, Histogram]]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        results = {}
        for episode_name in self._episodes:
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
    ) -> dict[str, dict[str, Histogram]]:
        """Calculate the spike cross-correlation histograms during the given episode."""
        result = {}
        for trigger_type in self._spike_types:
            result[trigger_type] = {}
            triggers = select_during_episode(
                select_spikes(spikes, trigger_type), intervals
            )
            for event_type in self._spike_types:
                if trigger_type == event_type:
                    result[trigger_type][event_type] = self._calculate_ac(
                        triggers
                    )
                else:
                    result[trigger_type][event_type] = self._calculate_cc(
                        triggers,
                        select_during_episode(
                            select_spikes(spikes, event_type), intervals
                        )
                    )
        return result

    def _calculate_cc(
        self,
        triggers: np.ndarray,
        events: np.ndarray
    ) -> Histogram:
        """Calculate the event cross-correlation histogram for the given triggers and events."""
        if len(triggers) == 0 or len(events) == 0:
            return Histogram(
                self._centers,
                np.zeros_like(self._centers),
                'ms',
                self._value_unit
            )

        sweeps = []
        for i, trigger in enumerate(triggers):
            other_spikes = (events - trigger) * 1000
            sweeps.append(np.histogram(other_spikes, bins=self._bins)[0])
        values = np.mean(np.stack(sweeps, axis=1), axis=1)

        if self._value_unit == 'freq':
            values = values / self._bin_size * 1000
        elif self._value_unit == 'prob':
            if sum(values) > 0:
                values = values / sum(values)

        if self._smooth_size > 0.0:
            values = gaussian_filter1d(values, self._smooth_size, axis=0)

        return Histogram(
            self._centers,
            values,
            'ms',
            self._value_unit
        )

    def _calculate_ac(self, spikes: np.ndarray) -> Histogram:
        """Calculate the autocorrelation histogram for the given spikes."""

        if len(spikes) == 0:
            return Histogram(
                self._centers,
                np.zeros_like(self._centers),
                'ms',
                self._value_unit
            )

        sweeps = []
        for i, trigger in enumerate(spikes):
            other_spikes = (np.delete(spikes, i) - trigger) * 1000
            sweeps.append(np.histogram(other_spikes, bins=self._bins)[0])
        values = np.mean(np.stack(sweeps, axis=1), axis=1)

        if self._value_unit == 'freq':
            values = values / self._bin_size * 1000
        elif self._value_unit == 'prob':
            if sum(values) > 0:
                values = values / sum(values)

        if self._smooth_size > 0.0:
            values = gaussian_filter1d(values, self._smooth_size, axis=0)

        return Histogram(
            self._centers,
            values,
            'ms',
            self._value_unit
        )

    def _plot(
        self,
        cell_result: CellWrapper[dict[str, dict[str, dict[str, Histogram]]]]
    ):
        """Plot the results."""
        cell_name = cell_result.info.name
        cell_color = self._palette['cell_type'][cell_result.info.cell_type]

        for episode_name, episode_results in cell_result.data.items():
            for trigger_name, trigger_results in episode_results.items():
                fig, hist_ax  = plt.subplots(1, 1, figsize=(8, 3))
                for spike_type, hist in trigger_results.items():
                    if spike_type == 'RA-AP':
                        color = self._palette['spike_type'][spike_type]
                    else:
                        color = cell_color

                    hist_ax.plot(
                        hist.centers,
                        hist.values,
                        '-',
                        color=color,
                        label=spike_type
                    )

                hist_ax.set_ylabel(self._y_label)
                hist_ax.set_xlabel('Time (ms)')
                hist_ax.set_title(
                    f'{cell_result.info.name} - ' +
                    f'{episode_name} correlation\n' +
                    f'trigger: {trigger_name}'
                )
                hist_ax.legend()
                plt.tight_layout(pad=1.0)

                if self._paths:
                    save_figure(
                        self._paths['figures'],
                        fig,
                        (
                            f'{cell_name}_{episode_name}_' +
                            f'{trigger_name}_spike_correlation'
                        )
                    )
                else:
                    plt.show()

    def _save_result(
        self,
        cell_result: CellWrapper[dict[str, dict[str, dict[str, Histogram]]]]
    ):
        """Save the results to file."""
        if not self._paths:
            return

        cell_name = cell_result.info.name
        save_name = f'{cell_name}_spike_correlation_results'
        frames = []
        for trigger_name in self._spike_types:
            for spike_type in self._spike_types:
                df = DataFrame({
                    'trigger_name': trigger_name,
                    'spike_type': spike_type,
                    'centers': self._centers
                })
                for episode_name, episode_results in cell_result.data.items():
                    df[episode_name] = episode_results[trigger_name][spike_type].values
                frames.append(df)
        pool_frame = pd.concat(frames, ignore_index=True)
        pool_frame.to_excel(
            f'{self._paths["results"]}/{save_name}.xlsx', index=False
        )
