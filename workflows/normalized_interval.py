

import functools
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from pandas import DataFrame
from matplotlib.figure import Figure

from analyses.model import RecordingWithSpikes
from common.data_utils import CellWrapper
from param_classes.workflows.normalized_interval_firing import (
    NormalizedIntervalFiringWorkflowParams
)
from triggered_firing.calculators import MultiTrialHistogram, select_spikes
from workflows.normalized_inter_raap import Interval
from workflows.run_util.io import ResultPaths
from workflows.result_classes import EventSequenceList, IntervalRelatedEvents, FiringAnalysisResults

SpikeTypeGroupedResults = dict[str, FiringAnalysisResults]
EpisodeRelatedResults = dict[str, SpikeTypeGroupedResults]

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: NormalizedIntervalFiringWorkflowParams
) -> list[CellWrapper[EpisodeRelatedResults]]:
    """Run the normalized inter RA-AP firing workflow on the given input files."""
    analysis = NormalizedIntervalFiringWorkflow(params)
    return [analysis(input_file) for input_file in input_files]

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

        self._make_histogram = functools.partial(
            MultiTrialHistogram,
            centers=self._centers,
            data_unit='',
            value_unit=self._value_unit
        )
        self._extend_interval = functools.partial(
            extend_interval,
            normalized=self._cut_window
        )

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
    ) -> CellWrapper[EpisodeRelatedResults]:
        """Process a single input file and return the results."""
        results = {
            self._episode_name: self._calculate_in_episode(
                cell.data.load(), self._episode_name
            )
        }
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate_in_episode(
        self, cell_data: RecordingWithSpikes, episode_name: str
    ) -> SpikeTypeGroupedResults:
        """Calculate normalized inter RA-AP interval firing for a single episode."""
        starts, ends = cell_data.get_episode_ends(episode_name)
        intervals = [Interval(start, end) for start, end in zip(starts, ends)]
        results = {}
        for spike_type in self._spike_types:
            results[spike_type] = self._calculate_on_subset(
                select_spikes(cell_data.spikes, spike_type), intervals
            )
        return results

    def _calculate_on_subset(
        self, spikes: np.ndarray, intervals: list[Interval]
    ) -> FiringAnalysisResults:
        """Calculate normalized inter RA-AP interval firing on a subset of spikes."""
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
        spikes_within = cutting_interval.contained(spikes)
        normalized_spikes = normalize_to_interval(spikes_within, interval)
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
        self, cell_result: CellWrapper[EpisodeRelatedResults]
    ):
        """Plot the results."""
        cell_name = cell_result.info.name
        cell_color = self._palette['cell_type'][cell_result.info.cell_type]

        for episode_name, episode_results in cell_result.data.items():
            fig, axes  = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            raster_ax = axes[0]
            hist_ax = axes[1]
            for spike_type, result in episode_results.items():
                if spike_type == 'RA-AP':
                    color = self._palette['spike_type'][spike_type]
                else:
                    color = cell_color

                for i, events in enumerate(result.sequences.sequences):
                    raster_ax.vlines(
                        events,
                        i-0.25,
                        i+0.25,
                        color=color
                    )

                hist: MultiTrialHistogram = result.histograms
                centers = hist.centers
                mean_values = np.mean(hist.values, axis=1)
                sd_values = np.std(hist.values, axis=1)
                sem_values = sd_values / np.sqrt(hist.values.shape[1])
                hist_ax.fill_between(
                    centers,
                    mean_values - sem_values,
                    mean_values + sem_values,
                    color=color,
                    alpha=0.3
                )
                hist_ax.plot(
                    centers,
                    mean_values,
                    '-',
                    color=color,
                    label=spike_type
                )

            raster_ax.set_ylabel(f"#{episode_name.capitalize()}")
            hist_ax.set_ylim(self._ylim)
            hist_ax.set_xlabel('Normalized time')

            if self._value_unit == 'count':
                hist_ax.set_ylabel('Avg spike count')
            elif self._value_unit == 'freq':
                hist_ax.set_ylabel('Firing (Hz)')
            elif self._value_unit == 'prob':
                hist_ax.set_ylabel(f'Spike distribution')

            plt.suptitle(
                f'{cell_result.info.name} Normalized {episode_name} firing'
            )
            if self._paths:
                fig_path = (
                    self._paths['figures'] /
                    f'{cell_name}_normalized_{episode_name}_firing'
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

    def _save_result(self, cell_result: CellWrapper[EpisodeRelatedResults]):
        """Save the results to file."""
        if not self._paths:
            return

        cell_name = cell_result.info.name
        for episode_name, episode_result in cell_result.data.items():
            result_path = (
                self._paths['results'] /
                f'{cell_name}_normalized_{episode_name}_firing.xlsx'
            )
            pool_frame = convert_episode_result_to_frame(
                episode_result, episode_name
            )
            pool_frame.to_excel(result_path, index=False)

def convert_episode_result_to_frame(
    results: SpikeTypeGroupedResults,
    episode_name: str
) -> pd.DataFrame:
    frames = []
    for spike_type, result in results.items():
        hist: MultiTrialHistogram = result.histograms
        df = DataFrame({'centers': hist.centers})
        for i in range(hist.values.shape[1]):
            df[f'{episode_name}_{i+1}'] = hist.values[:, i]
        df.insert(0, 'spike_type', spike_type)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def normalize_to_interval(
    spikes: np.ndarray,
    interval: Interval
) -> np.ndarray:
    """Normalize spike times by the given interval."""
    return (spikes - interval.start) / interval.duration

def extend_interval(interval: Interval, normalized: Interval) -> Interval:
    """Extend the interval based on normalized ends."""
    return Interval(
        interval.start + normalized.start * interval.duration,
        interval.start + normalized.end * interval.duration
    )