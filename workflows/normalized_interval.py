

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
from triggered_firing.calculators import MultiTrialHistogram
from workflows.normalized_inter_raap import Interval
from workflows.run_util.io import ResultPaths

EventSequenceList = list[np.ndarray]
EventHistogramSequencePair = tuple[MultiTrialHistogram, EventSequenceList]
SpikeTypeGroupedResults = dict[str, EventHistogramSequencePair]
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
        self._episode_names = params.episodes
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
        cell_data = cell.data.load()
        results = self._calculate(cell_data)
        cell_result = CellWrapper(cell.info, results)
        self._plot(cell_result)
        self._save_result(cell_result)
        return cell_result

    def _calculate(
        self, cell_data: RecordingWithSpikes
    ) -> EpisodeRelatedResults:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        spikes = cell_data.spikes
        results = {}
        for episode_name in self._episode_names:
            starts, ends = cell_data.get_episode_ends(episode_name)
            intervals = [Interval(start, end) for start, end in zip(starts, ends)]
            episode_results = {}
            for spike_type in ['RA-AP', 'AIS-AP']:
                type_results = {'values': [], 'events': []}
                timestamps = select_spikes(spikes, spike_type)
                for interval in intervals:
                    cut_events = self._collect_normalized_events(timestamps, interval)
                    type_results['values'].append(self._count_in_interval(cut_events))
                    type_results['events'].append(cut_events[1])
                episode_results[spike_type] = type_results
            results[episode_name] = {
                spike_type: (
                    MultiTrialHistogram(
                        self._centers,
                        np.stack(type_result['values'], axis=1),
                        '',
                        self._value_unit
                    ), type_result['events']
                ) for spike_type, type_result in episode_results.items()
            }
        return results

    def _collect_normalized_events(
        self, spikes: np.ndarray, interval: Interval
    ) -> tuple[Interval, np.ndarray]:
        """Collect normalized spike times within the given interval."""
        cut_interval = extend_interval(interval, self._cut_window)
        spikes_within = cut_interval.contained(spikes)
        return interval, normalize_to_interval(spikes_within, interval)

    def _count_in_interval(
        self, spikes_within: tuple[Interval, np.ndarray]
    ) -> np.ndarray:
        """Count the number of spikes within the given interval."""
        values = np.histogram(spikes_within[1], bins=self._bins)[0]
        if self._value_unit == 'freq':
            values = values / (spikes_within[0].duration * self._bin_size)
        elif self._value_unit == 'prob':
            total_spikes = len(spikes_within[1])
            if total_spikes > 0:
                values = values / total_spikes
        return values

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
            for spike_type, (hist, norm_events) in episode_results.items():
                if spike_type == 'RA-AP':
                    color = self._palette['spike_type'][spike_type]
                else:
                    color = cell_color

                for i, events in enumerate(norm_events):
                    raster_ax.vlines(
                        events,
                        i-0.25,
                        i+0.25,
                        color=color
                    )

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
            frames = []
            for spike_type, (hist, _) in episode_result.items():
                df = DataFrame({'centers': hist.centers})
                for i in range(hist.values.shape[1]):
                    df[f'{episode_name}_{i+1}'] = hist.values[:, i]
                df.insert(0, 'spike_type', spike_type)
                frames.append(df)
            pool_frame = pd.concat(frames, ignore_index=True)
            pool_frame.to_excel(result_path, index=False)

def select_spikes(spikes: DataFrame, type_name: str) -> np.ndarray:
    """Select spikes of the given type."""
    return spikes[spikes['spike_type'] == type_name]['timestamp'].to_numpy()

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