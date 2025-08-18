
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.data_utils import CellWrapper
from signalizer.model.experiment.sequences import EpisodeSequence, EventVector
from triggered_firing.exporters import create_result_saver
from triggered_firing.io_utils import load_input_data

from triggered_firing.plot_utils import create_figure_saver
from .plotters import SpikeTypePaletteBuilder, create_pec_plotter
from .calculators import (
    NormalizedHistogramCalculator,
    calculate_pec_for_spike_types,
    SpikeTypeResults,
    MultiTrialNormalizedHistogram
)

class IndividualSpikeTypePECAnalysis:

    def __init__(self, config: dict):

        self._build_palette = SpikeTypePaletteBuilder(
            config['palette'],
            config['plot']['coloring_mode']
        )

        self._episode = config['params']['episode']
        self._marker = config['params']['marker']
        self._spike_types = config['params']['spike_types']
        self._marker_condition = config['params'].get('marker_condition', [])
        self._suffix = f'{self._episode}_cycle'

        self._calculate = NormalizedHistogramCalculator(
            n_bins=config['params']['histogram']['num_bins'],
            target_range=(0, 360),
            value_unit=config['params']['histogram']['value_unit'],
            data_unit='\u00B0'
        )
        self._plot = create_pec_plotter(config['plot'])

        output_path = config['output']['path']
        self._save_result = create_result_saver(output_path, self._suffix)
        self._save_figure = create_figure_saver(output_path, self._suffix)

    @property
    def suffix(self):
        """Returns the suffix for the output files."""
        return self._suffix

    def __call__(self, input_file: CellWrapper[dict[str, Path]]):

        input_data = load_input_data(input_file)
        marker_events: EventVector = input_data.recording.get_event(self._marker)
        markers = np.array([event.timestamp.time for event in marker_events])
        episodes: EpisodeSequence = input_data.recording.get_episode(self._episode)

        cycles = []
        for episode in episodes:
            episode_markers = markers[
                (markers >= episode.interval.start) &
                (markers <= episode.interval.end)
            ]
            for i in range(len(episode_markers) - 1):
                cycles.append(
                    (episode_markers[i], episode_markers[i + 1])
                )

        results = calculate_pec_for_spike_types(
            input_data.spike_data,
            np.array(cycles),
            self._spike_types,
            self._calculate
        )

        cell_result = CellWrapper(input_file.info, results)

        result_to_plot = remove_zero_trials(
            cell_result.data, self._marker_condition
        )

        n_all_cycles = results[0].result.n_trials
        valid_trials = result_to_plot[0].result.n_trials
        valid_prob = valid_trials / n_all_cycles

        spike_palette = self._build_palette(input_file.info.cell_type)
        cell_figure = self._plot(
            CellWrapper(input_file.info, result_to_plot), spike_palette
        )
        plt.title(
            f"{input_file.name} {self._episode} " +
            f"(n={valid_trials}/{n_all_cycles})\n" +
            (
                f"P({'&'.join(self._marker_condition)})={valid_prob:.4f}"
                if self._marker_condition else ""
            )
        )
        self._save_result(cell_result)
        self._save_figure(cell_figure)


def remove_zero_trials(
    histograms: list[SpikeTypeResults], conditions: list[str]
) -> list[SpikeTypeResults]:
    """Collects the triggers from the input data."""

    if conditions is None:
        return histograms

    n_triggers = histograms[0].result.n_trials
    keep_flags = np.ones(n_triggers, dtype=bool)
    for type_result in histograms:
        spike_type = type_result.spike_type
        if spike_type not in conditions:
            continue
        histogram = type_result.result
        keep_flags &= np.sum(histogram.values, axis=0) > 0

    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=MultiTrialNormalizedHistogram(
                values=histogram.values[:, keep_flags],
                centers=histogram.centers,
                data_unit=histogram.data_unit,
                value_unit=histogram.value_unit
            )
        )
        for spike_type, histogram in histograms
    ]
