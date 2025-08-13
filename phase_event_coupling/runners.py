
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common.data_utils import CellWrapper
from signalizer.model.experiment.sequences import EpisodeSequence, EventVector
from triggered_firing.exporters import create_result_saver
from triggered_firing.io_utils import RecordingSpikePair, load_input_data

from triggered_firing.plot_utils import create_figure_saver
from .plotters import SpikeTypePaletteBuilder, create_pec_plotter
from .calculators import NormalizedHistogramCalculator, calculate_pec_for_spike_types

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

        n_all_cycles = len(cycles)
        valid_cycles = self._collect_cycles(np.array(cycles), input_data.spike_data)

        results = calculate_pec_for_spike_types(
            input_data.spike_data,
            valid_cycles,
            self._spike_types,
            self._calculate
        )

        cell_result = CellWrapper(input_file.info, results)

        spike_palette = self._build_palette(input_file.info.cell_type)
        cell_figure = self._plot(cell_result, spike_palette)
        plt.title(
            f"{input_file.name}{self._episode} " +
            f"(n={len(valid_cycles)}/{n_all_cycles})"
            " cycle Histogram"
        )
        self._save_result(cell_result)
        self._save_figure(cell_figure)

    def _collect_cycles(self, cycles: np.ndarray, spikes: pd.DataFrame) -> np.ndarray:
        """Collects the cycles from the input data."""
        if self._marker_condition is None:
            return cycles
        selected_cycles = []
        for cycle in cycles:
            cycle_spikes = spikes[
                (spikes['timestamp'] >= cycle[0]) &
                (spikes['timestamp'] <= cycle[1])
            ]
            if all(
                condition in cycle_spikes['spike_type'].unique()
                for condition in self._marker_condition
            ):
                selected_cycles.append(cycle)
        return np.array(selected_cycles)