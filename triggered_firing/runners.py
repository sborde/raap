
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.data_utils import CellWrapper
from signalizer.model.experiment.sequences import EventVector
from triggered_firing.exporters import create_result_saver
from triggered_firing.io_utils import RecordingSpikePair, load_input_data

from .plot_utils import create_figure_saver, create_ptth_plotter, create_spike_palette_for_celltype
from .calculators import HistogramCalculator, calculate_ptth_for_spike_types

class IndividualSpikeTypePTTHAnalysis:

    def __init__(self, config: dict):

        self._get_palette = lambda cell_type: (
            create_spike_palette_for_celltype(
                cell_type,
                config['plot']['coloring_mode'],
                config['palette']
            )
        )

        self._trigger = config['params']['trigger']
        self._spike_types = config['params']['spike_types']
        self._trigger_condition = config['params'].get('trigger_condition', [])

        self._window_size = np.array(config['params']['histogram']['bin_range'])
        self._data_unit = config['params']['histogram']['data_unit']

        self._calculate = HistogramCalculator(**config['params']['histogram'])
        self._plot = create_ptth_plotter(self._trigger, config['plot'])

        output_path = config['output']['path']
        self._suffix = f'{self._trigger}_{self._calculate.value_unit}'
        self._save_result = create_result_saver(output_path, self._suffix)
        self._save_figure = create_figure_saver(output_path, self._suffix)

    @property
    def suffix(self):
        """Returns the suffix for the output files."""
        return self._suffix

    def __call__(self, input_file: CellWrapper[dict[str, Path]]):

        input_data = load_input_data(input_file)
        all_triggers = input_data.recording.get_event(self._trigger)
        triggers = self._collect_triggers(input_data)

        results = calculate_ptth_for_spike_types(
            input_data.spike_data,
            triggers,
            self._spike_types,
            self._calculate
        )

        cell_result = CellWrapper(input_file.info, results)
        spike_palette = self._get_palette(input_file.info.cell_type)
        cell_figure = self._plot(cell_result, spike_palette)
        plt.title(
            f"{input_file.name}{self._trigger}" +
            f"(n={len(triggers)}/{len(all_triggers)})"
            "-triggered Histogram"
        )
        self._save_result(cell_result)
        self._save_figure(cell_figure)

    def _collect_triggers(self, input_data: RecordingSpikePair) -> np.ndarray:
        """Collects the triggers from the input data."""
        spikes = input_data.spike_data
        events: EventVector = input_data.recording.get_event(self._trigger)
        triggers = np.array([trigger.timestamp.time for trigger in events])
        if self._trigger_condition is None:
            return triggers
        selected_triggers = []
        window_sec = self._window_size / 1000 if self._data_unit == 'ms' else self._window_size
        for trigger in triggers:
            triggered_window = window_sec + trigger
            triggered_spikes = spikes[
                (spikes['timestamp'] >= triggered_window[0]) &
                (spikes['timestamp'] <= triggered_window[1])
            ]
            if all(
                condition in triggered_spikes['spike_type'].unique()
                for condition in self._trigger_condition
            ):
                selected_triggers.append(trigger)
        return np.array(selected_triggers)