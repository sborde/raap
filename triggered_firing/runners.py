
from pathlib import Path

import numpy as np

from common.data_utils import CellWrapper
from signalizer.model.experiment.sequences import EventVector
from triggered_firing.exporters import create_result_saver
from triggered_firing.io_utils import load_input_data

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

        self._calculate = HistogramCalculator(**config['params']['histogram'])
        self._plot = create_ptth_plotter(self._trigger, config['plot'])

        output_path = config['output']['path']
        self._save_result = create_result_saver(output_path, self._trigger)
        self._save_figure = create_figure_saver(output_path, self._trigger)

    def __call__(self, input_file: CellWrapper[dict[str, Path]]):

        input_data = load_input_data(input_file)
        triggers = np.array([
            trigger.timestamp.time
            for trigger
            in input_data.recording.get_event(self._trigger)
        ])

        results = calculate_ptth_for_spike_types(
            input_data.spike_data,
            triggers,
            self._spike_types,
            self._calculate
        )

        cell_result = CellWrapper(input_file.info, results)
        spike_palette = self._get_palette(input_file.info.cell_type)
        cell_figure = self._plot(cell_result, spike_palette)
        self._save_result(cell_result)
        self._save_figure(cell_figure)
