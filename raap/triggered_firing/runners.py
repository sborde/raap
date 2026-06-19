
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.data_utils import CellWrapper
from signalizer.model.experiment.sequences import EventVector
from triggered_firing.exporters import create_result_saver
from triggered_firing.io_utils import RecordingSpikePair, load_input_data

from .plot_utils import create_figure_saver, create_ptth_plotter, create_spike_palette_for_celltype
from .calculators import EventAnalysisResult, HistogramCalculator, MultiTrialHistogram, SpikeTypeResults, calculate_ptth_for_spike_types

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
        trigger_markers = input_data.recording.get_event(self._trigger)
        triggers = np.array([marker.timestamp.time for marker in trigger_markers])

        results = calculate_ptth_for_spike_types(
            input_data.spike_data,
            triggers,
            self._spike_types,
            self._calculate
        )
        cell_result = CellWrapper(input_file.info, results)
        spike_palette = self._get_palette(input_file.info.cell_type)

        result_to_plot = remove_zero_trials(
            results,
            self._trigger_condition
        )
        n_trials = results[0].result.histogram.n_trials
        valid_trials = result_to_plot[0].result.histogram.n_trials

        cell_figure = self._plot(CellWrapper(input_file.info, result_to_plot), spike_palette)
        plt.title(
            f"{input_file.name} {self._trigger} " +
            f"(n={valid_trials}/{len(trigger_markers)})\n"
            f"P({'&'.join(self._trigger_condition)})={valid_trials/n_trials:.4f}"
        )
        self._save_result(cell_result)
        self._save_figure(cell_figure)

def remove_zero_trials(
    triggered_histograms: list[SpikeTypeResults],
    trigger_condition: list[str]
) -> list[SpikeTypeResults]:
    """Collects the triggers from the input data."""

    if trigger_condition is None:
        return triggered_histograms

    n_triggers = triggered_histograms[0].result.histogram.n_trials
    keep_flags = np.ones(n_triggers, dtype=bool)
    for type_result in triggered_histograms:
        spike_type = type_result.spike_type
        if spike_type not in trigger_condition:
            continue
        histogram = type_result.result.histogram
        keep_flags &= np.sum(histogram.values, axis=0) > 0

    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=EventAnalysisResult(
                histogram=MultiTrialHistogram(
                    values=result.histogram.values[:, keep_flags],
                    centers=result.histogram.centers,
                    data_unit=result.histogram.data_unit,
                    value_unit=result.histogram.value_unit
                ),
                trial_events=result.trial_events.copy()
            )
        )
        for spike_type, result in triggered_histograms
    ]
