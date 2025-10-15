
from pathlib import Path

import numpy as np
import pandas as pd

from pandas import DataFrame
from common.data_utils import CellWrapper
from param_classes.workflows.peritrigger_analysis import PeriTriggerAnalysisWorkflowParams
from triggered_firing.calculators import (
    HistogramCalculator,
    calculate_ptth_for_spike_types,
    SpikeTypeResults
)
from triggered_firing.exporters import convert_spike_result_to_df
from triggered_firing.io_utils import load_input_data
from workflows.run_util.io import ResultPaths

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: PeriTriggerAnalysisWorkflowParams
) -> list[CellWrapper[DataFrame]]:
    analysis = IndividualSpikeTypePTTHAnalysis(params)
    return [analysis(input_file) for input_file in input_files]


class IndividualSpikeTypePTTHAnalysis:
    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._trigger_name = params.analysis.trigger
        self._spike_types = params.analysis.spike_types
        self._trigger_condition = params.analysis.trigger_condition

        hist_params = params.analysis.histogram
        self._calculate_histograms = HistogramCalculator(
            bin_range=hist_params.bin_range,
            bin_size=hist_params.bin_size,
            data_unit=hist_params.data_unit,
            value_unit=hist_params.value_unit
        )

        self._suffix = f'{self._trigger_name}_{hist_params.value_unit}'
        self._result_dir = Path(params.output.path) / '01_results'
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def _save_result(self, result: CellWrapper[DataFrame]):
        """Saves the result DataFrame to an Excel file."""
        result_path = self._result_dir / result.cell_type
        result_path.mkdir(parents=True, exist_ok=True)
        result.data.to_excel(
            result_path / f'{result.name}_{self._suffix}_histogram.xlsx',
            index=False
        )

    def _calculate(self, spikes: DataFrame, triggers: np.ndarray) -> DataFrame:
        """Calculate histograms for the given spikes and triggers."""
        return pd.concat([
            convert_spike_result_to_df(result) for result
            in calculate_ptth_for_spike_types(
                spikes, triggers, self._spike_types, self._calculate_histograms
            )
        ])

    def __call__(self, input_file: CellWrapper[ResultPaths]) -> CellWrapper[DataFrame]:
        """Process a single input file and return the results."""
        input_data = input_file.data.load()
        trigger_markers = input_data.recording.get_event(self._trigger_name)
        triggers = np.array([marker.timestamp.time for marker in trigger_markers])
        results = self._calculate(input_data.spikes, triggers)
        cell_result = CellWrapper(input_file.info, results)
        self._save_result(cell_result)
        return cell_result
