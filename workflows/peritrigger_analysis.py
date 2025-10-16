"""Workflow to perform peri-trigger time histogram analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import wilcoxon
from pandas import DataFrame

from common.data_utils import CellWrapper
from param_classes.workflows.peritrigger_analysis import PeriTriggerAnalysisWorkflowParams
from triggered_firing.calculators import (
    HistogramCalculator,
    HistogramCalculatorGaussSmoother,
    calculate_ptth_for_spike_types
)
from triggered_firing.exporters import convert_spike_result_to_df

from workflows.run_util.io import ResultPaths

def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: PeriTriggerAnalysisWorkflowParams
) -> list[CellWrapper[DataFrame]]:
    """Run the peri-trigger analysis workflow on the given input files."""
    analysis = IndividualSpikeTypePTTHAnalysis(params)
    return [analysis(input_file) for input_file in input_files]

def run_aggregation(
    input_files: list[CellWrapper[DataFrame]],
    group_name: str,
    params: PeriTriggerAnalysisWorkflowParams
) -> tuple[str, DataFrame]:
    """Run the cell-wise aggregation workflow on the given input files."""
    aggregator = SpikeTypePTTHLatencyDetection(params)
    max_points = aggregator(input_files, group_name)
    return (group_name, max_points)

def run_testing(
    groups: list[tuple[str, DataFrame]],
    params: PeriTriggerAnalysisWorkflowParams
):
    """Run statistical testing on the given groups."""
    test_results = []
    tester = PairedTester(params)
    for group_name, data in tqdm(groups, total=len(groups), desc='Testing groups'):
        result = tester(data, group_name)
        test_results.append(result)
    result = pd.DataFrame(test_results)
    result = result[['group', 'n', 'diff_median', 'pvalue']]
    stats_path = Path(params.output.path) / '03_statistics'
    stats_path.mkdir(parents=True, exist_ok=True)
    result.to_excel(
        stats_path / 'paired_test_results.xlsx', index=False
    )

class IndividualSpikeTypePTTHAnalysis:
    """Class to perform peri-trigger time histogram analysis for individual spike types."""
    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._trigger_name = params.analysis.trigger
        self._spike_types = params.analysis.spike_types

        hist_params = params.analysis.histogram
        self._calculate_histograms = HistogramCalculator(
            hist_params.bin_range,
            hist_params.bin_size,
            hist_params.value_unit,
            hist_params.data_unit
        )
        if params.analysis.smooth_size > 0:
            self._calculate_histograms = HistogramCalculatorGaussSmoother(
                self._calculate_histograms,
                params.analysis.smooth_size
            )

        self._suffix = f'{self._trigger_name}_{hist_params.value_unit}'
        self._result_dir = Path(params.output.path) / '01_results' / 'individual'
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
        histograms = calculate_ptth_for_spike_types(
            spikes, triggers, self._spike_types, self._calculate_histograms
        )
        return pd.concat([
            convert_spike_result_to_df(result) for result
            in histograms
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


class SpikeTypePTTHLatencyDetection:
    """Class to aggregate peri-trigger time histogram results across multiple cells."""

    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._palette = params.palette
        self._spike_types = params.analysis.spike_types
        self._trigger_condition = params.grouping.trigger_condition
        self._detection_start = params.grouping.detection_start
        self._suffix = (
            f'{params.analysis.trigger}' +
            f'_{params.analysis.histogram.value_unit}_mean'
        )
        self._paths = {
            'results': Path(params.output.path) / '01_results' / 'grouped',
            'figures': Path(params.output.path) / '02_figures',
            'cellwise_figures': Path(params.output.path) / '02_figures' / 'individual'
        }
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def _save_result(self, result: CellWrapper[DataFrame]):
        """Saves the aggregated result DataFrame to an Excel file."""
        result.data.to_excel(
            self._paths['results'] / f'{result.name}_{self._suffix}_histogram.xlsx',
            index=False
        )

    def _get_values(self, frame: DataFrame) -> np.ndarray:
        """Filter the DataFrame based on the trigger condition."""
        all_values = frame.iloc[:, 2:].to_numpy()
        if not self._trigger_condition:
            return all_values
        spike_type_vector = frame['spike_type']
        flag = np.ones(all_values.shape[1], dtype=bool)
        for condition in self._trigger_condition:
            flag &= (
                np.sum(all_values[spike_type_vector == condition, :], axis=0) > 0
            )
        return all_values[:, flag]

    def _aggregate(self, cells: list[CellWrapper[DataFrame]]) -> DataFrame:
        """Aggregate values within a single cell."""
        values = {}
        for cell in cells:
            cell_values = self._get_values(cell.data)
            values[cell.name] = np.mean(cell_values, axis=1)
        return pd.DataFrame(
            {
                'spike_type': cells[0].data['spike_type'],
                'centers': cells[0].data['centers'],
                **values
            }
        )

    def _find_max_points(self, frame: DataFrame) -> DataFrame:
        """Find the maximum time value in the DataFrame."""
        centers = frame['centers'].to_numpy()
        points = []
        for cell_name in frame.columns[2:]:
            values = frame[cell_name].to_numpy()
            for spike_type in self._spike_types:
                type_values = values[frame['spike_type'] == spike_type]
                type_centers = centers[frame['spike_type'] == spike_type]

                detection_part = type_centers >= self._detection_start
                type_centers = type_centers[detection_part]
                type_values = type_values[detection_part]

                max_index = np.argmax(type_values)
                points.append({
                    'cell': cell_name,
                    'spike_type': spike_type,
                    'time': type_centers[max_index],
                    'value': type_values[max_index]
                })
        return DataFrame(points)

    def _plot_cellwise_detection(
        self,
        histograms: DataFrame,
        points: DataFrame,
        group_name: str
    ):
        """Plot the detected maximum points for each cell."""
        group_fig_dir = self._paths['cellwise_figures'] / group_name
        group_fig_dir.mkdir(parents=True, exist_ok=True)

        cell_names = histograms.columns[2:]
        for cell_name in cell_names:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            cell_data = histograms[['spike_type', 'centers', cell_name]]
            cell_points = points[points['cell'] == cell_name]
            for i, spike_type in enumerate(self._spike_types):
                spike_data = cell_data[cell_data['spike_type'] == spike_type]
                spike_points = cell_points[cell_points['spike_type'] == spike_type]

                if spike_type == 'AIS-AP':
                    color = self._palette['cell_type'][group_name]
                else:
                    color = self._palette['spike_type']['RA-AP']

                ax.plot(
                    spike_data['centers'],
                    spike_data[cell_name],
                    color=color,
                    label=spike_type
                )
                ax.scatter(
                    spike_points['time'],
                    spike_points['value'],
                    color='black',
                    marker='x',
                    s=50
                )
            ax.set_xlabel('Time (ms)')
            ax.set_title(cell_name)
            ax.legend()

            plt.tight_layout()
            fig.savefig(
                group_fig_dir / f'{cell_name}_{self._suffix}_detection.png'
            )
            plt.close(fig)

    def __call__(
        self, input_files: list[CellWrapper[DataFrame]], group_name: str
    ) -> DataFrame:
        """Aggregate results from multiple input files."""
        aggregation = self._aggregate(input_files)
        max_points = self._find_max_points(aggregation)
        aggregation.to_excel(
            self._paths['results'] / f'{group_name}_{self._suffix}_histograms.xlsx',
            index=False
        )
        max_points.to_excel(
            self._paths['results'] / f'{group_name}_{self._suffix}_max_points.xlsx',
            index=False
        )

        self._plot_cellwise_detection(aggregation, max_points, group_name)
        return max_points


class PairedTester:
    """Class to perform statistical testing on paired data."""
    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._spike_types = params.analysis.spike_types
        self._palette = params.palette
        self._paths = {
            'figures': Path(params.output.path) / '02_figures'
        }
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def _create_pivot(self, data: DataFrame) -> DataFrame:
        """Pivot the DataFrame to have spike types as columns."""
        return pd.pivot(
            data, index='cell', columns='spike_type', values='time'
        ).dropna()

    def _plot_comparison(self, data: DataFrame, group_name: str):
        """Plot a comparison of the two spike types."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.boxplot(
            data=data,
            x='spike_type',
            y='time',
            hue='spike_type',
            legend=False,
            palette=[self._palette['cell_type'][group_name], self._palette['spike_type']['RA-AP']],
            fill=False,
            fliersize=0,
            width=0.5,
            ax=ax
        )
        pivot = self._create_pivot(data)
        for _, row in pivot.iterrows():
            ax.plot(
                self._spike_types,
                [row[self._spike_types[0]], row[self._spike_types[1]]],
                color='#808080',
                alpha=0.5
            )
            ax.scatter(
                self._spike_types,
                [row[self._spike_types[0]], row[self._spike_types[1]]],
                marker='o',
                edgecolor='none',
                facecolor='#808080',
                alpha=0.7
            )

        ax.set_ylabel('Time (ms)')
        ax.set_title(group_name)
        ax.legend([], [], frameon=False)
        plt.tight_layout()
        fig_path = self._paths['figures'] / f'{group_name}_comparison.png'
        fig.savefig(fig_path)
        plt.close(fig)

    def __call__(self, data: DataFrame, group_name: str) -> dict:
        """Perform statistical testing on the given DataFrame."""
        test_frame = self._create_pivot(data)
        stat_result = wilcoxon(
            test_frame[self._spike_types[0]],
            test_frame[self._spike_types[1]]
        )
        self._plot_comparison(data, group_name)
        return {
            'group': group_name,
            'n': len(test_frame),
            'diff_median': np.median(
                test_frame[self._spike_types[0]] -
                test_frame[self._spike_types[1]]
            ),
            'pvalue': stat_result.pvalue, # type: ignore
        }
