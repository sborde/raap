"""Workflow to perform peri-trigger time histogram analysis."""

from dataclasses import dataclass
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
    SpikeTypeResults,
    create_per_spike_type_calculator
)
from triggered_firing.exporters import convert_results_to_df_per_type

from workflows.run_util.io import ResultPaths


@dataclass
class CellPTTHAggregate:
    """A class that stores a results of a groupwise PTTH aggregation."""
    histograms: DataFrame
    maxpoints: DataFrame


def run_analysis(
    input_files: list[CellWrapper[ResultPaths]],
    params: PeriTriggerAnalysisWorkflowParams
) -> list[CellWrapper[list[SpikeTypeResults]]]:
    """Run the peri-trigger analysis workflow on the given input files."""
    analysis = IndividualSpikeTypePTTHAnalysis(params)
    return [analysis(input_file) for input_file in input_files]

def run_cellwise_aggregation(
    cell_results: list[CellWrapper[list[SpikeTypeResults]]],
    group_name: str,
    params: PeriTriggerAnalysisWorkflowParams
) -> tuple[str, DataFrame]:
    """Run the cell-wise aggregation workflow on the given input files."""
    aggregator = CellwisePTTHAggregator(params)
    aggregation = aggregator(cell_results, group_name)
    return (group_name, aggregation)

def run_detection(
    input_files: list[CellWrapper[list[SpikeTypeResults]]],
    group_name: str,
    params: PeriTriggerAnalysisWorkflowParams
) -> tuple[str, CellPTTHAggregate]:
    """Run the cell-wise aggregation workflow on the given input files."""
    aggregator = CellwisePTTHLatencyDetection(params)
    aggregation = aggregator(input_files, group_name)
    return (group_name, aggregation)

def run_testing(
    groups: list[tuple[str, CellPTTHAggregate]],
    params: PeriTriggerAnalysisWorkflowParams
):
    """Run statistical testing on the given groups."""
    test_results = []
    tester = PairedTester(params)
    for group_name, data in tqdm(groups, total=len(groups), desc='Testing groups'):
        result = tester(data.maxpoints, group_name)
        test_results.append(result)
    result = DataFrame(test_results)
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
        self._calculate = create_per_spike_type_calculator(params.analysis)
        self._suffix = (
            f'{params.analysis.trigger}' +
            f'_{params.analysis.histogram.value_unit}')
        self._result_dir = Path(params.output.path) / '01_results' / 'individual'
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, input_file: CellWrapper[ResultPaths]) -> CellWrapper[list[SpikeTypeResults]]:
        """Process a single input file and return the results."""
        input_data = input_file.data.load()
        spikes = input_data.spikes
        triggers = input_data.get_markers(self._trigger_name)
        histograms = self._calculate(spikes, triggers)
        cell_result = CellWrapper(input_file.info, histograms)
        self._save_result(cell_result)
        return cell_result

    def _save_result(self, result: CellWrapper[list[SpikeTypeResults]]):
        """Saves the result DataFrame to an Excel file."""
        result_path = self._result_dir / result.cell_type
        result_path.mkdir(parents=True, exist_ok=True)
        result_frame = convert_results_to_df_per_type(result.data)
        result_frame.to_excel(
            result_path / f'{result.name}_{self._suffix}_histogram.xlsx',
            index=False
        )


class CellwisePTTHAggregator:
    """Class to aggregate peri-trigger time histogram results within a single cell."""

    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._palette = params.palette
        self._show_raster = params.analysis.show_raster
        self._raster_size = params.analysis.raster_size
        self._raster_band_color = params.analysis.raster_band_color
        self._spike_types = params.analysis.spike_types
        self._trigger_condition = params.grouping.trigger_condition
        self._xlim = params.analysis.histogram.bin_range
        self._ylim = params.analysis.ylim
        self._suffix = (
            f'{params.analysis.trigger}' +
            f'_{params.analysis.histogram.value_unit}' +
            '_mean'
        )
        base_path = Path(params.output.path)
        self._paths = {
            'results': base_path / '01_results' / 'grouped',
            'cellwise_figures': base_path / '02_figures' / 'individual',
            'groupwise_figures': base_path / '02_figures' / 'groupwise'
        }
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        input_files: list[CellWrapper[list[SpikeTypeResults]]],
        group_name: str
    ) -> DataFrame:
        """Aggregate results from multiple input files."""
        aggregation = self._aggregate_per_cell(input_files)

        self._save_result(aggregation, group_name)
        for cell_result in input_files:
            self._plot_per_cell(cell_result, group_name)
        return aggregation

    def _aggregate_per_cell(self, cells: list[CellWrapper[list[SpikeTypeResults]]]) -> DataFrame:
        """Aggregate values within a single cell."""
        cell_frames = [
            CellWrapper(
                cell_result.info,
                convert_results_to_df_per_type(cell_result.data)
            ) for cell_result in cells
        ]
        values = {}
        for cell_frame in cell_frames:
            cell_values = self._get_values(cell_frame.data)
            cellwise_mean = np.mean(cell_values, axis=1)
            values[cell_frame.name] = cellwise_mean
        return DataFrame(
            {
                'spike_type': cell_frames[0].data['spike_type'],
                'centers': cell_frames[0].data['centers'],
                **values
            }
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

    def _aggregate_per_group(self, cells: DataFrame) -> DataFrame:
        """Aggregate values"""
        values = cells.iloc[:, 2:].to_numpy()
        mean_values = np.mean(values, axis=1)
        sd_values = np.std(values, axis=1)
        sem_values = sd_values / np.sqrt(values.shape[1])
        return DataFrame({
            'spike_types': cells['spike_type'],
            'centers': cells['centers'],
            'mean': mean_values,
            'sem': sem_values
        })

    def _save_result(self, data: DataFrame, name: str):
        """Saves the aggregated result DataFrame to an Excel file."""
        data.to_excel(
            self._paths['results'] / f'{name}_{self._suffix}_histograms.xlsx',
            index=False
        )

    def _plot_per_cell(
        self,
        cell_result: CellWrapper[list[SpikeTypeResults]],
        group_name: str
    ):
        """Plot the detected maximum points for each cell."""
        group_fig_dir = self._paths['cellwise_figures'] / group_name
        group_fig_dir.mkdir(parents=True, exist_ok=True)

        spike_colors = {
            'AIS-AP': self._palette['cell_type'][group_name],
            'RA-AP': self._palette['spike_type']['RA-AP']
        }

        if self._show_raster:
            fig, axes = plt.subplots(
                2, 1, figsize=(10, 10),
                constrained_layout=True,
                sharex=True
            )
            raster_ax = axes[0]
            for spike_result in cell_result.data:
                if spike_result.spike_type != 'RA-AP':
                    continue
                for i, events in enumerate(spike_result.result.trial_events):
                    if len(events) == 0:
                        continue
                    raster_ax.fill_between(
                        self._xlim,
                        [i - self._raster_size / 2, i - self._raster_size / 2],
                        [i + self._raster_size / 2, i + self._raster_size / 2],
                        color=self._raster_band_color,
                        alpha=0.1,
                        edgecolor='none'
                    )
            for spike_result in cell_result.data:
                color = spike_colors[spike_result.spike_type]
                for i, events in enumerate(spike_result.result.trial_events):
                    raster_ax.vlines(
                        events,
                        ymin=i - self._raster_size / 2,
                        ymax=i + self._raster_size / 2,
                        color=color
                    )
                raster_ax.set_ylim((
                    -self._raster_size/2,
                    self._raster_size/2 + len(spike_result.result.trial_events)
                ))
            raster_ax.set_yticks([])
            hist_ax = axes[1]
        else:
            fig, hist_ax = plt.subplots(1, 1, figsize=(10, 5))

        histograms = convert_results_to_df_per_type(cell_result.data)
        selected_values = self._get_values(histograms)
        for spike_type in self._spike_types:

            if spike_type == 'AIS-AP':
                color = self._palette['cell_type'][group_name]
            else:
                color = self._palette['spike_type']['RA-AP']

            flag = histograms['spike_type'] == spike_type
            centers = histograms.loc[flag, 'centers']

            values = selected_values[flag, :]
            mean_values = np.mean(values, axis=1)
            sd_values = np.std(values, axis=1)
            sem_values = sd_values / np.sqrt(values.shape[1])

            hist_ax.fill_between(
                centers,
                mean_values - sem_values,
                mean_values + sem_values,
                color=spike_colors[spike_type],
                alpha=0.3
            )
            hist_ax.plot(
                centers,
                mean_values,
                '-',
                color=spike_colors[spike_type],
                label=spike_type
            )

        hist_ax.set_ylim(self._ylim)
        hist_ax.set_xlabel('Time (ms)')
        hist_ax.set_title(
            f'n={selected_values.shape[1]}/{histograms.shape[1] - 2}'
        )
        hist_ax.set_xlim(self._xlim)
        hist_ax.legend()
        plt.suptitle(cell_result.name)

        figure_name = f'{cell_result.name}_{self._suffix}_histogram'
        fig.savefig(
            group_fig_dir / f'{figure_name}.png',
            dpi=100,
            format='png'
        )
        with plt.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(
                group_fig_dir / f'{figure_name}.svg',
                format='svg'
            )
        plt.close(fig)


class CellwisePTTHLatencyDetection:
    """Class to aggregate peri-trigger time histogram results across multiple cells."""

    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._palette = params.palette
        self._spike_types = params.analysis.spike_types
        self._trigger_condition = params.grouping.trigger_condition
        self._detection_start = params.grouping.detection_start
        self._ylim = params.analysis.ylim
        self._suffix = (
            f'{params.analysis.trigger}' +
            f'_{params.analysis.histogram.value_unit}_mean'
        )
        self._paths = {
            'results': Path(params.output.path) / '01_results' / 'grouped',
            'cellwise_figures': Path(params.output.path) / '02_figures' / 'individual',
            'groupwise_figures': Path(params.output.path) / '02_figures' / 'groupwise'
        }
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        input_files: list[CellWrapper[list[SpikeTypeResults]]],
        group_name: str
    ) -> CellPTTHAggregate:
        """Aggregate results from multiple input files."""
        aggregation = self._aggregate_per_cell(input_files)
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
        self._plot_grouped_detection(aggregation, max_points, group_name)
        return CellPTTHAggregate(aggregation, max_points)

    def _aggregate_per_cell(self, cells: list[CellWrapper[list[SpikeTypeResults]]]) -> DataFrame:
        """Aggregate values within a single cell."""
        cell_frames = [
            CellWrapper(
                cell_result.info,
                convert_results_to_df_per_type(cell_result.data)
            ) for cell_result in cells
        ]
        values = {}
        for cell_frame in cell_frames:
            cell_values = self._get_values(cell_frame.data)
            cellwise_mean = np.mean(cell_values, axis=1)
            values[cell_frame.name] = cellwise_mean
        return DataFrame(
            {
                'spike_type': cell_frames[0].data['spike_type'],
                'centers': cell_frames[0].data['centers'],
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

    def _groupwise_aggregate(self, cells: DataFrame) -> DataFrame:
        """Aggregate values"""
        values = cells.iloc[:, 2:].to_numpy()
        mean_values = np.mean(values, axis=1)
        sd_values = np.std(values, axis=1)
        sem_values = sd_values / np.sqrt(values.shape[1])
        return DataFrame({
            'spike_types': cells['spike_type'],
            'centers': cells['centers'],
            'mean': mean_values,
            'sem': sem_values
        })

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
            for spike_type in self._spike_types:
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
            ax.set_ylim(self._ylim)
            ax.set_xlabel('Time (ms)')
            ax.set_title(cell_name)
            ax.legend()

            plt.tight_layout()
            figure_name = f'{cell_name}_{self._suffix}_detection'
            fig.savefig(
                group_fig_dir / f'{figure_name}.png',
                dpi=100,
                format='png'
            )
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(
                    group_fig_dir / f'{figure_name}.svg',
                    format='svg'
                )
            plt.close(fig)

    def _plot_grouped_detection(
        self,
        histograms: DataFrame,
        points: DataFrame,
        group_name: str
    ):
        """Plots the group average histogram per spike-type and the detected points."""
        groupwise_aggregate = self._groupwise_aggregate(histograms)

        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.2, 0.8], hspace=0.05)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_top.set_axis_off()
        for spike_type in self._spike_types:
            spike_data = groupwise_aggregate[
                groupwise_aggregate['spike_types'] == spike_type
            ]
            if spike_type == 'AIS-AP':
                color = self._palette['cell_type'][group_name]
            else:
                color = self._palette['spike_type']['RA-AP']

            ax_bottom.fill_between(
                spike_data['centers'],
                spike_data['mean'] - spike_data['sem'],
                spike_data['mean'] + spike_data['sem'],
                color=color,
                alpha=0.3
            )
            ax_bottom.plot(
                spike_data['centers'],
                spike_data['mean'],
                color=color,
                label=spike_type
            )

            ax_bottom.set_ylim(self._ylim)
            ax_bottom.set_xlabel('Time (ms)')
            ax_bottom.legend()

        sns.boxplot(
            data=points,
            x='time',
            y='spike_type',
            hue='spike_type',
            orient='h',
            palette=[self._palette['cell_type'][group_name], self._palette['spike_type']['RA-AP']],
            ax=ax_top
        )
        figure_name = f'{group_name}_{self._suffix}_detection'
        fig.savefig(
            self._paths['groupwise_figures'] / f'{figure_name}.png',
            dpi=100,
            format='png'
        )
        with plt.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(
                self._paths['groupwise_figures'] / f'{figure_name}.svg',
                format='svg'
            )
        plt.close(fig)


class PairedTester:
    """Class to perform statistical testing on paired data."""
    def __init__(self, params: PeriTriggerAnalysisWorkflowParams):
        self._spike_types = params.analysis.spike_types
        self._palette = params.palette
        self._ylim = params.grouping.boxplot_ylim
        self._paths = {
            'figures': Path(params.output.path) / '02_figures' / 'statistics'
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

        ax.set_ylim(self._ylim)
        ax.set_ylabel('Time (ms)')
        ax.set_title(group_name)
        ax.legend([], [], frameon=False)
        plt.tight_layout()

        fig_name = self._paths['figures'] / f'{group_name}_comparison'
        fig.savefig(f'{fig_name}.png', dpi=100, format='png')
        with plt.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(f'{fig_name}.svg', format='svg')
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
