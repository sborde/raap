"""
Compare slowdown features of two different datasets.
"""

from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from collections import namedtuple

SpikeInfo = namedtuple('SpikeInfo', ['cell_name', 'spike_type', 'group_name'])


class InVitroFileNameProcessor:
    """
    Process the file name of a CSV file that contains spike features of invitro data.
    """
    def __init__(self, skip_files: list[str]):
        self._skip_files = skip_files

    def __call__(self, file_name: str) -> SpikeInfo:
        parts = file_name.split('_')
        cell_name = '_'.join(parts[:2])
        spike_type = parts[2].replace('juxta', '')

        if cell_name in self._skip_files:
            return None
        group_name = None
        return SpikeInfo(cell_name, spike_type, group_name)


class InvivoFileNameProcessor:
    """
    Process the file name of a CSV file that contains spike features of invivo data.
    """
    def __init__(self, skip_files: list[str], group_list: list[str]):
        self._skip_files = skip_files
        self._group_list = group_list

    def __call__(self, file_name: str) -> SpikeInfo:
        parts = file_name.split('_')
        cell_type = parts[0]
        group_name = '_'.join(parts[:2])
        cell_name = '_'.join(parts[2:-2])

        if cell_name in self._skip_files or group_name not in self._group_list:
            return None

        return SpikeInfo(cell_name, cell_type, group_name)


def load_feature_dataframe(
        file_path: Path,
        spike_info: SpikeInfo
) -> pd.DataFrame:
    """
    Load the feature dataframe from the given CSV file.
    """
    data_frame = pd.read_csv(file_path)
    data_frame['slowdown_index'] = data_frame['slowdown_index'] / 100
    data_frame['cell_name'] = spike_info.cell_name
    data_frame['spike_type'] = spike_info.spike_type

    return data_frame

def load_spike_feature_dataset(
        input_dir: Path,
        process_file_name: Callable[[str], SpikeInfo]
) -> pd.DataFrame:
    """
    Load the spike features from the given directory.
    """
    data_frames = []
    for file_path in input_dir.glob('*.csv'):
        spike_info = process_file_name(file_path.stem)

        if spike_info is None:
            continue

        data_frames.append(load_feature_dataframe(file_path, spike_info))
    return pd.concat(data_frames)

def calculate_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate the error metrics.
    """
    true_positives = np.sum(y_true & y_pred)
    true_negatives = np.sum(~y_true & ~y_pred)
    false_positives = np.sum(~y_true & y_pred)
    false_negatives = np.sum(y_true & ~y_pred)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)
    tpr = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'f1_score': f1_score
    }

def calculate_error_rate_for_slowdown_thresholds(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the error rate for different slowdown index thresholds.
    """
    index_values = data_frame['slowdown_index'].values
    unique_values = np.sort(np.unique(index_values))

    y_true = np.array([
        1 if spike_type == 'RAAP' else 0
        for spike_type in data_frame['spike_type']
    ])

    errors = []
    for threshold in unique_values:
        y_pred = index_values >= threshold

        error_metrics = calculate_error_metrics(y_true, y_pred)
        errors.append({
            'threshold': threshold,
            **error_metrics
        })

    return pd.DataFrame(errors)

def calculate_error_rate_for_two_feature_thresholds(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the error rate for threshold classifiers using two features.
    """
    x1 = data_frame['relative_dv_drop'].values
    x2 = data_frame['relative_rebound_time'].values

    x1_unique = np.sort(np.unique(x1))
    x2_unique = np.sort(np.unique(x2))

    y_true = np.array([
        1 if spike_type == 'RAAP' else 0
        for spike_type in data_frame['spike_type']
    ])

    errors = []
    for x1_threshold in x1_unique:
        for x2_threshold in x2_unique:
            y_pred = (x1 >= x1_threshold) & (x2 >= x2_threshold)

            error_metrics = calculate_error_metrics(y_true, y_pred)
            errors.append({
                'x1_threshold': x1_threshold,
                'x2_threshold': x2_threshold,
                **error_metrics
            })

    return pd.DataFrame(errors)

def calculate_crossing_threshold(data_frame: pd.DataFrame) -> float:
    """
    Calculate the crossing threshold.
    """
    precision_scores = data_frame['precision'].values
    recall_scores = data_frame['recall'].values

    position_change = np.diff(np.sign(precision_scores - recall_scores))
    position_change_index = np.where(position_change != 0)[0]

    return float(data_frame['threshold'].values[position_change_index][0])

def calculate_crossing_threshold_for_two_feature_thresholds(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the crossing threshold for two feature thresholds.
    """

    results = []
    for x1_threshold in data_frame['x1_threshold'].unique():
        sub_data_frame = data_frame[data_frame['x1_threshold'] == x1_threshold]

        x2_thresholds = sub_data_frame['x2_threshold'].values
        precision_scores = sub_data_frame['precision'].values
        recall_scores = sub_data_frame['recall'].values

        position_change = np.diff(np.sign(precision_scores - recall_scores))

        crossing_index = np.where(position_change != 0)[0]

        results.append({
            'x1_threshold': x1_threshold,
            'x2_threshold': x2_thresholds[crossing_index[0]] if len(crossing_index) > 0 else np.nan,
            'precision': precision_scores[crossing_index[0]] if len(crossing_index) > 0 else np.nan,
            'recall': recall_scores[crossing_index[0]] if len(crossing_index) > 0 else np.nan,
            'f1_score': sub_data_frame['f1_score'].values[crossing_index[0]] if len(crossing_index) > 0 else np.nan
        })

    result_frame = pd.DataFrame(results)
    result_frame['mean_errors'] = np.mean(
        np.stack([
            result_frame['precision'].values,
            result_frame['recall'].values
        ], axis=0),
        axis=0
    )

    return result_frame

def plot_error_rate_for_thresholds(data_frame: pd.DataFrame, ax: plt.Axes):
    """
    Plot the error rate for different thresholds.
    """
    precision_scores = data_frame['precision'].values
    recall_scores = data_frame['recall'].values

    precision_recall_diff = np.sign(precision_scores - recall_scores)
    position_change = np.diff(precision_recall_diff)
    position_change_index = np.where(position_change != 0)[0]
    position_change_threshold = float(data_frame['threshold'].values[position_change_index][0])

    sns.lineplot(x='threshold', y='f1_score', data=data_frame, label='F1 score', ax=ax)
    sns.lineplot(x='threshold', y='precision', data=data_frame, label='Precision', ax=ax)
    sns.lineplot(x='threshold', y='recall', data=data_frame, label='Recall', ax=ax)
    ax.vlines(position_change_threshold, 0, 1,
              color='black',
              linestyles='dashed',
              label='Crossing threshold'
              )
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error rate')
    ax.set_title('Error rate for different thresholds, ' +
                 f'crossing: {position_change_threshold:.4f}')

def plot_error_rate_for_two_features_thresholds(metrics_data: pd.DataFrame, ax: plt.Axes):
    """
    Plot the error rate for different two feature thresholds.
    """

    sns.scatterplot(
        x='x1_threshold',
        y='x2_threshold',
        data=metrics_data,
        hue='mean_errors',
        palette='viridis_r',
        ax=ax
    )

    ax.set_xlabel('Relative dv drop threshold')
    ax.set_ylabel('Relative rebound time threshold')
    ax.set_title('Precision at the crossing point for different thresholds')

def plot_auc_curves(data_frame: pd.DataFrame, ax: plt.Axes):
    """
    Plot the AUC curves.
    """
    sns.lineplot(
        x='fpr',
        y='tpr',
        data=data_frame,
        label='ROC curve',
        ax=ax
    )
    ax.plot([0, 0], [1, 1], 'k--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')

def plot_decision_boundary_2d(
        data_frame: pd.DataFrame,
        index_threshold: float,
        selected_palette: dict,
        ax: plt.Axes):
    """
    Plot the decision boundary for the 2D thresholds.
    """
    # scatter = sns.scatterplot(
    #     x='relative_dv_drop',
    #     y='relative_rebound_time',
    #     hue='slowdown_index',
    #     style='spike_type',
    #     size='spike_type',
    #     markers={
    #         'RAAP': '*',
    #         'AISAP': 's'
    #     },
    #     sizes={
    #         'RAAP': 80,
    #         'AISAP': 20
    #     },
    #     palette='viridis_r',
    #     data=data_frame,
    #     ax=ax
    # )
    data_frame['slowdown_index'] = (data_frame['slowdown_index'] +
                                    sys.float_info.epsilon)
    if 'spike_type' in data_frame.columns:
        # Sort the data frame to control plot order
        plot_order = {'AISAP': 1, 'RAAP': 0}
        data_frame['plot_order'] = data_frame['spike_type'].map(plot_order)
        data_frame = data_frame.sort_values('plot_order')

        sns.scatterplot(
            x='relative_dv_drop',
            y='relative_rebound_time',
            hue='spike_type',
            size='slowdown_index',
            marker='H',
            palette=selected_palette,
            data=data_frame,
            ax=ax
        )

        # Remove the hue from the legend
        handles, labels = ax.get_legend_handles_labels()
        # Keep only the style (marker) entries in the legend
        legend_components = [(h, l) for h, l in zip(handles, labels) if l in ['RAAP', 'AISAP']]
        style_handles = [h for h, l in legend_components]
        style_labels = [l for h, l in legend_components]
        ax.legend(style_handles, style_labels,
                title='Spike Type', loc='lower right')        
    else:
        sns.scatterplot(
            x='relative_dv_drop',
            y='relative_rebound_time',
            hue='spike_type',
            palette=selected_palette,
            size='slowdown_index',
            marker='H',
            data=data_frame,
            ax=ax
        )

    if len(index_threshold) == 1:
        x1_values = np.sort(data_frame['relative_dv_drop'].unique())
        x1_values = x1_values[x1_values >= index_threshold]
        x2_values = np.array([
            index_threshold / x1_value if x1_value != 0 else np.nan
            for x1_value in x1_values
        ])
        ax.plot(x1_values, x2_values, 'k--')
    else:
        x1_max = data_frame['relative_dv_drop'].max()
        x2_max = data_frame['relative_rebound_time'].max()
        ax.hlines(
            index_threshold[1], index_threshold[0], x1_max,
            color='black',
            linestyles='dashed'
        )
        ax.vlines(
            index_threshold[0], index_threshold[1], x2_max,
            color='black',
            linestyles='dashed'
        )

    ax.set_xlabel('Relative dV drop')
    ax.set_ylabel('Relative rebound time')
    ax.set_title('Index threshold')

def plot_slowdown_index_distribution(data_frame: pd.DataFrame,
                                     selected_palette: dict,
                                     ax: plt.Axes):
    """
    Plot the slowdown index distribution.
    """
    if 'spike_type' in data_frame.columns:
        sns.histplot(
            x='slowdown_index',
            data=data_frame,
            hue='spike_type',
            palette=selected_palette,
            bins=np.arange(0, data_frame['slowdown_index'].max(), 0.005),
            alpha=0.5,
            legend=True,
            multiple='layer',
            ax=ax
        )
    else:
        sns.histplot(
            x='slowdown_index',
            hue='spike_type',
            palette=selected_palette,
            data=data_frame,
            bins=np.arange(0, data_frame['slowdown_index'].max(), 0.005),
            alpha=0.5,
            ax=ax
        )
    ax.set_title('Slowdown index distribution')
    ax.set_xlabel('Slowdown index')
    ax.set_ylabel('Count')

def plot_slowdown_features(
        data_frame: pd.DataFrame,
        output_dir: Path,
        palette_name: str,
        index_threshold: list[float] = None
):
    """
    Plot the slowdown features.
    """
    data_frame['slowdown_index'] = (data_frame['slowdown_index'] +
                                    sys.float_info.epsilon)

    cell_type_palettes = {
        'invitro_juxta': {
            'RAAP': '#ff5000',
            'AISAP': '#0000ff'
        },
        'viridis_endpoints': {
            'RAAP': '#440154',
            'AISAP': '#FDE725'
        },
        'invivo_juxta': {
            'RSI': '#800080',
            'PYR': '#aa4400',
            'FSI': '#d40000'
        }
    }

    selected_palette = cell_type_palettes[palette_name]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # sns.kdeplot(
    #     x='relative_dv_drop',
    #     y='relative_rebound_time',
    #     data=data_frame,
    #     hue='spike_type',
    #     palette=cell_type_palettes[selected_palette],
    #     alpha=0.5,
    #     fill=True,
    #     multiple='layer',
    #     levels=[0.01,0.05, 0.1, 0.25, 0.5, 0.75, 0.9],
    #     ax=axs[0, 0]
    # )
    # axs[0, 0].set_title('Rising phase feture distribution')
    # axs[0, 0].set_xlabel('Relative dV drop')
    # axs[0, 0].set_ylabel('Relative rebound time')
    # axs[0, 0].set_xlim(0, data_frame['relative_dv_drop'].max())
    # axs[0, 0].set_ylim(0, data_frame['relative_rebound_time'].max())

    # Plot the slowdown index distribution histogram
    plot_slowdown_index_distribution(
        data_frame,
        selected_palette,
        axs[0, 0]
    )

    slowdown_index_xlm = axs[0, 0].get_xlim()

    if index_threshold is None:
        # slowdown_index_metric_data = calculate_error_rate_for_slowdown_thresholds(data_frame)
        # [index_threshold] = calculate_crossing_threshold(slowdown_index_metric_data)
        # # Plot the error rates relative to the slowdown index threshold
        # plot_error_rate_for_thresholds(slowdown_index_metric_data, axs[1, 0])
        # axs[1, 0].set_xlim(slowdown_index_xlm[0], slowdown_index_xlm[1])

        slowdown_index_metric_data = calculate_error_rate_for_two_feature_thresholds(data_frame)
        thresholds = calculate_crossing_threshold_for_two_feature_thresholds(slowdown_index_metric_data)
        plot_error_rate_for_two_features_thresholds(thresholds, axs[1, 0])

        best_threshold_index = thresholds['mean_errors'].idxmax()
        index_threshold = thresholds.loc[best_threshold_index, ['x1_threshold', 'x2_threshold']].values
        axs[1, 0].set_xlim(slowdown_index_xlm[0], slowdown_index_xlm[1])
        axs[1, 0].set_title(
            'Best threshold: ' +
            f'{index_threshold[0]:.4f}, {index_threshold[1]:.4f}' +
            f'precision-recall mean: {thresholds.loc[best_threshold_index, "mean_errors"]:.4f}'
        )


    # sns.histplot(
    #     x='relative_dv_drop',
    #     y='relative_rebound_time',
    #     data=data_frame,
    #     hue='spike_type',
    #     bins=100,
    #     stat='probability',
    #     alpha=0.5,
    #     multiple='layer',
    #     ax=axs[1, 0]
    # )

    # Plot the decision boundary for the 2D thresholds
    plot_decision_boundary_2d(
        data_frame,
        index_threshold,
        selected_palette,
        axs[0, 1]
    )
    
    scatter_xlm = axs[1, 0].get_xlim()
    scatter_ylm = axs[1, 0].get_ylim()

    # Plot the decision boundary for the 2D thresholds
    plot_decision_boundary_2d(
        data_frame,
        index_threshold,
        selected_palette,
        axs[1, 1]
    )
    axs[1, 1].set_xlim(scatter_xlm[0], 0.3)
    axs[1, 1].set_ylim(scatter_ylm[0], 0.6)

    plot_slowdown_index_distribution(
        data_frame[data_frame['slowdown_index'] > 0.005],
        selected_palette,
        axs[2, 0]
    )
    axs[2, 0].set_xlim(slowdown_index_xlm[0], slowdown_index_xlm[1])
    axs[2, 0].set_title('Slowdown index distribution (slowdown index > 0.005)')

    axs[2, 1].remove()

    if 'spike_type' in data_frame.columns:
        # Get the counts for each spike type
        spike_type_counts = [
            f'{spike_type}: #{len(data_frame[data_frame['spike_type'] == spike_type])}'
            for spike_type in data_frame['spike_type'].unique()
        ]

        fig.suptitle(f'{", ".join(spike_type_counts)}')
    fig.tight_layout()
    fig.savefig(output_dir / 'slowdown_comparison.svg')
    plt.show()
