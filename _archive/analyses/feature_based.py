"""
This module contains functions for calculating and plotting features of spike shapes.
"""

from pathlib import Path

import numpy as np

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from analyses.data_utils import PathEntry

from data.spikes import SpikeShape, SpikeShapes
from analyses.preprocessing import prepare_data

def find_start_index(v: np.ndarray) -> int:
    """
    Find the start index of a spike.
    """
    dv = np.diff(v)

    max_dv_i = np.argmax(dv)
    threshold_dv = 0.1 * np.max(dv)

    if not np.any(dv[:max_dv_i] < threshold_dv):
        return None

    return np.where(dv[:max_dv_i] < threshold_dv)[0][-1] + 1

def calculate_accelerating_features(
        spike: SpikeShape,
        rising_threshold: float = 0.1
    ):
    """
    Calculate the slowdown features.
    """

    v = spike.values
    dv = np.diff(v)
    dt = np.mean(np.diff(spike.times))

    start_i = find_start_index(v)
    dv_max_i = np.argmax(dv)

    # Calculate the given ratio of the amplitude
    amplitude = np.max(v) - v[start_i]
    threshold_v = v[start_i] + amplitude * rising_threshold

    # Find the closest point to the threshold
    rising_start_i = np.where(v[start_i:] > threshold_v)[0][0] + start_i
    if v[rising_start_i + 1]-v[rising_start_i] > v[rising_start_i]-v[rising_start_i-1]:
        rising_start_i = rising_start_i - 1

    # Calculate the duration of the rising phase
    rising_phase_duration = (dv_max_i - rising_start_i) * dt

    # Find the peaks of the first derivative
    peaks, _ = find_peaks(dv, prominence=(None, None))
    # Keep peaks only within the rising phase
    peaks = [peak for peak in peaks if rising_start_i <= peak <= dv_max_i]

    if len(peaks) < 2:
        return {
            'slowdown_index': 0,
            'relative_dv_drop': 0,
            'relative_rebound_time': 0,
            'debug': {
                'start_index': start_i,
                'dv_max_index': dv_max_i,
                'rising_start_index': rising_start_i
            }
        }

    features = []

    for i in range(len(peaks) - 1):

        # Get the neighboring peaks
        peak_i = peaks[i]
        peak_j = peaks[i + 1]

        # Find the valley between the peaks
        between_peaks = dv[peak_i:peak_j + 1]
        valley_i = np.argmin(between_peaks) + peak_i
        valley_dv = dv[valley_i]

        # Calculate the relative drop in dv
        relative_dv_drop = (dv[peak_i] - valley_dv) / dv[peak_i]

        # Find the elapsed time needed to reach the peak again
        rebound_i = np.where(dv[valley_i:peak_j+1] < dv[peak_i])[0][-1] + valley_i

        rebound_time = (rebound_i - peak_i) * dt
        rebound_j = rebound_i
        if dv[rebound_i] < dv[peak_i]:
            rebound_j = rebound_i + 1
            alpha = (dv[peak_i] - dv[rebound_i]) / (dv[rebound_j] - dv[rebound_i])
            rebound_time = rebound_time + alpha * dt
        relative_rebound_time = rebound_time / rising_phase_duration

        # Calculate the slowdown index
        slowdown_index = relative_dv_drop * relative_rebound_time

        features.append({
            'slowdown_index': slowdown_index,
            'relative_dv_drop': relative_dv_drop,
            'relative_rebound_time': relative_rebound_time,
            'debug': {
                'start_index': start_i,
                'dv_max_index': dv_max_i,
                'rising_start_index': rising_start_i,
                'peak_1_index': peak_i,
                'peak_2_index': peak_j,
                'valley_index': valley_i,
                'rebound_index': [rebound_i, rebound_j],
                'rebound_time': rebound_time
            }
        })

    return sorted(features, key=lambda x: x['slowdown_index'])[-1]

def plot_spike_shapes_and_features(
        spikes: SpikeShapes,
        features: list[dict],
        output_path: Path,
        plot_mode: str = 'all'
        ):
    """
    Plot the spike shapes and their features.
    """

    if plot_mode == 'slowdown':
        # Show only spikes with slowdown phase (i.e. at least 2 peaks)
        plot_indices = [
            i for i, features in enumerate(features)
            if len(features['dv_peaks']) > 1
        ]
        output_path = output_path / 'WithSlowdownPhase'
    elif plot_mode == 'slowdown_sorted':
        plot_indices = np.argsort([
            feature['slowdown_index'] for feature in features
        ])
        output_path = output_path / 'SortedBySlowdownIndex'
    else:
        plot_indices = list(range(len(features)))
        output_path = output_path / 'All'

    output_path.mkdir(parents=True, exist_ok=True)
    first_start_i = min(features[i]['debug']['start_index'] for i in plot_indices)

    axis_layout = (10, 1)
    plot_per_figure = axis_layout[0] * axis_layout[1]
    n_row, n_col = axis_layout[0], axis_layout[1] * 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(5, 20))

    for i, index in enumerate(plot_indices):
        print(f'Plotting spike {i + 1}/{len(plot_indices)}', end='\r')

        shape = spikes[index]
        t = shape.times
        v = shape.values
        dv = np.diff(v)
        slowdown_features = features[index]
        debug_features = slowdown_features['debug']

        fig_id = i // plot_per_figure
        axis_id = i % plot_per_figure
        row, col = axis_id // axis_layout[1], axis_id % axis_layout[1]

        start_index = debug_features['start_index']
        rising_start_index = debug_features['rising_start_index']
        dv_max_index = debug_features['dv_max_index']

        main_axis = axs[row, 0]
        axs2 = main_axis.twinx()
        phase_axis = axs[row, col + 1]

        main_axis.plot(t, v, 'b-')
        main_axis.plot(t[start_index:], v[start_index:], 'r-')

        main_axis.plot(t[start_index], v[start_index], 'ro')
        main_axis.vlines(t[start_index], np.min(v), np.max(v),
                           colors='red', linestyles='dashed')
        
        main_axis.plot(t[rising_start_index], v[rising_start_index], 'ko')
        main_axis.vlines(t[rising_start_index], np.min(v), np.max(v),
                           colors='black', linestyles='dashed')

        main_axis.set_xlim(t[first_start_i], t[-1])
        main_axis.set_title(
            f't={shape.timestamp:.3f}s index={slowdown_features["slowdown_index"]:.2f}\n'
            f'rel_dv_drop={slowdown_features["relative_dv_drop"]:.2f} '
            f'rel_rebound_time={slowdown_features["relative_rebound_time"]:.2f}'
        )

        time_labels = np.arange(t[first_start_i], t[-1], 0.0005)
        main_axis.set_xticks(time_labels)
        main_axis.set_xticklabels([f'{1000*t:.2f}' for t in time_labels])

        axs2.plot(t[:-1], dv, c='green')
        axs2.plot(t[start_index:-1], dv[start_index:], 'r-')

        axs2.plot(t[start_index], dv[start_index], 'ro')
        axs2.plot(t[rising_start_index], dv[rising_start_index], 'ko')
        axs2.plot(t[dv_max_index], dv[dv_max_index], 'bs')

        phase_axis.plot(v[:-1], dv, c='blue')
        phase_axis.plot(v[start_index:-1], dv[start_index:], 'r-')
        phase_axis.plot(v[dv_max_index], dv[dv_max_index], 'bs')

        if 'peak_1_index' in debug_features:
            peak_1_index = debug_features['peak_1_index']
            peak_2_index = debug_features['peak_2_index']
            valley_index = debug_features['valley_index']
            rebound_time = debug_features['rebound_time']

            axs2.plot(t[peak_1_index], dv[peak_1_index], 'kx')
            axs2.plot(t[peak_2_index], dv[peak_2_index], 'kx')
            axs2.plot(t[valley_index], dv[valley_index], 'gv')
            axs2.hlines(
                dv[peak_1_index],
                t[peak_1_index],
                t[peak_1_index] + rebound_time,
                colors='black', linestyles='dashed'
            )

            phase_axis.plot(v[peak_1_index], dv[peak_1_index], 'kx')
            phase_axis.plot(v[peak_2_index], dv[peak_2_index], 'kx')
            phase_axis.plot(v[valley_index], dv[valley_index], 'gv')

        if axis_id == plot_per_figure - 1 or i == len(plot_indices) - 1:
            plt.tight_layout()
            fig.savefig(
                output_path / f'{fig_id:03d}_spike_shapes_and_features.png'
            )
            plt.close(fig)
            fig, axs = plt.subplots(n_row, n_col, figsize=(5, 20))

    print('\nSpike shapes and features plotting finished.')

def calculate_spike_rising_phase_features(
        path_entry: PathEntry,
        config: dict
):
    """
    Calculate the spike features.
    """

    input_path = path_entry.path
    subdir = path_entry.subdir

    spikes = prepare_data(input_path, config['data'])
    if config['data']['preprocess']['shape_part'] == 'half':
        spikes = spikes.get_half_spikes()

    features = []
    for i, shape in enumerate(spikes):
        print(f'Describing spike {i + 1}/{len(spikes)}', end='\r')
        features.append(calculate_accelerating_features(shape))

    print('\nSpike shapes described.')

    output_dir = Path(config['analysis']['output_dir'])
    if 'sg_filter' in config['data']['preprocess']:
        filter_str = (
            f'w{config["data"]["preprocess"]["sg_filter"]["window_length"]}_'
            f'p{config["data"]["preprocess"]["sg_filter"]["poly_order"]}'
        )
    else:
        filter_str = 'no'
    output_dir = output_dir / f'Smooth_{filter_str}'

    result_dir_path = output_dir / 'Results'
    result_dir_path.mkdir(parents=True, exist_ok=True)
    figure_dir_path = output_dir / 'Figures'
    figure_dir_path.mkdir(parents=True, exist_ok=True)

    if config['analysis']['debug']:
        print('Plotting spike shapes and features...')
        debug_path = figure_dir_path / 'Individual' / subdir / input_path.stem
        debug_path.mkdir(parents=True, exist_ok=True)
        plot_spike_shapes_and_features(
            spikes,
            features,
            debug_path,
            config['analysis']['plot_mode']
        )

    dataframe = pd.DataFrame({
        'timestamps': [spike.timestamp for spike in spikes],
        'slowdown_index': [feature['slowdown_index'] * 100 for feature in features],
        'relative_dv_drop': [feature['relative_dv_drop'] for feature in features],
        'relative_rebound_time': [feature['relative_rebound_time'] for feature in features]
    })
    dataframe.to_csv(
        result_dir_path / f'{subdir}_{input_path.stem}_slowdown_features.csv',
        index=False
    )

    output_path = figure_dir_path / 'Overall'
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.scatterplot(
        x='relative_dv_drop',
        y='relative_rebound_time',
        data=dataframe,
        hue='slowdown_index',
        ax=axs[0, 0]
    )
    axs[0, 0].set_xlabel('Relative drop')
    axs[0, 0].set_ylabel('Relative rebound time')

    sns.histplot(
        x='relative_dv_drop',
        y='relative_rebound_time',
        bins=50,
        data=dataframe,
        ax=axs[1, 0]
    )
    axs[1, 0].set_xlabel('Relative drop')
    axs[1, 0].set_ylabel('Relative rebound time')

    sns.histplot(
        x='slowdown_index',
        bins=100,
        data=dataframe,
        ax=axs[0, 1]
    )
    axs[0, 1].set_xlabel('Slowdown index')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_title('All spikes')

    sns.histplot(
        x='slowdown_index',
        bins=100,
        data=dataframe.loc[dataframe['slowdown_index'] > 0],
        ax=axs[1, 1]
    )
    axs[1, 1].set_xlim(0, dataframe['slowdown_index'].max() + 1)
    axs[1, 1].set_xlabel('Slowdown index')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title('Spikes with slowdown phase')

    fig.suptitle(f'{subdir} {input_path.stem}')
    fig.savefig(output_path / f'{subdir}_{input_path.stem}_slowdown_features.png')
    plt.close(fig)

def plot_grouped_by_rising_phase_features(
    path_entry: PathEntry,
    config: dict
):
    """
    Plot the grouped spike shapes and their features.
    """
    input_path = path_entry.path
    subdir = path_entry.subdir

    spikes = prepare_data(input_path, config['data'])
    if config['data']['preprocess']['shape_part'] == 'half':
        spikes = spikes.get_half_spikes()

    features = []
    for i, shape in enumerate(spikes):
        print(f'Describing spike {i + 1}/{len(spikes)}', end='\r')
        features.append(calculate_accelerating_features(shape))
    print('\nSpike shapes described.')

    grouping_features = config['analysis']['group_features']
    dataframe = pd.DataFrame({
        feature_name: [
            feature[feature_name] for feature in features
        ]
        for feature_name in grouping_features
    })

    grouping_rules = config['analysis']['group_list']
    group_count = len(grouping_rules)

    output_dir = Path(config['analysis']['output_dir']) / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    t = spikes.times
    fig, axs = plt.subplots(group_count, 2, figsize=(10, group_count * 5))
    for i, grouping_rule in enumerate(grouping_rules):
        flags = np.full(len(spikes), True)
        for feature in grouping_features:
            if 'low' in grouping_rule['limits'][feature]:
                flags = flags & (
                    dataframe[feature] > grouping_rule['limits'][feature]['low']
                )
            if 'up' in grouping_rule['limits'][feature]:
                flags = flags & (
                    dataframe[feature] <= grouping_rule['limits'][feature]['up']
                )

        value_matrix = spikes.values[flags, :]
        for j in range(value_matrix.shape[0]):
            axs[i, 0].plot(t, value_matrix[j, :], '-', c='#999999')
            axs[i, 1].plot(value_matrix[j, :-1], np.diff(value_matrix[j, :]), '-', c='#999999')

        axs[i, 0].plot(
            t, np.mean(value_matrix, axis=0), '-',
            c=grouping_rule['color'],
            linewidth=2
        )
        axs[i, 1].plot(
            np.mean(value_matrix, axis=0)[:-1], np.diff(np.mean(value_matrix, axis=0)), '-',
            c=grouping_rule['color'],
            linewidth=2
        )

        axs[i, 0].set_title(
            f'{grouping_rule["name"]} ' +
            f'{np.sum(flags)}/{len(spikes)} ' +
            f'{100 * np.mean(flags):.1f}%'
        )

    plt.tight_layout()
    fig.savefig(
        output_dir / f'{subdir}_{input_path.stem}_grouped_average.png'
    )
    plt.show()
