"""
A module for analyzing and visualizing a cell's spike slowdown features.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from scipy.signal import savgol_filter

from common.data_utils import CellInfo, CellWrapper, GroupInfo, GroupWrapper
from analyses.slowdown_descriptors import (
    calculate_rising_features,
    calculate_sparse_2d_histogram,
    convert_features_to_dataframe
)
from analyses.slowdown_descriptors import FEATURE_PROPERTIES as FEATURES
from data.spikes import SpikeShape, SpikeShapes, load_spike_shapes
from processors.spike_transformers import create_transformer
from visualization.spikeshapes import plot_grouped_spikes

def analyze_cell(cell_path: CellWrapper[Path], config: dict) -> dict:
    """Describe spikes for a cell spread across multiple files."""

    cell_results = {
        'name': cell_path.name,
        'type': cell_path.cell_type,
        'images': []
    }

    analysis_type =  config['input']['dataset']
    # Load spike shapes
    if analysis_type == 'invivo':
        cell_spikes = load_spikes(cell_path, config['analysis']['extractor'])
        cell_features = describe_spikes(cell_spikes, config['analysis']['descriptor'])
        cell_features = categorize_spikes(cell_features, config['analysis']['classification'])
        cell_type_color = config['plot']['palette'][cell_features.cell_type]
        cell_features.data['cell_type'] = cell_features.cell_type
        color_mode = 'cell_type'
    elif analysis_type == 'invitro':
        file_name = cell_path.data.stem
        if file_name.endswith('AISAP'):
            return {}
        cell_info = cell_path.info
        cell_name = '_'.join(file_name.split('_')[:-1])
        cell_feature_frames = []
        cell_spike_shapes = []
        for spike_type, type_name in zip(
            ['AISAP', 'RAAP'],
            ['AIS-AP', 'RA-AP']
        ):
            file_path = CellWrapper(
                cell_info,
                cell_path.data.parent / f'{cell_name}_juxta{spike_type}.mat'
            )
            cell_spikes = load_spikes(file_path, config['analysis']['extractor'])
            cell_features = describe_spikes(cell_spikes, config['analysis']['descriptor'])
            cell_feature_frames.append(
                cell_features.data.assign(spike_type=type_name)
            )
            cell_spike_shapes.append(cell_spikes.data)
        combined_cell_info = CellInfo(cell_name, cell_info.cell_type)
        cell_features = CellWrapper(
            combined_cell_info,
            pd.concat(cell_feature_frames, ignore_index=True)
        )
        cell_spikes = CellWrapper(
            combined_cell_info,
            SpikeShapes(
                cell_spike_shapes[0][0].times,
                np.vstack([shapes.segments.values for shapes in cell_spike_shapes]),
                np.concatenate([shapes.timestamps for shapes in cell_spike_shapes], axis=0),
                cell_spike_shapes[0].dt
            )
        )
        cell_type_color = '#0000ff'
        color_mode = 'spike_type'
    else:
        raise ValueError(f'Unknown analysis type: {analysis_type}')

    # Save features
    output_path = (
        Path(config['output']['path']) /
        'Results' /
        cell_features.cell_type
    )
    output_path.mkdir(parents=True, exist_ok=True)
    cell_features.data.to_csv(output_path / f'{cell_features.name}.csv', index=False)

    # Do plots
    suffixes = config['output']['suffixes']


    if config['plot']['selector']['residual_histogram']:
        # Create histogram of noise powers
        cell_figure = create_residual_histogram(
            cell_features,
            config['plot']['residual_histogram']
        )
        figure_path = save_figure(cell_figure, suffixes['residual_histogram'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'residual_histogram'
        })

    if config['plot']['selector']['scatter']:
        # Create scatter histogram
        cell_figure = create_scatter_histogram(
            cell_features,
            config['plot'],
            tuple(config['analysis']['classification']['groups'][-1]['bottom-left']),
            color_mode=color_mode
        )
        figure_path = save_figure(cell_figure, suffixes['scatter'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'scatter'
        })

    if config['plot']['selector']['category_piechart']:
        # Create pie chart of spike types
        piechart_config = config['plot']['category_piechart']
        piechart_config['edge_color'] = {
            'AIS-AP': cell_type_color,
            'RA-AP': cell_type_color,
            'UNKNOWN': '#d0d0d0'
        }
        piechart_config['fill_color'] = {
            'AIS-AP': cell_type_color,
            'RA-AP': '#ffffff',
            'UNKNOWN': '#d0d0d0'
        } 
        piechart_config['hatch'] = {
            'AIS-AP': None,
            'RA-AP': '//',
            'UNKNOWN': None
        }
        cell_figure = create_spike_type_piechart(cell_features, piechart_config)
        figure_path = save_figure(cell_figure, suffixes['category_piechart'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'category_piechart'
        })

    if config['plot']['selector']['category_averages']:
        # Plot grouped spikes
        average_plot_config = config['plot']['category_averages']
        average_plot_config['average']['color'] = cell_type_color
        average_plot_config['group_order'] = [
            g['name'] for g in config['analysis']['classification']['groups']
        ]

        average_plot_config['transformer'] = create_transformer(
            average_plot_config['mode'],
            config['analysis']['descriptor']
        )

        cell_figure = create_grouped_average_plot(
            cell_spikes,
            cell_features.data,
            average_plot_config
        )
        figure_path = save_figure(cell_figure, suffixes['category_averages'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'category_averages'
        })

    if config['plot']['selector']['valley_time_deviation']:
        # Create histogram of valley times
        cell_figure = create_valley_time_deviation_histogram(
            cell_features,
            config['plot']['valley_time_deviation']
        )
        figure_path = save_figure(cell_figure, suffixes['valley_time_deviation'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'valley_time_deviation'
        })


    return cell_results

def save_figure(
    cell_figure: CellWrapper[Figure],
    suffix: str,
    config: dict
) -> str:
    """
    Save the figure to the specified output path.
    Args:
        cell_figure (CellWrapper[Figure]): The figure to save.
        output_path (Path): The path where the figure will be saved.
    """
    output_path = Path(config['path'])
    figure_dir = output_path / 'Figures' / cell_figure.cell_type
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = cell_figure.data
    file_name = f'{cell_figure.name}_{suffix}'
    figure_path = figure_dir / f'{file_name}'
    fig.savefig(f"{figure_path}.png", dpi=config['dpi'], format='png')
    if config['make_svg']:
        fig.savefig(f"{figure_path}.svg", dpi=config['dpi'], format='svg')
    plt.close(fig)
    del fig  # Clean up to avoid memory issues

    return str(figure_dir / f'{file_name}.png').replace('\\', '/')

def load_spikes(
    cell_path: CellWrapper[Path],
    config: dict
) -> CellWrapper[SpikeShapes]:
    """Load spikes from the given cell path using the specified extractor.
    Args:
        cell_path (CellWrapper[Path]): The path to the cell data.
        config (dict): Configuration dictionary containing extractor settings.
    Returns:
        CellWrapper[SpikeShapes]: A wrapper containing the loaded spike shapes.
    """
    spikes = load_spike_shapes(cell_path.data, config)
    return CellWrapper[SpikeShapes](cell_path.info, spikes)

def describe_spikes(
    cell_spikes: CellWrapper[SpikeShapes],
    config: dict
) -> CellWrapper[pd.DataFrame]:
    """
    Describe the spikes using the given descriptor configuration.
    """

    spikes = cell_spikes.data
    def describe(i, spike):
        """Describe a single spike."""
        print(f'Processing spike {i+1}/{len(spikes)}...', end='\r')
        return calculate_rising_features(spike, config)

    features = [describe(i, spike) for i, spike in enumerate(spikes)]
    start_detection_error = sum(
        1 for f in features if (isinstance(f, int) and f == 1)
    )
    too_noisy_error = sum(
        1 for f in features if (isinstance(f, int) and f == 2)
    )
    half_width_error = sum(
        1 for f in features if (isinstance(f, int) and f == 3)
    )
    features = [f for f in features if isinstance(f, dict)]
    print()

    if len(features) < len(spikes):
        n_errors = len(spikes) - len(features)
        print(f'[INFO]: {start_detection_error} spike\'s start couldn\'t be detected.')
        print(f'[INFO]: {too_noisy_error} spike\'s noise ratio is too high.')
        print(f'[INFO]: {half_width_error} spike\'s half-width couldn\'t be determined.')
        print(f'[INFO]: {n_errors} spikes were skipped in total.')

    noisiness_ratio = too_noisy_error / len(spikes)
    if config.get('noisiness_threshold') is not None:
        if noisiness_ratio > config['noisiness_threshold']:
            raise ValueError(
                f"[INFO]: Cell {cell_spikes.name} is too noisy, "
                f"should be skipped. ({too_noisy_error}/{len(spikes)} spikes dropped)"
            )
    feature_frame = convert_features_to_dataframe(features)

    return CellWrapper[pd.DataFrame](cell_spikes.info, feature_frame)

def categorize_spikes(
    cell_features: CellWrapper[pd.DataFrame],
    config: dict
) -> CellWrapper[pd.DataFrame]:
    """ Categorize spikes based on the provided configuration.
    Args:
        cell_features (CellWrapper[pd.DataFrame]): The features of the cell spikes.
        config (dict): Configuration dictionary containing categorization settings.
    Returns:
        CellWrapper[pd.DataFrame]: A DataFrame of spike features with category labels
    """
    features = cell_features.data.copy()
    groups = config['groups']
    features['spike_type'] = groups[0]['name']  # Default category
    for group in groups[1:]:
        bottom_left = group['bottom-left']
        flags = []
        for i, feature in enumerate(FEATURES):
            flags.append(bottom_left[i] < features[feature['name']])
        flag = np.logical_and.reduce(flags)
        features.loc[flag, 'spike_type'] = group['name']
    features['spike_type'] = features['spike_type'].fillna('other')
    return CellWrapper[pd.DataFrame](cell_features.info, features)

def create_scatter_histogram(
    cell_features: CellWrapper[pd.DataFrame],
    plot_config: dict,
    threshold: tuple[float, float],
    color_mode: str = 'cell_type'
) -> CellWrapper[Figure]:
    """
    Create a scatter histogram of the spike features.
    """

    if color_mode == 'cell_type':
        labels = cell_features.data['cell_type'].values
        palette = plot_config['palette']
    elif color_mode == 'spike_type':
        labels = cell_features.data['spike_type'].values
        palette = ['#0000ff', '#ff0000']
    else:
        raise ValueError(f'Unknown color mode: {color_mode}')

    binned_features = calculate_sparse_2d_histogram(
        cell_features.data[[f['name'] for f in FEATURES]].values,
        labels,
        plot_config['scatter']['bin_size']
    )

    fig_size = tuple(plot_config['scatter']['axis_size'])
    fig, ax = plt.subplots(figsize=fig_size)
    sns.scatterplot(
        data=binned_features,
        x='x1',
        y='x2',
        size='ratio',
        hue='label',
        palette=palette,
        #color=plot_config['palette'][cell_features.cell_type],
        sizes=tuple(plot_config['scatter']['point_sizes']),
        ax=ax
    )
    ax.set_xlabel(FEATURES[0]['label'])
    ax.set_ylabel(FEATURES[1]['label'])
    ax.set_xlim(plot_config['scatter']['axis_limits']['x'])
    ax.set_ylim(plot_config['scatter']['axis_limits']['y'])

    ax.hlines(
        threshold[1],
        threshold[0],
        ax.get_xlim()[1],
        color='black',
        linestyle='--'
    )
    ax.vlines(
        threshold[0],
        threshold[1],
        ax.get_ylim()[1],
        color='black',
        linestyle='--'
    )

    ax.get_legend().remove()
    ax.set_title(f'{cell_features.name}')

    return CellWrapper[Figure](cell_features.info, fig)

def create_grouped_average_plot(
    spike_shapes: CellWrapper[SpikeShapes],
    grouping_vector: pd.DataFrame,
    plot_config: dict
) -> CellWrapper[Figure]:
    """
    Create a grouped average plot of the spike shapes.
    """
    groups = [
        form_group(spike_shapes, grouping_vector, group_name)
        for group_name in plot_config['group_order']
    ]

    fig = plot_grouped_spikes(groups, plot_config)

    return CellWrapper[Figure](spike_shapes.info, fig)

def create_spike_type_piechart(
    features: CellWrapper[pd.DataFrame],
    config: dict
) -> CellWrapper[Figure]:
    """
    Create a pie chart of the spike types.
    """
    group_counts = features.data['spike_type'].value_counts()
    labels = group_counts.index.tolist()
    sizes = group_counts.values.tolist()
    fillcolors = [config['fill_color'][label] for label in labels]
    edgecolors = [config['edge_color'][label] for label in labels]
    hatches = [config['hatch'][label] for label in labels]

    fig, ax = plt.subplots(figsize=tuple(config['axis_size']))
    # Matplotlib's pie chart does not support setting hatch colors directly.
    # Hatches will use the edgecolor of each wedge.
    wedges, _, _ = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.2f%%',
        startangle=90,
        colors=fillcolors,
        wedgeprops=dict(linewidth=1.5),
    )

    # Apply hatches and edgecolors manually
    for i, wedge in enumerate(wedges):
        if hatches[i]:
            wedge.set_hatch(hatches[i])
        wedge.set_edgecolor(edgecolors[i])

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return CellWrapper[Figure](features.info, fig)

def create_valley_time_deviation_histogram(
    features: CellWrapper[pd.DataFrame],
    config: dict
) -> CellWrapper[Figure]:
    """
    Create a histogram of the valley time deviations.
    """
    valley_times = features.data['valley_t'] * 1000
    fig, ax = plt.subplots(figsize=tuple(config['axis_size']))
    ax.hist(
        valley_times,
        bins=config['bins'],
        range=config['limits'],
        color=config['color'],
        alpha=0.7
    )
    ax.set_xlabel('Valley time (ms)')
    ax.set_title(f'{features.name} Valley Time Histogram')

    return CellWrapper[Figure](features.info, fig)

def create_residual_histogram(
    features: CellWrapper[pd.DataFrame],
    config: dict
) -> CellWrapper[Figure]:
    """
    Create a histogram of the residuals of the spike shapes.
    """
    noise_power = features.data['noise_power']
    raw_power = features.data['raw_power']
    noise_ratio = (noise_power / raw_power) * 100
    noise_median = np.median(noise_ratio)
    noise_q1 = np.percentile(noise_ratio, 25)
    noise_q3 = np.percentile(noise_ratio, 75)

    fig, ax = plt.subplots(figsize=tuple(config['axis_size']))
    ax.hist(
        noise_ratio,
        bins=config['bins'],
        range=config['limits'],
        color=config['color'],
        alpha=0.7
    )
    ax.set_xlabel('Residual-Raw power ratio (%)')
    ax.set_title(
        f'{features.name}\n' +
        f'Noise ratio median: {noise_median:.2f} %' +
        f'Q1-Q3: {noise_q1:.2f}-{noise_q3:.2f} %'
    )

    return CellWrapper[Figure](features.info, fig)

def form_group(
    spike_shapes: CellWrapper[SpikeShapes],
    assignments: pd.DataFrame,
    name: str
) -> GroupWrapper[SpikeShape]:
    """
    Form a group of spike shapes based on the grouping vector.
    """
    timestamps = spike_shapes.data.timestamps
    member_timestamps = assignments.loc[
        assignments['spike_type'] == name, 'timestamp'
    ]
    members = []
    for ts in member_timestamps:
        actual_index = np.argmin(np.abs(timestamps - ts)).astype(int)
        members.append(spike_shapes.data[actual_index])

    info = GroupInfo(name)
    return GroupWrapper[SpikeShape](info, members)

def calculate_snr_from_residuals(
    spikes: CellWrapper[SpikeShapes],
    config: dict
) -> CellWrapper[pd.DataFrame]:
    """ Estimate the SNR of the spikes using Savitzky-Golay filter.
    Args:
        spikes (CellWrapper[SpikeShapes]): The spike shapes to analyze.
        config (dict): Configuration dictionary containing smoothing parameters.
    Returns:
        CellWrapper[pd.DataFrame]: A DataFrame containing the SNR values for each spike
    """
    snrs = []
    for spike in spikes.data:
        v = spike.values
        smoothed_v = savgol_filter(
            v,
            config['window_length'],
            config['polyorder']
        )
        residuals = v - smoothed_v

        signal_power = np.mean(smoothed_v ** 2)
        noise_power = np.mean(residuals ** 2)

        if noise_power > 0:
            snrs.append(signal_power / noise_power)

    return CellWrapper[pd.DataFrame](
        spikes.info,
        pd.DataFrame({
            'snr': 10 * np.log10(np.array(snrs))
        })
    )

def analyze_noise_characteristics(
    cell_path: CellWrapper[Path],
    config: dict
) -> CellWrapper[pd.DataFrame]:
    """Analyze the noise characteristics of the spikes in a cell.
    Args:
        cell_path (CellWrapper[Path]): The path to the cell data.
        config (dict): Configuration dictionary containing smoothing parameters.
    Returns:
        CellWrapper[pd.DataFrame]: A DataFrame containing the noise characteristics.
    """
    cell_spikes = load_spikes(cell_path, config['analysis']['extractor'])
    cell_noise_properties = calculate_noise_properties(cell_spikes, config['analysis']['smooth'])

    # Save features
    output_path = (
        Path(config['output']['path']) /
        'Preprocessed' /
        cell_path.cell_type
    )
    output_path.mkdir(parents=True, exist_ok=True)
    cell_noise_properties.data.to_csv(
        output_path / f'{cell_path.name}_noise_powers.csv',
        index=False
    )

    noise_property_figure = create_noise_property_histograms(
        cell_noise_properties,
        config['plot']
    )
    save_figure(
        noise_property_figure,
        'noise_properties',
        config['output']
    )

    cell_results = {
        'name': cell_path.name,
        'type': cell_path.cell_type,
        'images': []
    }

    return cell_noise_properties

def calculate_noise_properties(
    spikes: CellWrapper[SpikeShapes],
    config: dict
) -> CellWrapper[pd.DataFrame]:
    """ Estimates the noise reducing capability of a Savitzky-Golay filter on 
    the spikes. It returns the power of the raw, smoothed signal and the noise 
    after smoothing.
    Args:
        spikes (CellWrapper[SpikeShapes]): The spike shapes to analyze.
        config (dict): Configuration dictionary containing smoothing parameters.
    Returns:
        CellWrapper[pd.DataFrame]: A DataFrame containing the powers
    """

    results = []
    for spike in spikes.data:
        v = spike.values
        smoothed_v = savgol_filter(
            v, config['window_length'], config['polyorder']
        )

        results.append({
            'timestamp': spike.timestamp,
            'raw': np.mean(v ** 2),
            'smoothed': np.mean(smoothed_v ** 2),
            'noise': np.mean((v - smoothed_v) ** 2)
        })
    return CellWrapper(spikes.info, pd.DataFrame(results))

def create_noise_property_histograms(
    noise_properties: CellWrapper[pd.DataFrame],
    config: dict
) -> CellWrapper[Figure]:
    """
    Create histograms of the noise properties.
    """
    figsize = (config['axis_size'][0] * 3, config['axis_size'][1])
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, prop in enumerate(['raw', 'smoothed', 'noise']):
        sns.histplot(
            data=noise_properties.data,
            x=prop,
            bins=config['bins'],
            binrange=config.get('limits', None),
            color=config['color'],
            ax=ax[i]
        )
        ax[i].set_title(f'{noise_properties.name} {prop.capitalize()} power')
        ax[i].set_xlabel('Power')
        ax[i].set_ylabel('Count')

    return CellWrapper[Figure](noise_properties.info, fig)