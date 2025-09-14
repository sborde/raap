"""
This script runs a full slowdown analysis on each cell of the dataset.

It iterates over the given input cells and for each cell:
 - calculates the slowdown features,
 - visualizes the distribution of the spike features
 - plots the spikes grouped by the features according to a given threshold
 - estimates and plots the SNR using the provided Savitzky-Golay filter

Versions:
   - 2025. 07. 19. - initial version
   - 2025. 07. 29. - way of starting the analysis changed
"""

from pathlib import Path

import pandas as pd
import numpy as np

from matplotlib.figure import Figure

from workflows.plotters.base import save_figure
from workflows.plotters.spike_shapes import plot_grouped_spikes
from workflows.processors.spike_transformers import create_transformer
from workflows.common.data_io import enumerate_input_files
from workflows.common.models.cell import (
    CellWrapper
)
from workflows.common.models.spike import (
    SpikeShapes,
    SpikeShape,
    load_spike_shapes
)
from workflows.common.models.group import (
    GroupInfo,
    GroupWrapper
)
from workflows.plotters.spike_types import make_spike_type_piechart
from workflows.plotters.spike_properties import make_scatter_histogram, make_slowdown_debug_images, make_valley_time_deviation_histogram, make_residual_histogram
from workflows.common.report_generation import create_report_generator
from workflows.calculators.slowdown_descriptors import FEATURE_PROPERTIES as FEATURES, calculate_rising_features, convert_features_to_dataframe

def run_slowdown_analysis(config: dict) -> None:
    """
    Runs the full slowdown analysis on the dataset 
    based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary for the analysis.
    """

    report_generator = create_report_generator(config)

    cells = enumerate_input_files(config)
    results = []
    for i, cell in enumerate(cells):
        print(f"Processing file {i+1}/{len(cells)}: {cell.name}")
        results.append(analyze_cell(cell, config))
        report_generator(results)

def analyze_cell(cell_path: CellWrapper[Path], config: dict) -> dict:
    """Describe spikes for a cell spread across multiple files."""

    cell_results = {
        'name': cell_path.name,
        'type': cell_path.cell_type,
        'images': []
    }

    # Load spike shapes
    cell_spikes = load_spikes(cell_path, config['analysis']['extractor'])
    # Calculate features
    cell_features = describe_spikes(cell_spikes, config['analysis']['descriptor'])
    # Categorize spikes
    cell_features = categorize_spikes(cell_features, config['analysis']['classification'])

    # Save features
    output_path = (
        Path(config['output']['path']) /
        'Preprocessed' /
        cell_path.cell_type
    )
    output_path.mkdir(parents=True, exist_ok=True)
    cell_features.data.to_csv(output_path / f'{cell_path.name}.csv', index=False)

    # Do plots
    suffixes = config['output']['suffixes']
    cell_type_color = config['plot']['palette'][cell_features.cell_type]

    if config['plot']['selector']['residual_histogram']:
        # Create histogram of noise powers
        cell_figure = make_residual_histogram(
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
        cell_figure = make_scatter_histogram(
            cell_features,
            FEATURES,
            config['plot'],
            tuple(config['analysis']['classification']['groups'][-1]['bottom-left']))
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
        cell_figure = make_spike_type_piechart(cell_features, piechart_config)
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
        cell_figure = make_valley_time_deviation_histogram(
            cell_features,
            config['plot']['valley_time_deviation']
        )
        figure_path = save_figure(cell_figure, suffixes['valley_time_deviation'], config['output'])
        cell_results['images'].append({
            'path': figure_path,
            'type': 'valley_time_deviation'
        })

    return cell_results

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
    features = [f for f in features if f is not None]
    print()

    if len(features) < len(spikes):
        n_errors = len(spikes) - len(features)
        print(f'[INFO]: {n_errors} spike\'s start couldn\'t be detected.')

    output_path = Path(config['output_dir'])
    if config['single_spike_images']['do']:
        debug_dir = output_path / 'Figures' / 'Individual'
        if cell_spikes.cell_type is not None:
            debug_dir = debug_dir / cell_spikes.cell_type
        debug_dir = debug_dir / cell_spikes.name
        debug_dir.mkdir(parents=True, exist_ok=True)
        plot_features = features
        if config['single_spike_images']['skip_zero']:
            plot_features = [
                f for f in features 
                if (
                    f['relative_dv_drop'] > 0 or
                    f['relative_rebound_time'] > 0
                )
            ]
        make_slowdown_debug_images(
            plot_features,
            debug_dir,
            output_dpi=config['single_spike_images']['dpi']
        )
        print('Individual images created.')

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


# def describe_spikes_per_cell(cell_data: tuple[str, list[PathEntry]], config: dict) -> None:
#     """Describe spikes for a cell spread across multiple files."""

#     cell_name = cell_data[0]
#     entries = cell_data[1]
#     output_path = Path(config['output_dir'])

#     all_features = []
#     for entry in entries:
#         print(f'Processing {entry.path.stem}...')
#         features = describe_spikes(entry, config)
#         features['cell_name'] = cell_name

#         features['spike_type'] = entry.spike_type
#         all_features.append(features)

#     all_features_df = pd.concat(all_features, ignore_index=True)

#     if entries[0].subdir is not None:
#         all_features_df['cell_type'] = entries[0].subdir

#     all_features_df.to_csv(
#         output_path / 'Preprocessed' / f'{cell_name}_features.csv',
#         index=False
#     )

#     binned_features = calculate_sparse_2d_histogram(
#         all_features_df[['relative_dv_drop', 'relative_rebound_time']].values,
#         all_features_df['spike_type'].values,
#         bin_size=0.005
#     )

#     palette = config['palette']
#     if config['coloring'] == 'cell_type':
#         if entries[0].subdir is None:
#             palette = None
#         binned_features['label'] = entries[0].subdir

#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.scatterplot(
#         data=binned_features,
#         x='x1',
#         y='x2',
#         size='ratio',
#         hue='label',
#         sizes=(10, 1000),
#         palette=palette,
#         ax=ax
#     )
#     ax.set_xlabel('Relative dV Drop')
#     ax.set_ylabel('Relative Rebound Time')

#     handles, labels = ax.get_legend_handles_labels()
#     components_to_keep = [
#         (handle, label) for handle, label
#         in zip(handles, labels)
#         if label in ['RAAP', 'AISAP']
#     ]
#     style_handles = [h for h, _ in components_to_keep]
#     style_labels = [l for _, l in components_to_keep]
#     ax.legend(
#         style_handles,
#         style_labels,
#         title='Spike Type',
#         loc='lower right'
#     )
#     ax.set_title(f'{cell_name} spike features')

#     fig.savefig(
#         output_path / 'Figures' / f'{cell_name}_features.png', dpi=300
#     )
#     plt.close(fig)
