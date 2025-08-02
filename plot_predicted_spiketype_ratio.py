"""
This script creates a stacked bar chart showing the predicted
ratios of different spike types per cells.
 - 2025-07-03: initial version
 - 2025-07-22: modified to follow the new result folder structure
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from common.wrappers import CellInfo, CellWrapper

from common.run_utils import load_config

def parse_args():
    """Parse command line arguments for the analysis script.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = ArgumentParser(
        description='Plot predicted spike type ratios from CSV files for each cell.'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to the configuration YAML file.'
    )
    return parser.parse_args()

def enumerate_input_files(config: dict) -> list[CellWrapper[pd.DataFrame]]:
    """
    Enumerate input files based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        list[Path]: List of input file paths.
    """

    if 'input' not in config:
        raise ValueError('Input configuration is missing in the config.')
    if 'mode' not in config['input']:
        raise ValueError('Input mode is not specified in the config.')
    if 'path' not in config['input']:
        raise ValueError('Input path is not specified in the config.')

    skip_files = set(config['input'].get('skip_files', []) or [])
    skip_dirs = set(config['input'].get('skip_dirs', []) or [])

    input_mode = config['input']['mode']
    input_path = Path(config['input']['path'])

    if input_mode == 'file':
        if not input_path.is_file():
            raise ValueError(f"Input path {input_path} is not a file.")
        return [
            CellWrapper[pd.DataFrame](
                CellInfo(
                    input_path.stem,
                    input_path.parent.name
                ),
                pd.read_csv(input_path)
            )
        ]

    if input_mode == 'directory':
        dir_list = [input_path]
    elif input_mode == 'recursive':
        dir_list = [
            sub_dir for sub_dir in input_path.rglob('*')
            if sub_dir.is_dir()
        ]
    else:
        raise ValueError(f"Unknown input type: {input_mode}")

    return [
        CellWrapper[pd.DataFrame](
            CellInfo(
                file_path.stem,
                file_path.parent.name
            ),
            pd.read_csv(file_path)
        ) for dir_path in dir_list
        for file_path in dir_path.glob('*.csv')
        if (
            file_path.is_file()
            and dir_path.name not in skip_dirs
            and file_path.name not in skip_files
            and not file_path.stem.endswith('_snr')
        )
    ]

def create_groups(
    cell_results: list[CellWrapper[pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    """
    Create a DataFrame with cell names, types, and spike counts.
    Args:
        cell_results (list[CellWrapper[pd.DataFrame]]): List of cell results.
    Returns:
        pd.DataFrame: DataFrame with cell information and spike counts.
    """
    groups = {}
    for cell in cell_results:
        cell_type = cell.cell_type
        cell_name = cell.name

        spike_counts = cell.data.groupby('spike_type').size().to_dict()
        total_spikes = sum(spike_counts.values())
        spike_ratios = {
            spike_type: count / total_spikes if total_spikes > 0 else 0
            for spike_type, count in spike_counts.items()
        } | {
            'cell_name': cell_name,
            'cell_type': cell_type,
            'n': total_spikes
        }

        if cell_type not in groups:
            groups[cell_type] = []

        groups[cell_type].append(pd.DataFrame(spike_ratios, index=[0]))

    return {
        cell_type: (
            pd.concat(data_frames, ignore_index=True)
            .fillna(0)
        )
        for cell_type, data_frames in groups.items()
    }

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    cell_results = enumerate_input_files(config)

    grouped_results = create_groups(cell_results)
    for group_name, group_data in grouped_results.items():
        group_data = group_data.sort_values(by='RA-AP', ascending=False)
        print(f"Processing group: {group_name}")

        x_labels = []
        figsize = tuple(config['plot']['axis_size'])
        fig, ax = plt.subplots(figsize=figsize)

        for i, (_, row) in enumerate(group_data.iterrows()):
            x_labels.append(row['cell_name'])
            last_value = 0
            for j, spike_style in enumerate(config['plot']['styles']):
                spike_type = spike_style['type']

                bar_color = spike_style['barcolor']
                if bar_color == 'cell_type':
                    bar_color = config['plot']['palette'][group_name]

                edge_color = spike_style['edgecolor']
                if edge_color == 'cell_type':
                    edge_color = config['plot']['palette'][group_name]

                hatch = spike_style['hatch']

                value = row.get(spike_type, 0)
                ax.fill_between(
                    [i + 0.75, i + 1.25],
                    last_value,
                    last_value + value,
                    color=bar_color,
                    edgecolor=edge_color,
                    label=f'{group_name} {spike_type}',
                    hatch=hatch
                )
                last_value += value

        handles, labels = ax.get_legend_handles_labels()
        handle_dict = {}
        for handle, label in zip(handles, labels):
            if label not in handle_dict:
                handle_dict[label] = handle

        # Get unique handles and labels
        unique_handles = list(handle_dict.values())
        unique_labels = list(handle_dict.keys())

        ax.legend(
            unique_handles,
            unique_labels,
            title='Spike Type',
            loc='upper left',
            bbox_to_anchor=(1.0, 1),
            borderaxespad=0.
        )
        ax.set_xticks(np.arange(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=80)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_title(f'Cellwise spike type ratios for {group_name}')

        plt.tight_layout(pad=0.1)
        formats = ['png'] + (['svg'] if config['output']['do_svg'] else [])
        for fmt in formats:
            plt.savefig(
                Path(config['output']['path']) /
                f'{group_name}_{config["output"]["suffix"]}.{fmt}',
                dpi=config['output']['dpi'],
                bbox_inches='tight'
            )
        plt.close(fig)
        # plt.show()
