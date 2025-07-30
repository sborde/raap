"""
This script creates a box and jitter plot showing the cellwise median SNR
for each cell types.
 - 2025-07-22: initial version
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from utils.file_utils import load_config

from common.data_utils import CellInfo, CellWrapper, CellPath


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

def wrap_path(path: Path, suffix: str) -> CellPath:
    """
    Wrap a path in a CellWrapper with an empty DataFrame.
    Args:
        path (Path): Path to the CSV file.
    Returns:
        CellWrapper[pd.DataFrame]: Wrapper containing cell information and an empty DataFrame.
    """
    info = CellInfo(path.stem.replace(suffix, ''), path.parent.name)
    return CellPath(info, path)

def load_noise_properties(path: CellPath) -> CellWrapper[pd.DataFrame]:
    """
    Load noise properties from a CSV file.
    Args:
        path (CellWrapper[Path]): Wrapper containing cell
        information and path to the CSV file.
    Returns:
        CellWrapper[pd.DataFrame]: Wrapper containing cell information and data.
    """
    data = pd.read_csv(path.path)
    data['name'] = path.info.name
    data['cell_type'] = path.info.cell_type
    return CellWrapper(path.info, data)

def enumerate_single_file(
    path: Path,
    extension: str,
    suffix: str
) -> list[CellPath]:
    """
    Enumerate a single file path.
    Args:
        path (Path): Path to the CSV file.
    Returns:
        list[Path]: List containing the single file path.
    """
    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    if not path.suffix == f'.{extension}':
        raise ValueError(f"{path} is not a {extension} file.")
    return [wrap_path(path, suffix)]

def enumerate_single_directory(
    path: Path,
    extension: str,
    skip_files: set[str],
    suffix: str
) -> list[CellPath]:
    """
    Enumerate all files with a specific extension in a single directory.
    Args:
        path (Path): Path to the directory.
        extension (str): File extension to filter by (e.g., 'csv').
    Returns:
        list[CellPath]: List of paths to files with the specified extension
        in the directory.
    """
    if not path.is_dir():
        raise ValueError(f'{path} is not a directory.')
    file_list = []
    for file_path in path.glob(f'*.{extension}'):
        if file_path.name in skip_files:
            continue
        file_list += enumerate_single_file(file_path, extension, suffix)
    return file_list

def enumerate_multiple_directories(
    path: Path,
    skip_dirs: set[str],
    extension: str,
    skip_files: set[str],
    suffix: str
) -> list[CellPath]:
    """
    Enumerate all files with a specific extension in multiple directories.
    Args:
        path (Path): List of directory paths.
        skip_dirs (set[str]): Set of directory names to skip.
        extension (str): File extension to filter by (e.g., 'csv').
        skip_files (set[str]): Set of file names to skip.
    Returns:
        list[CellPath]: List of paths to files with the specified extension
        in the directories.
    """
    if not path.is_dir():
        raise ValueError(f'{path} is not a directory.')

    file_list = []
    for dir_path in path.glob('*'):
        if not dir_path.is_dir() or dir_path.name in skip_dirs:
            continue
        file_list += enumerate_single_directory(
            dir_path,
            extension,
            skip_files,
            suffix
        )
    return file_list

def enumerate_input_files(config: dict) -> list[CellPath]:
    """
    Enumerate input files based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        list[Path]: List of input file paths.
    """

    if 'mode' not in config:
        raise ValueError('Input mode is not specified in the config.')
    if 'path' not in config:
        raise ValueError('Input path is not specified in the config.')

    skip_files = set(config.get('skip_files', []) or [])
    skip_dirs = set(config.get('skip_dirs', []) or [])

    input_mode = config['mode']
    input_path = Path(config['path'])
    suffix = config.get('suffix', '')

    if input_mode == 'file':
        return enumerate_single_file(
            input_path,
            config['extension'],
            suffix
        )
    if input_mode == 'directory':
        return enumerate_single_directory(
            input_path,
            config['extension'],
            skip_files,
            suffix
        )
    if input_mode == 'recursive':
        return enumerate_multiple_directories(
            input_path,
            skip_dirs,
            config['extension'],
            skip_files,
            suffix
        )
    raise ValueError(f"Unknown input type: {input_mode}")

def create_box_and_jitter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: dict,
    ax: Axes
):
    sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        fill=False,
        width=0.3,
        gap=0.2,
        showfliers=False,
        palette=palette,
        ax=ax
    )
    sns.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        jitter=True,
        alpha=0.5,
        ax=ax
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout(pad=0.1)

if __name__ == '__main__':

    args = parse_args()
    global_config = load_config(args.config)
    cell_paths = enumerate_input_files(global_config['input'])
    pooled_result = pd.concat(
        [load_noise_properties(cell).data for cell in cell_paths],
        ignore_index=True
    )

    plot_config = global_config['plot']
    figsize = tuple(plot_config['axis_size'])
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x='cell_type',
        y='noise',
        hue='cell_type',
        fill=False,
        width=0.3,
        gap=0.2,
        showfliers=False,
        palette=plot_config['palette'],
        data=pooled_result,
        ax=ax
    )
    # Scatter plot of all points
    sns.stripplot(
        x='cell_type',
        y='noise',
        hue='cell_type',
        data=pooled_result,
        palette=plot_config['palette'],
        jitter=True,
        alpha=0.5,
        ax=ax
    )
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Noise Power')
    plt.tight_layout(pad=0.1)

    formats = ['png'] + (['svg'] if global_config['output']['do_svg'] else [])
    for fmt in formats:
        plt.savefig(
            Path(global_config['output']['path']) /
            f'groupwise_{global_config["output"]["suffix"]}.{fmt}',
            dpi=global_config['output']['dpi'],
            bbox_inches='tight'
        )
    plt.close(fig)

    # fig, ax = plt.subplots(figsize=(global_config['plot']['axis_size'][0]//2, global_config['plot']['axis_size'][1]))
    # sns.boxplot(
    #     x=np.zeros(len(grouped_results)),
    #     y=grouped_results['median_snr'],
    #     fill=False,
    #     ax=ax
    # )
    # # Scatter plot of all points
    # sns.stripplot(
    #     x=np.zeros(len(grouped_results)),
    #     y=grouped_results['median_snr'],
    #     ax=ax
    # )
    # ax.set_ylabel('Median SNR (dB)')
    # plt.tight_layout(pad=0.1)
    # formats = ['png'] + (['svg'] if config['output']['do_svg'] else [])
    # for fmt in formats:
    #     plt.savefig(
    #         Path(config['output']['path']) /
    #         f'pooled_{config["output"]["suffix"]}.{fmt}',
    #         dpi=config['output']['dpi'],
    #         bbox_inches='tight'
    #     )
    # plt.close(fig)

    # grouped_results.to_csv(
    #     Path(config['output']['path']) / f'cellwise_median_snr.csv',
    #     index=False
    # )

    # for group_name, group_data in grouped_results.items():
    #     group_data = group_data.sort_values(by='RA-AP', ascending=False)
    #     print(f"Processing group: {group_name}")

    #     x_labels = []
    #     figsize = tuple(config['plot']['axis_size'])
    #     fig, ax = plt.subplots(figsize=figsize)

    #     for i, (_, row) in enumerate(group_data.iterrows()):
    #         x_labels.append(row['cell_name'])
    #         last_value = 0
    #         for j, spike_style in enumerate(config['plot']['styles']):
    #             spike_type = spike_style['type']

    #             bar_color = spike_style['barcolor']
    #             if bar_color == 'cell_type':
    #                 bar_color = config['plot']['palette'][group_name]

    #             edge_color = spike_style['edgecolor']
    #             if edge_color == 'cell_type':
    #                 edge_color = config['plot']['palette'][group_name]

    #             hatch = spike_style['hatch']

    #             value = row.get(spike_type, 0)
    #             ax.fill_between(
    #                 [i + 0.75, i + 1.25],
    #                 last_value,
    #                 last_value + value,
    #                 color=bar_color,
    #                 edgecolor=edge_color,
    #                 label=f'{group_name} {spike_type}',
    #                 hatch=hatch
    #             )
    #             last_value += value

    #     handles, labels = ax.get_legend_handles_labels()
    #     handle_dict = {}
    #     for handle, label in zip(handles, labels):
    #         if label not in handle_dict:
    #             handle_dict[label] = handle

    #     # Get unique handles and labels
    #     unique_handles = list(handle_dict.values())
    #     unique_labels = list(handle_dict.keys())

    #     ax.legend(
    #         unique_handles,
    #         unique_labels,
    #         title='Spike Type',
    #         loc='upper left',
    #         bbox_to_anchor=(1.0, 1),
    #         borderaxespad=0.
    #     )
    #     ax.set_xticks(np.arange(1, len(x_labels) + 1))
    #     ax.set_xticklabels(x_labels, rotation=80)
    #     ax.tick_params(axis='x', labelsize=14)
    #     ax.tick_params(axis='y', labelsize=14)
    #     ax.set_title(f'Cellwise spike type ratios for {group_name}')

    #     plt.tight_layout(pad=0.1)
    #     formats = ['png'] + (['svg'] if config['output']['do_svg'] else [])
    #     for fmt in formats:
    #         plt.savefig(
    #             Path(config['output']['path']) /
    #             f'{group_name}_{config["output"]["suffix"]}.{fmt}',
    #             dpi=config['output']['dpi'],
    #             bbox_inches='tight'
    #         )
    #     plt.close(fig)
    #     # plt.show()