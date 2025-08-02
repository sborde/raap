"""
This script performs a parametrized slowdown feature calculation.

This is a blank-sheet rewrite of the run_analysis_on_files.py script.
"""

import argparse
from pathlib import Path
import pandas as pd

from common.data_utils import (
    handle_output_dir
)
from common.run_utils import load_config
from analyses.slowdown_visualizers import visualize_slowdown_features

def collect_dataset(config: dict) -> list[pd.DataFrame]:
    """
    Collect all dataset from the given path and returns as a list.
    """
    path = Path(config['base_path'])
    skip_list = config['skip'] if 'skip' in config and config['skip'] else []

    data = [
        pd.read_csv(path / file_name, index_col=0)
        for file_name in path.glob('*.csv')
        if file_name.name not in skip_list
    ]
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'analysis_config',
        type=str,
        help='Path of the analysis config file.'
    )
    parser.add_argument(
        'input_config',
        type=str,
        help='Path of the input config file.'
    )
    args = parser.parse_args()

    analysis_config = load_config(args.analysis_config)
    input_config = load_config(args.input_config)

    # analysis_config = handle_output_dir(analysis_config)

    datasets = collect_dataset(input_config)

    if analysis_config['run_mode'] == 'pooled':
        pooled_dataset = pd.concat(datasets, ignore_index=True)
        visualize_slowdown_features(pooled_dataset, analysis_config)
    elif analysis_config['run_mode'] == 'celltype':
        pooled_dataset = pd.concat(datasets, ignore_index=True)
        group_names = pooled_dataset['cell_type'].unique()
        for group_name in group_names:
            analysis_config['plot_title'] = group_name
            group_dataset = pooled_dataset[pooled_dataset['cell_type'] == group_name]
            visualize_slowdown_features(group_dataset, analysis_config)
    elif analysis_config['run_mode'] == 'individual':
        for dataset in datasets:
            analysis_config['plot_title'] = dataset.iloc[0]['cell_name']
            visualize_slowdown_features(dataset, analysis_config)
