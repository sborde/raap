"""
Runs grouped spike plotting based on thresholds.
"""

import argparse
from pathlib import Path
import yaml

from analyses.data_utils import get_input_file_paths
from _archive.analyses.feature_based import plot_grouped_by_rising_phase_features

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_config', type=str)
    parser.add_argument('config_path', type=str)

    parser.add_argument(
        '--spike_config',
        type=str,
        default='spike_config.yaml',
        help='Path of the spike extractor config file.'
    )    

    args = parser.parse_args()

    with open(Path(args.input_config), 'r', encoding='utf-8') as file:
        input_config = yaml.safe_load(file)

    with open(Path(args.config_path), 'r', encoding='utf-8') as file:
        analysis_config = yaml.safe_load(file)

    with open(Path(args.spike_config), 'r', encoding='utf-8') as file:
        spike_config = yaml.safe_load(file)

    files = get_input_file_paths(input_config)

    config = {
        'data': spike_config,
        'analysis': analysis_config
    }

    for file_path in files:
        print(f'Processing {file_path.path.stem}...')
        plot_grouped_by_rising_phase_features(file_path, config)
        print('Done.')
