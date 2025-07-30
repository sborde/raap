"""
This script runs the analysis on the files specified in the input config file.
"""

import argparse
import yaml
from pathlib import Path

from analyses.shape_based import project_to_umap
from analyses.feature_based import calculate_spike_rising_phase_features
from analyses.data_utils import get_input_file_paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        type=str,
        help='Command to run.'
    )

    parser.add_argument(
        'analysis_config',
        type=str,
        help='Path of the analysis config file.'
    )

    parser.add_argument(
        '--input_config',
        type=str,
        default='inputs.yaml',
        help='Path of the input config file.'
    )
    parser.add_argument(
        '--spike_config',
        type=str,
        default='spike_config.yaml',
        help='Path of the spike extractor config file.'
    )
    args = parser.parse_args()

    with open(args.analysis_config, 'r', encoding='utf-8') as f:
        analysis_config = yaml.safe_load(f)

    with open(args.input_config, 'r', encoding='utf-8') as f:
        input_config = yaml.safe_load(f)

    files = get_input_file_paths(input_config)

    with open(args.spike_config, 'r', encoding='utf-8') as f:
        spike_config = yaml.safe_load(f)

    output_dir = Path(analysis_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {'data': spike_config}
    if args.command == 'umap':
        config['umap'] = analysis_config['umap']
        
        for file_path in files:
            print(f'Processing {file_path.path.stem}...')
            project_to_umap(file_path, config, output_dir)
            print('Done.')
    elif args.command == 'features':
        config['analysis'] = analysis_config
        for file_path in files:
            print(f'Processing {file_path.path.stem}...')
            try:
                calculate_spike_rising_phase_features(file_path, config)
            except Exception as e:
                print(f'Error processing {file_path.path.stem}: {e}')
            print('Done.')
