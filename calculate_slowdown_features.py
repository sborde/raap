"""
This script calculates a parametrized slowdown feature calculation.

This is a blank-sheet rewrite of the run_analysis_on_files.py script.
"""

import argparse
from pathlib import Path

from common.data_utils import (
    get_input_file_paths
)
from common.run_utils import load_config
from analyses.slowdown_descriptors import describe_spikes_per_cell

def prepare_output_dir(config: dict) -> dict:
    """
    Create the output directory if it does not exist.
    """
    if not config['output_dir']:
        raise ValueError('Output directory is not specified in the config.')

    output_path = Path(config['output_dir'])
    if 'smooth' in config['descriptor'] and config['descriptor']['smooth']:
        output_path = (
            output_path /
            f"w{config['descriptor']['smooth']['window_length']}_"
            f"p{config['descriptor']['smooth']['polyorder']}"
        )
    else:
        output_path = output_path / 'raw'

    config['output_dir'] = str(output_path)

    for subdir in ['Figures', 'Preprocessed']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    if config['descriptor']['single_spike_images']['do']:
        (output_path / 'Figures' / 'Individual').mkdir(parents=True, exist_ok=True)

    return config

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

    analysis_config = prepare_output_dir(analysis_config)
    entries = get_input_file_paths(input_config)
    cell_entries = dict()
    for entry in entries:
        if analysis_config['measurement_type'] == 'invivo_juxta':
            CELL_NAME = entry.path.stem
            cell_entries[CELL_NAME] = cell_entries.get(CELL_NAME, []) + [entry]
        elif analysis_config['measurement_type'] == 'invitro_juxta':
            parts = entry.path.stem.split('_')
            CELL_NAME = '_'.join(parts[:2]).upper()
            entry.spike_type = parts[2].replace('juxta', '').replace('whole', '')
            cell_entries[CELL_NAME] = cell_entries.get(CELL_NAME, []) + [entry]

    for i, (cell_name, entries) in enumerate(cell_entries.items()):
        print(f'Processing cell {i + 1}/{len(cell_entries)}: {cell_name}')
        describe_spikes_per_cell((cell_name, entries), analysis_config)
    print('Done.')
