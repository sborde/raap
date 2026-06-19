import argparse
from pathlib import Path
import yaml

from _archive.analyses.shape_based import calculate_mean_deviation_index


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run concavity index analysis.')
    parser.add_argument(
        'input_path',
        type=str,
        help='Path of the input file.'
    )
    parser.add_argument(
        '--analysis_config',
        type=str,
        default='spike_extractor.yaml',
        help='Path for the config file.'
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    with open(args.analysis_config, 'r', encoding='utf-8') as config_file:
        configuration = yaml.safe_load(config_file)

    if input_path.is_dir():
        for file_name in input_path.glob('*.mat'):
            print(f'Processing {file_name}')
            calculate_mean_deviation_index(file_name, configuration)
    elif input_path.is_file():
        calculate_mean_deviation_index(input_path, configuration)
    else:
        raise ValueError(f'Invalid input path: {input_path}')
