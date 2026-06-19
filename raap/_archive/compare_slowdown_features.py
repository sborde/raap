"""
Compare slowdown features of two different datasets.
"""

import argparse
import yaml
from pathlib import Path
from _archive.analyses.slowdown_comparison import (
    load_spike_feature_dataset,
    plot_slowdown_features,
    InVitroFileNameProcessor,
    InvivoFileNameProcessor
)

if __name__ == '__main__':
    """Compare the slowdown features of two different datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)

    args = parser.parse_args()

    with open(Path(args.config_path), 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])

    skip_files = config['skip_files']
    group_list = config['group_list']

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'invitro':
        file_name_processor = InVitroFileNameProcessor(skip_files)
        data_frame = load_spike_feature_dataset(input_dir, file_name_processor)
        plot_slowdown_features(data_frame, output_dir, 'invitro_juxta')
    elif args.mode == 'invivo':
        file_name_processor = InvivoFileNameProcessor(skip_files, group_list)
        data_frame = load_spike_feature_dataset(input_dir, file_name_processor)
        save_dir = output_dir / '_'.join(group_list)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_slowdown_features(data_frame, save_dir, 'invivo_juxta', 0.0034)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')