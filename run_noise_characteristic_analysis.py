

from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common.wrappers import CellWrapper
from common.run_utils import load_config
from analyses.slowdown_analysis import analyze_noise_characteristics
from data.spikes import SpikeShapes
from common.run_utils import collect_inputs


def parse_args():
    """Parse command line arguments for the analysis script.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = ArgumentParser(
        description='Run full slowdown analysis on dataset.'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to the analysis configuration file.'
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args.config)
    cell_paths = collect_inputs(config)

    cell_results = []
    for i, cell_path in enumerate(cell_paths):
        print(f"Processing file {i+1}/{len(cell_paths)}: {cell_path.name}")
        cell_results.append(analyze_noise_characteristics(cell_path, config))
