

import sys
from pathlib import Path

import pandas as pd

from analyses.slowdown_descriptors import RisingFeatureCalculator
# pylint: disable=wrong-import-position
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from data.spikes import SpikeShapeLoader, SpikeShapes


from common.run_utils import (
    load_config,
    create_default_argument_parser,
    setup_result_directory,
    collect_inputs
)



if __name__ == "__main__":
    parser = create_default_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    result_dirs = setup_result_directory(config['output']['path'])
    inputs = collect_inputs(config['input'])

    load_spikes = SpikeShapeLoader(config['analysis']['extractor'])
    describe_spikes = RisingFeatureCalculator(config['analysis']['descriptor'])

    for cell_path in inputs:
        spikes = load_spikes(cell_path.path)
        features  = describe_spikes(spikes)

