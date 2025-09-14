
from collections import namedtuple

import json
from pathlib import Path

import pandas as pd
import yaml

from signalizer.model.experiment.recordings import MatRecording

from common.data_utils import CellWrapper, CellInfo

RecordingSpikePair = namedtuple('RecordingSpikePair', ['recording', 'spike_data'])

def load_customized_config(name: str) -> dict:
    """
    Load a customized configuration file based on the provided name.
    Args:
        name (str): Name of the configuration file (without extension).
    Returns:
        dict: Loaded configuration dictionary.
    """
    default_path = f'config_defaults/{name}.yaml'
    custom_path = f'config/{name}.yaml'

    config = {}
    if Path(default_path).exists():
        with open(Path(default_path), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

    if not Path(custom_path).exists():
        return config

    with open(Path(custom_path), 'r', encoding='utf-8') as file:
        custom_config = yaml.safe_load(file)

    stack = [(config, custom_config)]
    while stack:
        curr_config, curr_custom = stack.pop()
        for key, value in curr_custom.items():
            if key in curr_config and isinstance(value, dict):
                stack.append((curr_config[key], value))
            else:
                curr_config[key] = value
    return config

def load_input_data(
    file_info: CellWrapper[dict[str, Path]]
) -> RecordingSpikePair:
    """
    Load raw data from the specified cell path.
    """
    return RecordingSpikePair(
        recording=MatRecording(file_info.data['raw']),
        spike_data=pd.read_csv(file_info.data['processed'], index_col=None)
    )
