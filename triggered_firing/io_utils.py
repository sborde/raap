
from collections import namedtuple

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
    custom_path = f'config{name}.yaml'

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
            if key not in curr_config:
                curr_config[key] = value
            elif isinstance(value, dict):
                stack.append((curr_config[key], value))
    return config

def make_file_name(info: CellInfo, config: dict) -> str:
    """
    Create a file name based on the cell information and
    the suffix and extension provided in the configuration.
    """
    suffix = f'_{config["suffix"]}' if config.get('suffix') else ''
    extension = f'.{config.get("extension", "")}'
    return f'{info.name}{suffix}{extension}'

def make_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    base_dir = Path(config['path'])
    subdir = info.cell_type
    file_name = make_file_name(info, config)
    return base_dir / subdir / file_name

def need_to_skip(path: Path, config: dict) -> bool:
    """
    Determine if a file should be skipped based on the configuration.
    """
    if config['mode'] == 'all':
        return (
            path.name in config.get('skip_files', []) or
            path.parent.name in config.get('skip_dirs', [])
        )
    if config['mode'] == 'directory':
        if 'select_dirs' not in config:
            raise ValueError(
                "Configuration must include 'select_dirs' for directory mode"
            )
        return (
            path.parent.name not in config['select_dirs'] or
            (path.name in config.get('skip_files', []))
        )
    if config['mode'] == 'file':
        if 'select_files' not in config:
            raise ValueError(
                "Configuration must include 'select_files' for file mode"
            )
        return path.name not in config['select_files']
    raise ValueError("Invalid configuration mode")

def list_files_recursively(path: str, extension: str) -> list[Path]:
    """
    Returns a list of files with the given extension under the 
    given directory or in any of its subdirectories.
    """
    return list(Path(path).glob(f'**/*.{extension}'))

def list_input_files(
    data_config: dict, input_config: dict
) -> list[CellWrapper[dict[str, Path]]]:
    """
    List input files based on the data and input configuration.
    """
    all_files = list_files_recursively(**data_config['raw_data'])
    selected = [
        CellWrapper(
            CellInfo(file_path.stem, file_path.parent.name),
            {'raw': file_path}
        ) for file_path in all_files
        if not need_to_skip(file_path, input_config)
    ]

    for entry in selected:
        processed_path = make_processed_path(entry.info, data_config['processed_data'])
        if not processed_path.exists():
            raise FileNotFoundError(
                f'Processed file for {entry.info.name} does not exist.'
            )
        entry.data['processed'] = processed_path

    return selected

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
