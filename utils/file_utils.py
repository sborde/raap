"""Utility functions for file system operations."""

from pathlib import Path

import yaml

from common.data_utils import CellWrapper, CellInfo

def load_config(path: str) -> dict:
    """
    Load the configuration from the specified path.
    Args:
        path (str): Path to the configuration file.
    Returns:
        dict: Loaded configuration dictionary.
    """
    with open(
        Path(path).with_suffix('.yaml'),
        'r',
        encoding='utf-8'
    ) as file:
        return yaml.safe_load(file)

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
        config = load_config(default_path)
    if not Path(custom_path).exists():
        return config

    custom_config = load_config(custom_path)
    stack = [(config, custom_config)]
    while stack:
        curr_config, curr_custom = stack.pop()
        for key, value in curr_custom.items():
            if key not in curr_config:
                curr_config[key] = value
            elif isinstance(curr_config[key], dict) and isinstance(value, dict):
                stack.append((curr_config[key], value))
    return config

def enumerate_input_files(config: dict) -> list[CellWrapper[Path]]:
    """
    Enumerate input files based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        list[Path]: List of input file paths.
    """

    if 'input' not in config:
        raise ValueError('Input configuration is missing in the config.')
    if 'mode' not in config['input']:
        raise ValueError('Input mode is not specified in the config.')
    if 'path' not in config['input']:
        raise ValueError('Input path is not specified in the config.')

    skip_files = set(config['input'].get('skip_files', []) or [])
    skip_dirs = set(config['input'].get('skip_dirs', []) or [])

    input_mode = config['input']['mode']
    input_path = Path(config['input']['path'])

    if input_mode == 'file':
        if not input_path.is_file():
            raise ValueError(f"Input path {input_path} is not a file.")
        return [
            CellWrapper[Path](
                CellInfo(
                    input_path.stem,
                    input_path.parent.name
                ),
                input_path
            )
        ]

    if input_mode == 'directory':
        dir_list = [input_path]
    elif input_mode == 'recursive':
        dir_list = [
            sub_dir for sub_dir in input_path.rglob('*')
            if sub_dir.is_dir()
        ]
    else:
        raise ValueError(f"Unknown input type: {input_mode}")

    return [
        CellWrapper[Path](
            CellInfo(
                file_path.stem,
                file_path.parent.name
            ),
            file_path
        ) for dir_path in dir_list
        for file_path in dir_path.glob('*.mat')
        if (
            file_path.is_file()
            and dir_path.name not in skip_dirs
            and file_path.name not in skip_files
        )
    ]
