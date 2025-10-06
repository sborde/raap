"""Utility functions for file system operations."""

from pathlib import Path

import numpy as np
import pandas as pd
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

def list_input_from_table(
    data_config: dict, input_config: dict, processed: bool = True
) -> list[CellWrapper[dict[str, Path]]]:
    """
    List input files based on a table file.
    Args:
        path (str): Path to the table file.
    Returns:
        list[CellWrapper[Path]]: List of CellWrapper objects containing cell info and file paths.
    """
    table_path = Path(input_config['table_path'])
    if not table_path.exists():
        raise FileNotFoundError(f"Table file {table_path} does not exist.")

    table = pd.read_excel(table_path, index_col=0)
    rules = input_config.get('select_rules', [])
    flag = np.zeros(len(table), dtype=bool)
    for rule in rules:
        rule_flag = np.ones(len(table), dtype=bool)
        for column, value in rule.items():
            rule_flag &= (table[column] == value)
        flag |= rule_flag

    dataset_config = data_config[input_config['dataset']]
    table = table[flag]
    entries = []
    for _, row in table.iterrows():
        row_dict = row.to_dict()
        cell_info = CellInfo(row_dict['name'], row_dict.get('group', ''))
        raw_path = (
            Path(dataset_config['raw_data']['path']) /
            cell_info.cell_type /
            f"{cell_info.name}.mat"
        )
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file {raw_path} does not exist.")
        entry = CellWrapper(cell_info, {'raw': raw_path})

        if processed:
            processed_path = make_processed_path(entry.info, dataset_config['results'])
            if not processed_path.exists():
                raise FileNotFoundError(
                    f'Processed file for {entry.info.name} does not exist.'
                )
            entry.data['processed'] = processed_path

        entries.append(entry)
    return entries

def get_cell_info_list(input_config: dict) -> list[CellInfo]:
    """
    Get a list of CellInfo objects based on a table file.
    Args:
        input_config (dict): Input configuration containing the table path.
    """
    table_path = Path(input_config['table_path'])
    if not table_path.exists():
        raise FileNotFoundError(f"Table file {table_path} does not exist.")

    table = pd.read_excel(table_path, index_col=0)
    rules = input_config.get('select_rules', [])
    flag = np.zeros(len(table), dtype=bool)
    for rule in rules:
        rule_flag = np.ones(len(table), dtype=bool)
        for column, value in rule.items():
            rule_flag &= (table[column] == value)
        flag |= rule_flag

    table = table[flag]
    entries = [
        CellInfo(row['name'], row.get('group', ''))
        for _, row in table.iterrows()
    ]
    return entries

def list_input_files(
    data_config: dict, input_config: dict, processed: bool = True
) -> list[CellWrapper[dict[str, Path]]]:
    """
    List input files based on the data and input configuration.
    """
    dataset_config = data_config[input_config['dataset']]
    all_files = list_files_recursively(**dataset_config['raw_data'])
    selected = [
        CellWrapper(
            CellInfo(file_path.stem, file_path.parent.name),
            {'raw': file_path}
        ) for file_path in all_files
        if not need_to_skip(file_path, input_config)
    ]

    if not processed:
        return selected

    for entry in selected:
        processed_path = make_processed_path(entry.info, dataset_config['results'])
        if not processed_path.exists():
            raise FileNotFoundError(
                f'Processed file for {entry.info.name} does not exist.'
            )
        entry.data['processed'] = processed_path

    return selected

def list_files_recursively(path: str, extension: str) -> list[Path]:
    """
    Returns a list of files with the given extension under the 
    given directory or in any of its subdirectories.
    """
    return list(Path(path).glob(f'**/*.{extension}'))

def need_to_skip(path: Path, config: dict) -> bool:
    """
    Determine if a file should be skipped based on the configuration.
    """
    if config['mode'] == 'all':
        return (
            path.name in (config.get('skip_files') or []) or
            path.parent.name in (config.get('skip_dirs') or [])
        )
    if config['mode'] == 'directory':
        if 'select_dirs' not in config:
            raise ValueError(
                "Configuration must include 'select_dirs' for directory mode"
            )
        return (
            path.parent.name not in config['select_dirs'] or
            (path.name in (config.get('skip_files') or []))
        )
    if config['mode'] == 'file':
        if 'select_files' not in config:
            raise ValueError(
                "Configuration must include 'select_files' for file mode"
            )
        return path.name not in config['select_files']
    raise ValueError("Invalid configuration mode")

def make_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    base_dir = Path(config['path'])
    subdir = info.cell_type
    file_name = make_file_name(info, config)
    return base_dir / subdir / file_name

def make_file_name(info: CellInfo, config: dict) -> str:
    """
    Create a file name based on the cell information and
    the suffix and extension provided in the configuration.
    """
    suffix = f'_{config["suffix"]}' if config.get('suffix') else ''
    extension = f'.{config.get("extension", "")}'
    return f'{info.name}{suffix}{extension}'

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
