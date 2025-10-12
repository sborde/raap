"""Module for common input/output operations for the workflows."""

import abc

from typing import Callable, Protocol
from pathlib import Path

import numpy as np
import pandas as pd

from param_classes.workflows.episode_analysis import GroupingInputParams, InputDataParams, CellwiseResultsPathParams
from signalizer.model.experiment.recordings import MatRecording

from analyses.model import RecordingWithSpikes

from common.data_utils import CellWrapper, CellInfo


class FileChecker(Protocol):
    """A protocol for checking file inclusion."""
    @abc.abstractmethod
    def __call__(self, path: Path) -> bool:
        """Checks if the file at the given path should be included."""


CellEnumerator = Callable[[dict], list[CellWrapper[Path]]]


class ResultPaths:
    """A data class that contains the paths for raw and processed data."""
    def __init__(self, raw: Path, processed: Path):
        self.raw = raw
        self.processed = processed

    def load(self) -> RecordingWithSpikes:
        return RecordingWithSpikes(
            MatRecording(self.raw),
            pd.read_csv(self.processed.as_posix(), index_col=None)
        )

def collect_paths_from_table(
    data_config: InputDataParams,
    input_config: dict
) -> list[CellWrapper[ResultPaths]]:
    """Returns the result-raw data path pairs for the cells specified in the table."""
    cell_infos = get_cell_info_list(input_config)
    return collect_path_for_infos(cell_infos, data_config.model_dump())

def collect_results_from_table(
    results_config: CellwiseResultsPathParams,
    input_config: dict
) -> list[CellWrapper[Path]]:
    """Returns the result tables' path for the cells specified in the table."""
    cell_infos = get_cell_info_list(input_config)
    result_params = results_config.model_dump()
    result_params['path'] = str(
        Path(result_params['path']) / '01_results' / 'Individual'
    )
    return collect_processed_paths_for_infos(
        cell_infos, result_params
    )

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
    paths: dict, input_config: dict
) -> list[CellWrapper[ResultPaths]]:
    """
    List input files based on the data and input configuration.
    """
    enumerate_input = create_file_lister(input_config)
    raw_paths = enumerate_input(paths['raw_data'])

    outputs = []
    for entry in raw_paths:
        processed_path = get_processed_path(entry.info, paths['processed_data'])
        if not processed_path.exists():
            raise FileNotFoundError(
                f'Processed file for {entry.info.name} does not exist.'
            )
        outputs.append(
            CellWrapper(
                entry.info,
                ResultPaths(entry.data, processed_path)
            )
        )
    return outputs

def collect_path_for_infos(
    infos: list[CellInfo], paths: dict
) -> list[CellWrapper[ResultPaths]]:
    """
    Collects paths for the given cell infos based on the data configuration.
    Args:
        infos (list[CellInfo]): List of CellInfo objects.
        paths (dict): Data configuration containing paths.
    Returns:
        list[CellWrapper[ResultPaths]]: List of CellWrapper objects with paths.
    """
    outputs = []
    for info in infos:
        raw_path = (
            Path(paths['raw_data']['path']) /
            info.cell_type /
            f"{info.name}.{paths['raw_data']['extension']}"
        )
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file {raw_path} does not exist.")
        results_path = get_processed_path(info, paths['results'])
        if not results_path.exists():
            raise FileNotFoundError(
                f'Processed file for {info.name} does not exist.'
            )
        outputs.append(
            CellWrapper(
                info,
                ResultPaths(raw_path, results_path)
            )
        )
    return outputs

def collect_processed_paths_for_infos(
    infos: list[CellInfo], result_paths: dict
) -> list[CellWrapper[Path]]:
    """
    Collects processed paths for the given cell infos based on the data configuration.
    Args:
        infos (list[CellInfo]): List of CellInfo objects.
        paths (dict): Data configuration containing paths.
    Returns:
        list[CellWrapper[ResultPaths]]: List of CellWrapper objects with paths.
    """
    outputs = []
    for info in infos:
        results_path = get_processed_path(info, result_paths)
        if not results_path.exists():
            raise FileNotFoundError(
                f'Processed file for {info.name} does not exist.'
            )
        outputs.append(CellWrapper(info, results_path))
    return outputs

def create_file_lister(input_config: dict) -> CellEnumerator:
    """
    Returns a function that lists files that conforms to the input
    conditions on the path provided as an input argument.
    """
    checker = create_file_checker(input_config)
    return lambda config: list_selected_files(
        config['path'], config['extension'], checker
    )

def list_selected_files(path: str, ext: str, checker: FileChecker) -> list[CellWrapper[Path]]:
    """Lists paths of the raw data that conforms the selection conditions."""
    return [
        CellWrapper(
            CellInfo(file_path.stem, file_path.parent.name), file_path
        )
        for file_path in Path(path).glob(f'**/*.{ext}')
        if checker(file_path)
    ]

def create_file_checker(config: dict) -> FileChecker:
    """
    Creates a checker function that determines by a Path object if that file
    should be included in the input list.
    """
    if config['mode'] == 'all':
        return lambda path: not (
            is_skipped_file(path, (config.get('skip_files') or [])) or
            is_in_skipped_dir(path, (config.get('skip_dirs') or []))
        )
    if config['mode'] == 'directory':
        if 'select_dirs' not in config:
            raise ValueError(
                "Configuration must include 'select_dirs' for directory mode"
            )
        return lambda path: (
            is_in_selected_dir(path, config['select_dirs']) and
            not is_skipped_file(path, (config.get('skip_files') or []))
        )
    if config['mode'] == 'file':
        if 'select_files' not in config:
            raise ValueError(
                "Configuration must include 'select_files' for file mode"
            )
        return lambda path: is_selected_file(path, config['select_files'])
    raise ValueError("Invalid configuration mode")

def is_skipped_file(path: Path, names: list[str] = []) -> bool:
    """
    Check if a file should be skipped based on its name and the provided
    exclusion lists.
    """
    return path.name in names

def is_selected_file(path: Path, names: list[str] = []) -> bool:
    """
    Check if the given file is in the selected files list.
    """
    return path.name in names

def is_in_skipped_dir(path: Path, names: list[str] = []) -> bool:
    """
    Check if the containing directory of the given file should be skipped based
    on its path and the provided exclusion lists.
    """
    return path.parent.name in names

def is_in_selected_dir(path: Path, names: list[str] = []) -> bool:
    """
    Check if the containing directory of the given file is in the selected
    directories list.
    """
    return path.parent.name in names

def get_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    suffix = f'_{config["suffix"]}' if config.get('suffix') else ''
    extension = f'.{config.get("extension", "")}'
    file_name = f'{info.name}{suffix}{extension}'
    return Path(config['path']) / info.cell_type / file_name

def setup_result_dir(output_dir: str) -> dict[str, Path]:
    """
    Prepare the output directories for saving results.
    """
    output_dirs = {}
    output_path = Path(output_dir)
    for i, dir_type in enumerate(['results', 'figures', 'stats']):
        output_dirs[dir_type] = output_path / f"{i+1:02d}_{dir_type}"
        output_dirs[dir_type].mkdir(parents=True, exist_ok=True)
    return output_dirs
