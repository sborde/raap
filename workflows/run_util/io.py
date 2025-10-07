"""Module for common input/output operations for the workflows."""

import abc

from typing import Callable, Protocol
from pathlib import Path

import pandas as pd

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

def list_selected_files(path: str, ext: str, checker: FileChecker) -> list[CellWrapper[Path]]:
    """Lists paths of the raw data that conforms the selection conditions."""
    return [
        CellWrapper(
            CellInfo(file_path.stem, file_path.parent.name), file_path
        )
        for file_path in Path(path).glob(f'**/*.{ext}')
        if checker(file_path)
    ]

def create_file_lister(input_config: dict) -> CellEnumerator:
    """
    Returns a function that lists files that conforms to the input
    conditions on the path provided as an input argument.
    """
    checker = create_file_checker(input_config)
    return lambda config: list_selected_files(
        config['path'], config['extension'], checker
    )

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
