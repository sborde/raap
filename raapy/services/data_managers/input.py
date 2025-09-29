"""Utility functions for file system operations."""

import abc
from pathlib import Path
from typing import Optional, Protocol

from workflows.common.models.cell import CellInfo


def list_input_files(config: dict) -> list[CellInfo]:
    """
    List input files based on the data and input configuration.
    """
    selected_paths = list_selected_paths(
        config['data']['raw_data'], config['input']
    )
    return [
        CellInfo(file_path.stem, file_path.parent.name)
        for file_path in selected_paths
    ]

    # for entry in selected:
    #     processed_path = make_processed_path(entry.info, config['data']['processed_data'])
    #     if not processed_path.exists():
    #         raise FileNotFoundError(
    #             f'Processed file for {entry.info.name} does not exist.'
    #         )
    #     entry.data['processed'] = processed_path
    # return selected

def list_selected_paths(data_config: dict, input_config: dict) -> list[Path]:
    """
    List selected paths based on the input configuration.
    """
    base_dir = Path(data_config['path'])
    if not base_dir.exists() or not base_dir.is_dir():
        raise ValueError(
            f"The path {base_dir.name} is not a valid directory."
        )
    all_files = list(base_dir.rglob(f'*.{data_config["extension"]}'))

    is_selected = create_selector(input_config)

    return [
        file_path for file_path in all_files if is_selected(file_path)
    ]

class FileSelector(Protocol):
    """A protocol for file selection strategies."""

    @abc.abstractmethod
    def __call__(self, path: Path) -> bool:
        """Determine if a file should be selected."""


class AllModeSelector(FileSelector):
    """
    A selector class that determines if a file should be selected when
    the selection is 'all'. In this case all files are selected except
    those that are explicitly skipped by name or directory.
    """
    def __init__(
        self,
        skip_files: Optional[list[str]] = None,
        skip_dirs: Optional[list[str]] = None
    ):
        self._skip_files = skip_files or []
        self._skip_dirs = skip_dirs or []

    def __call__(self, path: Path) -> bool:
        return not (
            path.name in self._skip_files or
            path.parent.name in self._skip_dirs
        )


class DirectoryModeSelector(FileSelector):
    """
    A selector class that determines if a file should be selected when
    the selection is 'directory'. In this case, only files in the
    selected directories are included, except those that are explicitly
    skipped by name.
    """
    def __init__(
        self,
        selected_dirs: list[str],
        skip_files: Optional[list[str]] = None
    ):
        self._select_dirs = selected_dirs
        self._skip_files = skip_files or []

    def __call__(self, path: Path) -> bool:
        return (
            path.parent.name in self._select_dirs and
            path.name not in self._skip_files
        )


class FileModeSelector(FileSelector):
    """
    A selector class that determines if a file should be selected when
    the selection is 'file'. In this case, only files that are explicitly
    selected by name are included.
    """
    def __init__(self, selected_files: list[str]):
        self._select_files = selected_files

    def __call__(self, path: Path) -> bool:
        return path.name in self._select_files

def create_selector(config: dict) -> FileSelector:
    """
    Create a file selector based on the configuration.
    """
    if config['mode'] == 'all':
        return AllModeSelector(
            config.get('skip_files'), config.get('skip_dirs')
        )
    if config['mode'] == 'directory':
        if 'select_dirs' not in config:
            raise ValueError(
                "Configuration must include 'select_dirs' for directory mode"
            )
        return DirectoryModeSelector(
            config['select_dirs'], config.get('skip_files')
        )
    if config['mode'] == 'file':
        if 'select_files' not in config:
            raise ValueError(
                "Configuration must include 'select_files' for file mode"
            )
        return FileModeSelector(config['select_files'])
    raise ValueError("Invalid configuration mode")

def make_grouped_path(cell: CellInfo, config: dict) -> Path:
    """
    Create the path to the raw file for a given cell.
    """
    file_name = cell.name
    if 'suffix' in config and config['suffix']:
        file_name += '_' + config["suffix"]
    file_name += '.' + config["extension"]
    return Path(config['path']) / cell.group / file_name

def make_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    suffix = config.get('suffix', '')
    if suffix:
        suffix = f'_{suffix}'
    file_name = f'{info.name}{suffix}.{config["extension"]}'
    return Path(config['path']) / info.group / file_name

