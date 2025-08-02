
import abc

from typing import Protocol
from pathlib import Path


class FilePathCollector(Protocol):
    """
    A protocol for a callable that collects file paths based on a given path.
    """
    @abc.abstractmethod
    def __call__(self, path: Path) -> list[Path]:
        """Collects file paths from the specified path."""


class SingleFilePathCollector:
    """ A callable class that collects a single file path based on the given name. """
    def __call__(self, path: Path) -> list[Path]:
        """
        Collects a single file path based on the given name.
        Args:
            name (Path): The name of the file to collect.
        Returns:
            list[Path]: A list containing the file path.
        """
        return [path] if path.is_file() else []


class DirectoryFilePathCollector:
    """ A callable class that collects all files in the given directory
    of the specified extension, optionally skipping files based on their names.
    """
    def __init__(
        self,
        extension: str = 'mat',
        skip_files: list[str] = []
    ):
        self._extension = extension
        self._skip_files = set(skip_files or [])

    def __call__(self, path: Path) -> list[Path]:
        """
        Collects all files in the given directory and its subdirectories.
        Args:
            path (Path): The directory to traverse.
        Returns:
            list[Path]: A list of file paths in the directory.
        """
        return [
            file_path for file_path in path.glob(f'*.{self._extension}')
            if file_path.is_file() and file_path.name not in self._skip_files
        ]


class DirectoryGroupedFilePathCollector:
    """ A callable class that collects all files in the given directory's
    subdirectories treating them as groups. Specific groups and files can
    be skipped.
    """
    def __init__(
        self,
        extension: str = 'mat',
        skip_files: list[str] = [],
        skip_dirs: list[str] = []
    ):
        self._dir_collector = DirectoryFilePathCollector(
            extension=extension,
            skip_files=skip_files
        )
        self._skip_dirs = set(skip_dirs or [])

    def __call__(self, path: Path) -> list[Path]:
        """
        Collects all files in the given directory and its subdirectories.
        Args:
            path (Path): The directory to traverse.
        Returns:
            list[Path]: A list of file paths in the directory.
        """
        result = []
        for sub_dir in path.iterdir():
            if not sub_dir.is_dir() or sub_dir.name in self._skip_dirs:
                continue
            result.extend(self._dir_collector(sub_dir))
        return result


def create_file_path_collector(config: dict) -> FilePathCollector:
    """
    Create a file path collector based on the configuration.
    """
    if 'mode' not in config:
        raise ValueError('Input mode is not specified in the config.')
    if config['mode'] == 'file':
        return SingleFilePathCollector()
    if config['mode'] == 'directory':
        return DirectoryFilePathCollector(
            extension=config.get('extension', 'mat'),
            skip_files=config.get('skip_files', []),
        )
    if config['mode'] == 'grouped':
        return DirectoryGroupedFilePathCollector(
            extension=config.get('extension', 'mat'),
            skip_files=config.get('skip_files', []),
            skip_dirs=config.get('skip_dirs', []),
        )
    raise ValueError(f"Unknown mode: {config['mode']}")


def collect_paths(config: dict) -> list[Path]:
    """
    Collects input files based on the configuration.
    Args:
        config (dict): dictionary containing input configuration.
    Returns:
        list[CellPath]: List of input file paths.
    """

    collector = create_file_path_collector(config)

    if 'path' not in config:
        raise ValueError('Input path is not specified in the config.')

    input = Path(config['path'])
    if not input.exists() or not input.is_dir():
        raise ValueError(f"Input path {input} is not a valid directory.")

    return collector(input)
