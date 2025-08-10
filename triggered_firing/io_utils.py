
from collections import namedtuple

from pathlib import Path

import pandas as pd

from signalizer.model.experiment.recordings import MatRecording

from common.data_utils import CellWrapper, CellInfo

RecordingSpikePair = namedtuple('RecordingSpikePair', ['recording', 'spike_data'])

def make_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    base_dir = Path(config['path'])
    subdir = info.cell_type
    file_name = info.name
    suffix = f'_{config.get("suffix", "")}'
    extension = f'.{config.get("extension", "")}'
    return base_dir / subdir / f'{file_name}{suffix}{extension}'

def need_to_skip(path: Path, config: dict) -> bool:
    """
    Determine if a file should be skipped based on the configuration.
    """
    if config['mode'] == 'all':
        return (
            (config['skip_files'] and path.name in config['skip_files']) or
            (config['skip_dirs'] and path.parent.name in config['skip_dirs'])
        )
    elif config['mode'] == 'directory':
        return (
            path.parent.name not in config['select_dirs'] or
            (config['skip_files'] and path.name in config['skip_files'])
        )
    elif config['mode'] == 'file':
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