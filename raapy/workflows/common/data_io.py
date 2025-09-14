"""Utility functions for file system operations."""

from pathlib import Path

from workflows.common.models.cell import CellWrapper, CellInfo


def list_input_files(config: dict) -> list[CellWrapper[dict[str, Path]]]:
    """
    List input files based on the data and input configuration.
    """
    all_files = list(
        Path(config['data']['raw_data']['path']).
        glob(f'**/*.{config['data']['raw_data']['extension']}')
    )
    selected = [
        CellWrapper(
            CellInfo(file_path.stem, file_path.parent.name),
            {'raw': file_path}
        ) for file_path in all_files
        if not need_to_skip(file_path, config['input'])
    ]

    for entry in selected:
        processed_path = make_processed_path(entry.info, config['data']['processed_data'])
        if not processed_path.exists():
            raise FileNotFoundError(
                f'Processed file for {entry.info.name} does not exist.'
            )
        entry.data['processed'] = processed_path
    return selected

def make_processed_path(info: CellInfo, config: dict) -> Path:
    """
    Create the path to the processed file for a given cell.
    """
    suffix = config.get('suffix', '')
    if suffix:
        suffix = f'_{suffix}'
    file_name = f'{info.name}{suffix}.{config["extension"]}'
    return Path(config['path']) / info.cell_type / file_name

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
