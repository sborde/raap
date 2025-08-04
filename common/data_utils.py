from pathlib import Path


import yaml

class CellInfo:
    """
    A class that holds information about a cell. Name of the cell and its
    type are mandatory.
    """
    def __init__(self, name: str, cell_type: str):
        self._name = name
        self._cell_type = cell_type

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._cell_type


class CellWrapper[T]:
    """A wrapper class for a cell's data and its information."""
    def __init__(self, cell_info: CellInfo, data: T):
        self._cell_info = cell_info
        self._data = data

    @property
    def info(self) -> CellInfo:
        """Get the information about the cell."""
        return self._cell_info

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._cell_info.name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._cell_info.cell_type

    @property
    def data(self) -> T:
        """Get the data of the cell."""
        return self._data


class CellPath:
    """
    A class that holds the path information for a cell.
    """
    def __init__(self, info: CellInfo, path: Path):
        self._info = info
        self._path = path

    @property
    def info(self) -> CellInfo:
        """Get the information about the cell."""
        return self._info

    @property
    def name(self) -> str:
        """Get the name of the cell."""
        return self._info.name

    @property
    def cell_type(self) -> str:
        """Get the type of the cell."""
        return self._info.cell_type

    @property
    def path(self) -> Path:
        """Get the path of the cell."""
        return self._path


class GroupInfo:
    """
    A class that holds information about a group of cells.
    It contains the name of the group and the members of the group.
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the group."""
        return self._name


class GroupWrapper[T]:
    """A wrapper class for a group of cells' data and its information."""
    def __init__(self, group_info: GroupInfo, members: list[T]):
        self._group_info = group_info
        self._members = members

    @property
    def info(self) -> GroupInfo:
        """Get the information about the group."""
        return self._group_info

    @property
    def name(self) -> str:
        """Get the name of the group."""
        return self._group_info.name

    @property
    def members(self) -> list[T]:
        """Get the members of the group."""
        return self._members

    @property
    def n_members(self) -> int:
        """Get the number of members in the group."""
        return len(self._members)

class PathEntry:
    """
    A named tuple to represent a file path and its subdirectory.
    """
    def __init__(
        self,
        path: Path,
        subdir: str | None,
        spike_type: str = ''
    ):
        self.path = path
        self.subdir = subdir
        self.spike_type = spike_type


def get_input_file_paths(input_config: dict) -> list[PathEntry]:
    """
    Get the input file paths from the input config file.
    """
    base_path = Path(input_config['base_path'])

    if input_config['mode'] == 'all':
        files = [PathEntry(base_path / f, None) for f in base_path.glob('*.mat')]
    elif input_config['mode'] == 'multi_dir':
        files = []
        for dir_name in input_config['dirs']:
            files.extend([
                PathEntry(base_path / dir_name / f, dir_name)
                for f in (base_path / dir_name).glob('*.mat')
            ])
    elif input_config['mode'] == 'multi_file':
        files = [
            PathEntry(base_path / f['path'] / f['name'], f['path'])
            for f in input_config['files']
        ]
    else:
        raise ValueError(f'Invalid mode: {input_config["mode"]}')

    return files

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def handle_output_dir(config: dict) -> dict:
    """
    Create the output directory if it does not exist.
    """
    if not config['output_dir']:
        raise ValueError('Output directory is not specified in the config.')

    output_path = Path(config['output_dir'])
    if 'smooth' in config['descriptor'] and config['descriptor']['smooth']:
        output_path = (
            output_path /
            f"w{config['descriptor']['smooth']['window_length']}_"
            f"p{config['descriptor']['smooth']['polyorder']}"
        )
    else:
        output_path = output_path / 'raw'

    config['output_dir'] = str(output_path)

    for subdir in ['Figures', 'Preprocessed']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    if config['descriptor']['debug']:
        (output_path / 'Figures' / 'Individual').mkdir(parents=True, exist_ok=True)

    return config
