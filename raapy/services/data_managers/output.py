
import abc
from typing import Optional, Protocol
from pathlib import Path

from pandas import DataFrame

from workflows.calculators.slowdown_descriptors import DescribedSpikeShape
from workflows.common.models.cell import CellInfo, CellWrapper
from workflows.common.file_io import save_json

NUMERIC_FEATURES = [
    'timestamp',
    'relative_dv_drop',
    'relative_rebound_time',
    'raw_power',
    'noise_power',
    'valley_time'
]

class SpikeFeatureExporter(Protocol):
    """A protocol for exporting spike features."""

    @abc.abstractmethod
    def export(self, cell: CellWrapper[list[DescribedSpikeShape]]):
        """Exports the spike features of the given cell."""


class ExportPathBuilder:
    """A class to build export paths based on cell information and configuration."""

    def __init__(self, base_path: str, suffix: Optional[str] = None):
        self._base_path = base_path
        self._suffix = f'_{suffix}' if suffix else ''

    def build(self, info: CellInfo) -> Path:
        """Builds the export path for the given cell information."""
        dir_path = Path(self._base_path) / info.group
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f'{info.name}{self._suffix}'


class SpikeToCsvExporter(SpikeFeatureExporter):
    """Exports spike features to a CSV file."""

    def __init__(self, path_builder: ExportPathBuilder):
        self._path_builder = path_builder

    def export(self, cell: CellWrapper[list[DescribedSpikeShape]]):
        """Exports the spike features to a CSV file."""
        save_path = self._path_builder.build(cell.info)
        data_table = DataFrame([
            {
                name: features.as_dict()[name] for name in NUMERIC_FEATURES
            } for features in cell.data
        ])
        data_table.to_csv(save_path.with_suffix('.csv'), index=False)

class SpikeToJsonExporter(SpikeFeatureExporter):
    """Exports spike features to a JSON file."""

    def __init__(self, path_builder: ExportPathBuilder):
        self._path_builder = path_builder

    def export(self, cell: CellWrapper[list[DescribedSpikeShape]]):
        """Exports the spike features to a JSON file."""
        save_path = self._path_builder.build(cell.info)
        export_data = [features.as_dict() for features in cell.data]
        save_json(save_path.with_suffix('.json').as_posix(), export_data)


class SpikeFeatureMultiFormatExporter(SpikeFeatureExporter):
    """Exports spike features to multiple formats."""

    def __init__(self, exporters: list[SpikeFeatureExporter]):
        self._exporters = exporters

    def export(self, cell: CellWrapper[list[DescribedSpikeShape]]):
        """Exports the spike features using all configured exporters."""
        for exporter in self._exporters:
            exporter.export(cell)

def create_exporter(config: dict) -> SpikeFeatureExporter:
    """Creates a spike feature exporter based on the given configuration."""
    path_builder = ExportPathBuilder(
        str(Path(config['path']) / '01_processed_data'),
        config.get('suffix')
    )
    return SpikeFeatureMultiFormatExporter([
        SpikeToCsvExporter(path_builder),
        SpikeToJsonExporter(path_builder)
    ])
