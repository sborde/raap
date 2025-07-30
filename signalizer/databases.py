import abc
from pathlib import Path

import pandas as pd

from signalizer.model.wrappers import Entity


class Database(abc.ABC):

    @abc.abstractmethod
    def find_all(self) -> list[Entity]: pass

    @abc.abstractmethod
    def find_by_subset(self, name: str) -> list[Entity]: pass

    @abc.abstractmethod
    def find_by_names(self, names: list[str]) -> list[Entity]: pass


class SheetDatabase(Database):
    """Database handles the entities and its properties, provides selecting queries"""

    def __init__(self, records: pd.DataFrame, subsets: pd.DataFrame = None):
        self._records = records
        self._subsets = subsets

    # def select(self, selector: DatasetQuery) -> pd.DataFrame: return selector.execute(self._records)

    def find_all(self) -> list[Entity]:
        """Returns all records"""
        return self._convert_records(self._records)

    def find_by_names(self, names: list[str]) -> list[Entity]:
        """Returns the records matching the names"""
        return self._convert_records(self._records.loc[self._records.index.isin(names), :])

    def find_by_subset(self, name: str) -> list[Entity]:
        """Returns the records from the given subset"""
        if self._subsets is None:
            raise ValueError('No subsets defined.')
        if name not in self._subsets.columns:
            raise ValueError(f'Subset {name} not found.')
        return self._convert_records(self._records.loc[self._subsets.loc[:, name] == 1, :])

    @staticmethod
    def _convert_records(records: pd.DataFrame) -> list[Entity]:
        """Converts the records to entities"""
        return [Entity(str(row.name), row.to_dict()) for _, row in records.iterrows()]

    @classmethod
    def from_table(cls, table_path: Path):
        subsets = None
        if table_path.suffix == '.csv':
            records = pd.read_csv(table_path).set_index('name')
        elif table_path.suffix in ['.xls', '.xlsx']:
            records = pd.read_excel(table_path).set_index('name')
            subsets_name = table_path.stem + '_subsets.xlsx'
            subset_path = table_path.parent / subsets_name
            if subset_path.exists():
                subsets = pd.read_excel(subset_path).set_index('name')
        else:
            raise ValueError('Unknown file extension.')

        return SheetDatabase(records, subsets)


class DatabaseFactory:
    """Factory for creating databases from different formats"""

    @staticmethod
    def create(config: dict) -> Database:
        if config['type'] == 'sheet':
            path = Path(config['path']) / f'{config["name"]}_{config["version"]}.{config["format"]}'
            return SheetDatabase.from_table(path)
        else:
            raise ValueError(f"Unsupported database type: {config['type']}")