from pathlib import Path

from signalizer.io.loaders import Loader
from signalizer.model.wrappers import EntityData, Entity


class DataManager[T]:
    """Loads the corresponding data for the given entities."""

    def __init__(self, path: Path, loader: Loader[T]):
        self._path = path
        self._loader = loader

    def get_data(self, entities: list[Entity]) -> list[EntityData[T]]:
        return [entity.wrap(self._load(entity)) for entity in entities]

    def _load(self, entity: Entity) -> T:
        return self._loader(self._path, entity.label)

    @classmethod
    def from_dataset[T](cls, path: Path) -> 'DataManager[T]':

        config_path = path / 'dataset_config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration of the dataset is not found: {config_path}")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        return DataManager[T](path, Loader.create(config))
