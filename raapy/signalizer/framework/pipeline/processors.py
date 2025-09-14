import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from signalizer.framework.pipeline.protocols import SavablePipeline, ExtractorPipeline, Pipeline
from signalizer.model.analysis.categorizations import CategorizationMatrix
from signalizer.model.analysis.projections import ProjectionMatrix
from signalizer.model.experiment.recordings import Recording
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike, SegmentMatrix
from signalizer.model.experiment.sequences import EventVector
from signalizer.model.wrappers import InstanceMatrix, Entity
from signalizer.processing.categorizers import InstanceCategorizer
from signalizer.processing.extractors import InstanceExtractor
from signalizer.processing.projectors import InstanceProjector


class EventRelatedSegmentExtractorPipeline(SavablePipeline[Recording, InstanceMatrix[EventVector, SegmentMatrix]]):

    def __init__(self, extractor: InstanceExtractor[EventVector, SegmentMatrix], build_time: str = None):
        self._extractor = extractor
        self._build_time = build_time or datetime.now().strftime("%Y-%m-%d %H:%M")

    def run(self, data: Recording) -> InstanceMatrix[EventVector, SegmentMatrix]:
        return self._extractor.extract(data)

    @property
    def name(self) -> str:
        return 'event_related_segment_extractor_pipeline_' + self.hash

    @property
    def hash(self) -> str:
        return hashlib.md5(self._extractor.name.encode('utf-8')).hexdigest()

    @property
    def result_path(self) -> Path:
        path = Path('.')
        for param_string in self._extractor.name.split(' '):
            path /= param_string
        return path

    @property
    def extractor(self) -> InstanceExtractor[EventVector, SegmentMatrix]:
        return self._extractor

    def save(self, path: Path, name: str = None) -> None:

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        file_name = name or f'{self.name}.ppln'
        pipeline_path = path / file_name

        if pipeline_path.exists():
            return

        config = {
            'hash': self.hash,
            'type': 'event_related_segment_extractor_pipeline',
            'build_time': self._build_time,
            'blocks': {
                'extractor': self._extractor.params
            }
        }

        temp_dir = (path / 'tmp')
        temp_dir.mkdir(parents=True, exist_ok=True)
        with open(temp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        try:
            with zipfile.ZipFile(pipeline_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.iterdir():
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
                        file_path.unlink()
            temp_dir.rmdir()
        except Exception as e:
            print(f"An error occurred: {e}")

        with open(path / f'{pipeline_path.stem}_README.md', 'w') as f:
            f.write(f"# Event Related Segment Extractor Pipeline\n")
            f.write(f"## Details\n")
            f.write(f"- **Name:** {self.name}\n")
            f.write(f"- **Created:** {self._build_time}\n")
            f.write(f"## Blocks\n")
            f.write(f"1. **Extractor:**\n{'\n'.join([f'\t{row}' for row in self._extractor.details.split('\n')])}")

    @classmethod
    def load(cls, path: Path) -> 'EventRelatedSegmentExtractorPipeline':

        temp_dir = (path.parent / 'tmp')
        temp_dir.mkdir()
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall(temp_dir)

        with open(temp_dir / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        shutil.rmtree(temp_dir)

        extractor = InstanceExtractor.load(config['blocks']['extractor'])
        return cls(extractor, config['build_time'])


class RecordingInstanceProjectorCategorizerPipeline[R: TimeReferenced, M: MatrixLike](
    SavablePipeline[Recording, InstanceMatrix[R, CategorizationMatrix]],
    ExtractorPipeline[InstanceMatrix[R, M], InstanceMatrix[R, CategorizationMatrix]]
):
    """Pipeline that extracts instances using a reference, projects them onto a vector space, and categorizes them."""

    def __init__(self,
                 extractor: InstanceExtractor[R, M],
                 projector: InstanceProjector[R, M],
                 categorizer: InstanceCategorizer[R, ProjectionMatrix],
                 training_time: str,
                 training_entities: list[Entity]):

        self._extractor = extractor
        self._projector = projector
        self._categorizer = categorizer

        self._training_time = training_time
        self._training_entities = sorted(
            training_entities,
            key=lambda entity: entity.label
        )

    @property
    def name(self) -> str:
        return 'instance_projector_categorizer_' + self.hash

    @property
    def hash(self) -> str:
        string_representation = f"{self._extractor.name}_{self._projector.name}_{self._categorizer.name}" + \
            f"_{'_'.join([entity.label for entity in self._training_entities])}"
        return hashlib.md5(string_representation.encode('utf-8')).hexdigest()

    @property
    def result_path(self) -> Path:
        path = Path('.') / self._projector.name / self._categorizer.name
        return path

    def save(self, path: Path, name: str = None) -> None:
        """Saves the pipeline to a file."""

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        file_name = name or f'{self.name}.ppln'
        pipeline_path = path / file_name

        if pipeline_path.exists():
            return

        # create the pipeline config file
        config = {
            'type': 'instance_projector_categorizer',
            'hash': self.hash,
            'training_time': self._training_time,
            'blocks': {
                'extractor': self._extractor.params,
                'projector': self._projector.params,
                'categorizer': self._categorizer.params
            }
        }

        temp_dir = (path / 'tmp')
        temp_dir.mkdir(parents=True, exist_ok=True)
        with open(temp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        with open(temp_dir / 'entities.csv', 'w') as f:
            f.write('name,version\n')
            for entity in self._training_entities:
                f.write(f"{entity.name},{entity.version}\n")

        # save block models if they were trained
        self._extractor.save(temp_dir / 'extractor.pkl')
        self._projector.save(temp_dir / 'projector.pkl')
        self._categorizer.save(temp_dir / 'categorizer.pkl')

        # bundle pipeline components into a zip file
        try:
            with zipfile.ZipFile(pipeline_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.iterdir():
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
                        file_path.unlink()
            temp_dir.rmdir()
        except Exception as e:
            print(f"An error occurred: {e}")

        with open(path / f'{pipeline_path.stem}_README.md', 'w') as f:
            f.write(f"# Instance Projector Categorizer Pipeline\n")
            f.write(f"## Details\n")
            f.write(f"- **Name:** {self.name}\n")
            f.write(f"- **Trained:** {self._training_time}\n")
            f.write(f"## Blocks\n")
            f.write(f"1. **Extractor:**\n{'\n'.join([f'\t{row}' for row in self._extractor.details.split('\n')])}\n")
            f.write(f"2. **Projector:**\n{'\n'.join([f'\t{row}' for row in self._projector.details.split('\n')])}\n")
            f.write(f"3. **Categorizer:**\n{'\n'.join([f'\t{row}' for row in self._categorizer.details.split('\n')])}\n")
            f.write(f"## Training Entities\n")
            for entity in self._training_entities:
                f.write(f"- **{entity.name}** (v{entity.version})\n")

    @classmethod
    def load(cls, path: Path) -> 'RecordingInstanceProjectorCategorizerPipeline':

        temp_dir = (path.parent / path.stem)
        temp_dir.mkdir()
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall(temp_dir)

        with open(temp_dir / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        with open(temp_dir / 'entities.csv', 'r') as f:
            entities = f.readlines()[1:]
            entities = [Entity.from_name_and_version(line.split(',')[0], line.split(',')[1]) for line in entities]

        config['blocks']['projector']['params']['model'] = temp_dir / 'projector.pkl'
        config['blocks']['categorizer']['params']['model'] = temp_dir / 'categorizer.pkl'

        extractor = InstanceExtractor.load(config['blocks']['extractor'])

        projector = InstanceProjector.load(config['blocks']['projector'])
        categorizer = InstanceCategorizer.load(config['blocks']['categorizer'])

        shutil.rmtree(temp_dir)

        return cls(extractor, projector, categorizer, config['training_time'], entities)

    def run(self, data: Recording) -> InstanceMatrix[R, CategorizationMatrix]:
        return self.run_on_instances(self._extractor.extract(data))

    def run_on_instances(self, instances: InstanceMatrix[R, M]) -> InstanceMatrix[R, CategorizationMatrix]:
        projection = self._projector.project(instances)
        return self._categorizer.assign(projection)


class OnInstancePipelineAdapter[R: TimeReferenced, M: MatrixLike](
    Pipeline[InstanceMatrix[R, M], InstanceMatrix[R, CategorizationMatrix]]
):

    def __init__(self, pipeline: ExtractorPipeline[InstanceMatrix[R, M], InstanceMatrix[R, CategorizationMatrix]]):
        self._adaptee = pipeline

    def run(self, data: InstanceMatrix[R, M]) -> InstanceMatrix[R, CategorizationMatrix]:
        return self._adaptee.run_on_instances(data)
