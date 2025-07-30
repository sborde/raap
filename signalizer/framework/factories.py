
from signalizer.framework.analyzers import IndividualProcessingAnalysis, IndividualAnalyzer, TrainableIndividualAnalysis
from signalizer.framework.pipeline.protocols import EntityHandler
from signalizer.framework.pipeline_factory_registry import pipeline_factories
from signalizer.io.savers import EntitySaver


class AnalysisFactory:

    @staticmethod
    def create(config: dict) -> IndividualAnalyzer:
        if config['pipeline']['type'] == 'processing':
            analyzer = PipelineFactory.create_processing_analyzer(config['pipeline'])
        elif config['pipeline']['type'] == 'training':
            analyzer = PipelineFactory.create_trainable_analyzer(config['pipeline'])
        else:
            raise ValueError(f"Unsupported pipeline type: {config['pipeline']['type']}")

        return analyzer.add_handlers(AnalysisFactory._create_post_processors(config['postprocessing']))

    @staticmethod
    def _create_post_processors(configs: dict) -> list[EntityHandler]:
        try:
            return [PipelineFactory.create_entity_handler(config) for config in configs]
        except ValueError as e:
            raise ValueError(f'Unsupported post-processor type: {e}') from e


class PipelineFactory:

    @staticmethod
    def create_processing_analyzer[I, O](config: dict) -> IndividualProcessingAnalysis[I, O]:
        if config['mode'] == 'build':
            params = config['params']
            pipeline = pipeline_factories[params['name']].create(params['params'])
        elif config['mode'] == 'load':
            raise NotImplemented("Loading pipelines is not implemented yet.")
        else:
            raise ValueError(f"Unsupported pipeline creation mode: {config['mode']}")

        return IndividualProcessingAnalysis(pipeline)

    @staticmethod
    def create_trainable_analyzer[I, O](config: dict) -> TrainableIndividualAnalysis[I, O]:
        # TrainableIndividualAnalysis(pipeline)
        raise NotImplemented("Training pipelines are not yet supported.")

    @staticmethod
    def create_entity_handler[I](config: dict) -> EntityHandler[I]:
        if config['type'] == 'save':
            handler = EntitySaver.from_config(config['params'])
        elif config['type'] == 'plot':
            raise NotImplemented("Plotting handler is not implemented yet.")
        else:
            raise ValueError(f"Unsupported handler type: {config['type']}")
        return handler
