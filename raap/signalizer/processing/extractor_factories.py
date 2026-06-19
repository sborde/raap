from signalizer.processing.extractors import EventRelatedSegmentExtractor
from signalizer.processing.factories import EventRelatedSegmentCutterFactory
from signalizer.processing.retriever_factories import SignalRetrieverFactory, EventRetrieverFactory


class EventRelatedSegmentExtractorFactory:
    """Factory for creating TriggeredSegmentExtractor instances based on a configuration dictionary."""
    @staticmethod
    def create(config: dict) -> EventRelatedSegmentExtractor:
        signal_extractor = SignalRetrieverFactory.create(config['signal_retriever'])
        trigger_extractor = EventRetrieverFactory.create(config['event_retriever'])
        segmenter_cutter = EventRelatedSegmentCutterFactory.create(config['segment_cutter'])
        return EventRelatedSegmentExtractor(signal_extractor, trigger_extractor, segmenter_cutter)
