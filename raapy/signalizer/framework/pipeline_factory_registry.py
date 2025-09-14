from signalizer.framework.pipeline.processors import EventRelatedSegmentExtractorPipeline
from signalizer.processing.extractor_factories import EventRelatedSegmentExtractorFactory

pipeline_factories = {
    'event_related_segment_extractor': EventRelatedSegmentExtractorFactory,
    'event_related_segment_extractor_pipeline': EventRelatedSegmentExtractorPipeline,
}
