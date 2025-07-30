from signalizer.processing.factories import ProcessingSequence
from signalizer.processing.retrievers import SignalRetriever, EventRetriever


class SignalRetrieverFactory:
    """Factory for creating SignalRetriever instances based on a configuration dictionary."""
    @staticmethod
    def create(config: dict) -> SignalRetriever:
        signal_name = config['signal_name']

        if 'processing_sequence' in config:
            processing_sequence = ProcessingSequence.from_config(config['processing_sequence'])
        else:
            processing_sequence = ProcessingSequence([])

        return SignalRetriever(signal_name, processing_sequence)


class EventRetrieverFactory:
    """Factory for creating EventRetriever instances based on a configuration dictionary."""
    @staticmethod
    def create(config: dict) -> EventRetriever:
        event_name = config['event_name']
        return EventRetriever(event_name)
