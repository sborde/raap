from pathlib import Path

from data.spikes import load_spike_shapes


def prepare_data(input_path: Path, config: dict):
    """
    Preprocess the spikes.
    """
    spikes = load_spike_shapes(input_path, config['extractor'])
    if 'sg_filter' in config['preprocess'] and config['preprocess']['sg_filter']:
        spikes = spikes.apply_sg_filter(**config['preprocess']['sg_filter'])
    if config['preprocess']['normalize'] == 'start_to_peak':
        spikes = spikes.start_to_peak_normalized()

    return spikes