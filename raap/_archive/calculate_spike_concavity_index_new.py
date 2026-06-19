import argparse
from pathlib import Path
from typing import Any
import yaml
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np

from data.spikes import SpikeShapes

from signalizer.model.experiment.recordings import MatRecording
from signalizer.model.experiment.segments import Segment, SegmentMatrix
from signalizer.model.experiment.sequences import EventVector
from signalizer.model.general.data import Interval
from signalizer.model.general.series import DataSeries
from signalizer.model.temporal.data import TimeInterval
from signalizer.model.wrappers import InstanceMatrix
from signalizer.processing.extractors import EventRelatedSegmentExtractor
from signalizer.processing.factories import ProcessingSequence
from signalizer.processing.retrievers import SignalRetriever, EventRetriever
from signalizer.processing.segment_cutters import EventRelatedSegmentCutter
from signalizer.processing.signalprocessing.resamplers import Resampler

def extract_triggered_segments(
        recording, signal_name, event_name, cut_window, fs
) -> SpikeShapes:
    """
    Extracts the spikes from the recording.
    """

    spikes = EventRelatedSegmentExtractor(
        SignalRetriever(signal_name, ProcessingSequence([Resampler(fs)])),
        EventRetriever(event_name),
        EventRelatedSegmentCutter(TimeInterval(Interval.as_interval(cut_window)))
    ).extract(recording)

    return SpikeShapes(
        spikes.data.times,
        spikes.data.signals.as_matrix(),
        np.array([event.timestamp.time for event in spikes.reference])
    )


def analyse_concavity(input_path: Path, config: dict):

    recording = MatRecording(input_path)
    spikes = extract_triggered_segments(recording, **config['extractor'])

    timestamps = spikes.timestamps
    concavity_indices = spikes.concavity_indices

    shapes = spikes.segments
    phases = spikes.get_phases()

    rising_parts = spikes.get_rising_parts()
    rising_phases = rising_parts.get_phases()
    rising_phase_hulls = rising_phases.get_convex_hulls()

    output_dir = Path(config['output_dir']) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (5, 10)
    ax_per_plot = grid_size[0] * grid_size[1]
    fig, ax = plt.subplots(2 * grid_size[0], grid_size[1], figsize=(20, 20))

    for i in range(len(spikes)):

        row = i % ax_per_plot // grid_size[1]
        col = i % ax_per_plot % grid_size[1]

        grid_row = row * 2

        ax[grid_row, col].plot(shapes[i].x, shapes[i].y)
        ax[grid_row, col].plot(rising_parts[i].x, rising_parts[i].y)
        ax[grid_row, col].set_xticks([])
        ax[grid_row, col].set_yticks([])
        ax[grid_row, col].set_xticklabels([])
        ax[grid_row, col].set_yticklabels([])
        ax[grid_row, col].set_title(f'{timestamps[i]:.3f}')

        ax[grid_row+1, col].plot(phases[i].x, phases[i].y)
        ax[grid_row+1, col].plot(rising_phase_hulls[i].x, rising_phase_hulls[i].y)
        ax[grid_row+1, col].set_xticks([])
        ax[grid_row+1, col].set_yticks([])
        ax[grid_row+1, col].set_xticklabels([])
        ax[grid_row+1, col].set_yticklabels([])
        ax[grid_row+1, col].set_title(f'{concavity_indices[i]:.2f}')

        if i % ax_per_plot == ax_per_plot - 1 or i == len(spikes) - 1:
            print(f'Saving {i//ax_per_plot:03d}_shapes.png')
            fig.savefig(
                output_dir /
                f'{i//ax_per_plot:03d}_shapes.png'
            )
            plt.close(fig)
            fig, ax = plt.subplots(2 * grid_size[0], grid_size[1], figsize=(20, 20))

    return spikes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run concavity index analysis.')
    parser.add_argument(
        'input_path',
        type=str,
        help='Path of the input file.'
    )
    parser.add_argument(
        'analysis_config',
        type=str,
        help='Path for the config file.'
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    with open(args.analysis_config, 'r', encoding='utf-8') as config_file:
        configuration = yaml.safe_load(config_file)

    if input_path.is_dir():
        for file_name in input_path.glob('*.mat'):
            print(f'Processing {file_name}')
            analyse_concavity(file_name, configuration)
    elif input_path.is_file():
        analyse_concavity(input_path, configuration)
    else:
        raise ValueError(f'Invalid input path: {input_path}')
    
