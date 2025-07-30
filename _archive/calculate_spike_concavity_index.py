import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull

from datetime import datetime

import hdf5storage
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


def find_start(spike_shape: Segment) -> int:
    dv = np.diff(spike_shape.values)
    max_dv, max_dv_i = np.max(dv), np.argmax(dv)
    threshold_v = 0.1 * max_dv
    max_indices = np.where(dv[:max_dv_i] < threshold_v)[0]
    return int(max_indices[-1] + 1) if len(max_indices) > 0 else None


def find_peak(spike_shape: Segment) -> int:
    return int(np.argmin(np.abs(spike_shape.times - 0)))


def calculate_area(polygon: DataSeries) -> float:
    return 0.5 * np.abs(
        np.dot(polygon.x, np.roll(polygon.y, 1)) - np.dot(polygon.y, np.roll(polygon.x, 1))
    )


def calculate_concavity_index(polygon: DataSeries, convex_hull: DataSeries) -> float:
    convex_area = calculate_area(convex_hull)
    phase_area = calculate_area(polygon)
    return (convex_area - phase_area) / convex_area if convex_area != 0 else 0


def calculate_spike_phase(spike_shape: Segment) -> DataSeries:
    return DataSeries(spike_shape.values[:-1], np.diff(spike_shape.values))


def extract_triggered_segments(
        recording, signal_name, event_name, cut_window, fs
) -> InstanceMatrix[EventVector, SegmentMatrix]:

    return EventRelatedSegmentExtractor(
        SignalRetriever(signal_name, ProcessingSequence([Resampler(fs)])),
        EventRetriever(event_name),
        EventRelatedSegmentCutter(TimeInterval(Interval.as_interval(cut_window)))
    ).extract(recording)


def analyse_concavity(config: dict):

    input_path = Path(config['input']['dir']) / config['input']['file']
    output_path = Path(config['output']['dir'])

    figure_path = output_path / input_path.stem

    signal_name = 'Wideband'
    recording = MatRecording(input_path)
    signal_names = recording.list_waveforms()
    if signal_name not in signal_names:
        for name in signal_names:
            if signal_name.lower() == name.lower():
                signal_name = name
                break

    if signal_name not in signal_names:
        raise ValueError(f'Signal {signal_name} not found in {input_path}')

    spikes = extract_triggered_segments(
        MatRecording(input_path),
        signal_name,
        'spk',
        [-0.001, 0.002],
        40000
    )

    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    events: EventVector = spikes.reference
    shapes: SegmentMatrix = spikes.data

    concavity_indices = []
    for i, (event, shape) in enumerate(zip(events, shapes)):

        print(f'Calculate concavity index for {i + 1}/{len(spikes)} spikes', end='\r')

        spike_phase = calculate_spike_phase(shape)
        start_i = find_start(shape)
        peak_i = find_peak(shape)

        if start_i is None or start_i > peak_i:
            concavity_index = np.nan
        else:

            rising_phase_plot = DataSeries(shape.values[start_i:peak_i], np.diff(shape.values[start_i:peak_i+1]))

            hull = ConvexHull(np.stack([rising_phase_plot.x, rising_phase_plot.y], axis=1))
            hull_vertices_indices = hull.vertices

            convex_hull = DataSeries(
                rising_phase_plot.x[hull_vertices_indices],
                rising_phase_plot.y[hull_vertices_indices]
            )

            concavity_index = calculate_concavity_index(rising_phase_plot, convex_hull)

        concavity_indices.append(concavity_index * 100)  # Convert to percentage

    concavity_indices = np.array(concavity_indices)
    print(f'Concavity index calculated for {len(spikes)} spikes')

    if config['plots']['individual']:

        sorted_indices = np.argsort(concavity_indices)

        grid_size = (5, 10)
        spike_per_grid = grid_size[0] * grid_size[1]
        fig, ax = plt.subplots(grid_size[0] * 2, grid_size[1], figsize=(20, 20))
        for i in range(len(events)):

            idx = int(sorted_indices[i])
            event = events[idx]
            shape = shapes[idx]

            print(f'Plotting {i + 1}/{len(spikes)} spikes', end='\r')

            spike_phase = DataSeries(shape.values[:-1], np.diff(shape.values))

            row_index = (i % spike_per_grid) // grid_size[1]
            col_index = (i % spike_per_grid) % grid_size[1]

            shape_axis = ax[2 * row_index][col_index]
            shape_axis.plot(shape.times, shape.values, 'b-')
            shape_axis.set_xticks([])
            shape_axis.set_yticks([])
            shape_axis.set_xticklabels([])
            shape_axis.set_yticklabels([])

            phase_axis = ax[2 * row_index + 1][col_index]
            phase_axis.plot(spike_phase.x, spike_phase.y, 'b-')
            phase_axis.set_xticks([])
            phase_axis.set_yticks([])
            phase_axis.set_xticklabels([])
            phase_axis.set_yticklabels([])

            start_i = find_start(shape)
            peak_i = find_peak(shape)

            if start_i is not None and start_i < peak_i-2:

                shape_axis.plot(shape.times[start_i:peak_i+1], shape.values[start_i:peak_i+1], 'r-')

                phase_plot = DataSeries(shape.values[start_i:peak_i], np.diff(shape.values[start_i:peak_i+1]))
                hull = ConvexHull(np.stack([phase_plot.x, phase_plot.y], axis=1))
                hull_vertices_indices = hull.vertices

                convex_hull = DataSeries(phase_plot.x[hull_vertices_indices], phase_plot.y[hull_vertices_indices])
                phase_axis.plot(convex_hull.x, convex_hull.y, 'r-')

            shape_axis.set_title(f'{event.timestamp.time:.3f} {event.timestamp.unit}')
            phase_axis.set_title(f'{concavity_indices[idx]:.4f}')

            if i % spike_per_grid == spike_per_grid-1 or i == len(spikes) - 1:
                plt.tight_layout(pad=0.5)
                fig.savefig(figure_path / f'spike_{i // spike_per_grid}.png')
                plt.close(fig)
                fig, ax = plt.subplots(grid_size[0] * 2, grid_size[1], figsize=(20, 20))

        print(f'Plotting {len(spikes)} spikes done')

    if config['plots']['swarm_plot']:
        plt.figure(figsize=(20, 10))
        sns.swarmplot(x=concavity_indices)
        plt.xlabel('Concavity Index')
        plt.savefig(figure_path / '_concavity_index_swarm_plot.png')
        plt.close()

        print('Swarm plot done')

    if config['plots']['histogram']:
        plt.figure(figsize=(20, 10))
        plt.hist(concavity_indices, bins=50)
        plt.plot(concavity_indices, np.zeros_like(concavity_indices), 'r.', markersize=10)
        plt.xlabel('Concavity Index')
        plt.ylabel('Count')
        plt.savefig(figure_path / '_concavity_index_hist_and_scatter.png')
        plt.close()

    if config['plots']['cdf']:
        sorted_concavity_indices = np.sort(concavity_indices)
        cumulative_probabilities = np.arange(1, len(sorted_concavity_indices) + 1) / len(sorted_concavity_indices)

        plt.figure(figsize=(20, 10))
        plt.plot(sorted_concavity_indices, cumulative_probabilities)

        for ratio, color in zip([0.975, 0.95, 0.925, 0.9], ['r', 'g', 'b', 'c']):
            index = int(len(sorted_concavity_indices) * ratio)
            threshold_concavity_index = sorted_concavity_indices[index]
            plt.hlines(ratio, 0, threshold_concavity_index, colors=color, linestyles='dashed')
            plt.vlines(threshold_concavity_index, 0, ratio, colors=color, linestyles='dashed')

        plt.xlabel('Concavity Index')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function of Concavity Indices')
        plt.grid(True)
        plt.savefig(figure_path / '_concavity_index_cdf.png')
        plt.close()

    if config['save']:
        index_vector = concavity_indices.reshape(-1, 1)

        new_path = input_path.parent / 'with_index' / input_path.name

        dtype = np.dtype([('indices', '<f8', index_vector.shape), ('title', 'O', (1,))])

        data_file = hdf5storage.loadmat(str(input_path))
        data_file['spk_concavity'] = np.array([(index_vector, 'concavity_index')], dtype=dtype)

        hdf5storage.savemat(str(new_path), data_file, format='7.3',
                            appendmat=True, oned_as='column',
                            truncate_existing=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run concavity index analysis.')
    parser.add_argument('input_dir', type=str, help='Input dir path.')
    parser.add_argument('output_dir', type=str, help='Output dir path.')
    args = parser.parse_args()

    configuration: dict[str, Any] = {
        'input': {
            'dir': args.input_dir
        },
        'output': {
            'dir': args.output_dir
        },
        'plots': {
            'individual': True,
            'swarm_plot': False,
            'histogram': True,
            'cdf': True
        },
        'save': True
    }

    date_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(Path(args.output_dir) / f'logfile_{date_string}.log', 'a') as log_file:
        with open(Path(args.output_dir) / f'summary_{date_string}.csv', 'a') as csv_file:
            csv_file.write('name,start,finish,status\n')
            for file_name in Path(args.input_dir).glob('*.mat'):
                try:
                    configuration['input']['file'] = str(file_name)
                    print(f'Processing {file_name}')
                    log_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Processing {file_name} started\n')
                    log_file.flush()
                    csv_file.write(f'{file_name},{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},')
                    csv_file.flush()
                    analyse_concavity(configuration)
                    log_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Processing {file_name} finished\n')
                    log_file.flush()
                    csv_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},success\n')
                    csv_file.flush()
                except ValueError as e:
                    log_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Error processing {file_name}: {e}\n')
                    log_file.flush()
                    csv_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},error\n')
                    csv_file.flush()
                except Exception as e:
                    log_file.write(f'Unexpected error processing {file_name}: {e}\n')
                    log_file.flush()
                    csv_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},error\n')
                    csv_file.flush()
