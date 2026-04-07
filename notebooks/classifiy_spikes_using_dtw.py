"""This module contains the functions that act as a backend of the same notebook."""

from dataclasses import dataclass
import sys
from typing import Self

sys.path.append('..')
from pathlib import Path

import numpy as np

from scipy.signal import savgol_filter

from common.data_utils import CellInfo, CellWrapper
from analyses.slowdown_descriptors import find_start_index
from data.spikes import SpikeShape, SpikeShapes, extract_triggered_segments
from signalizer.model.experiment.recordings import MatRecording


def load_training_data(
    cells: list[str],
    path: str,
    cut_params: dict,
) -> list[tuple[SpikeShapes, SpikeShapes]]:
    shapes = []
    for cell in cells:
        cell_shapes = []
        for i, spike_type in enumerate(['AISAP', 'RAAP']):
            file_name = f'{cell}_juxta{spike_type}.mat'
            recording = MatRecording(Path(path) / file_name)
            spike_shapes = extract_triggered_segments(
                recording,
                signal_name=cut_params['signal_name'],
                event_name=cut_params['event_name'],
                cut_window=cut_params['cut_window'],
                fs=cut_params['fs']
            )
            cell_shapes.append(
                SpikeShapes(
                    spike_shapes.times * 1000,
                    spike_shapes.segments.values,
                    spike_shapes.timestamps,
                    spike_shapes.dt
                )
            )
        shapes.append(tuple(cell_shapes))
    return shapes

@dataclass
class ProcessedSpikeShape:
    """A dataclass to hold the processed spike shape and its label."""
    times: np.ndarray
    values: np.ndarray
    timestamp: float
    dt: float
    peak_i: int
    start_i: int

    def get_rising(
        self,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
    ) -> Self:
        """Extract the rising phase of the spike shape."""
        return get_rising_phase(self, start_ratio, end_ratio)

@dataclass
class MeanSpikeShape:
    """A dataclass to hold the mean spike shape for a class."""
    times: np.ndarray
    values: np.ndarray
    dt: float
    peak_i: int
    start_i: int

def preprocess_shapes(
    shapes: SpikeShapes,
    smooth_params: dict,
    dv_threshold: float,
) -> list[ProcessedSpikeShape]:
    """Preprocess spike shapes by applying a Savitzky-Golay filter."""
    processed = []
    for spike in shapes:
        t, v = spike.times, spike.values
        peak_i = np.argmin(np.abs(t - 0)).astype(int)
        v = savgol_filter(v, **smooth_params)
        start_i = find_start_index(
            v[:peak_i + 1],
            {'start_threshold': dv_threshold}
        )
        if start_i is None:
            continue
        amplitude = v[peak_i] - v[start_i]
        v = (v - v[start_i]) / amplitude
        processed.append(
            ProcessedSpikeShape(t, v, spike.timestamp, spike.dt, peak_i, start_i)
        )
    return processed

def average_shapes(
    shapes: list[ProcessedSpikeShape],
    dv_threshold: float,
) -> MeanSpikeShape:
    """Average a list of processed spike shapes to get a mean shape."""
    if not shapes:
        raise ValueError("No shapes to average.")
    t = shapes[0].times
    dt = shapes[0].dt
    v = np.mean([s.values for s in shapes], axis=0)
    peak_i = np.argmin(np.abs(t - 0)).astype(int)
    start_i = find_start_index(
        v[:peak_i + 1],
        {'start_threshold': dv_threshold}
    )
    return MeanSpikeShape(t, v, dt, peak_i, start_i)

def get_rising_phase(
    shape: ProcessedSpikeShape,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> ProcessedSpikeShape:
    """Extract the rising phase of a spike shape."""
    peak_i = shape.peak_i
    start_i = shape.start_i
    left_i = int(start_ratio * (peak_i - start_i) + start_i)
    right_i = int(end_ratio * (peak_i - start_i) + start_i)
    return ProcessedSpikeShape(
        shape.times[left_i:right_i + 1],
        shape.values[left_i:right_i + 1],
        shape.timestamp,
        shape.dt,
        right_i - left_i,
        0
    )