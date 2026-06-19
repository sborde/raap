"""This module contains the functions that act as a backend of the same notebook."""

import sys

from dataclasses import dataclass
from typing import Self
from pathlib import Path

from dtw_classifier.dataclasses import (
    MeanSpikeShape,
    ProcessedSpikeShape
)
from dtw_classifier.transformers import get_rising_phase


sys.path.append('..')

import pandas as pd
import numpy as np

from scipy.signal import savgol_filter

from signalizer.model.experiment.recordings import MatRecording
from analyses.slowdown_descriptors import find_start_index
from data.spikes import SpikeShape, SpikeShapes, extract_triggered_segments

def load_training_data(
    cell_names: list[str],
    data_path: str,
    cut_params: dict,
    proc_params: dict
) -> list[tuple[list[ProcessedSpikeShape], list[ProcessedSpikeShape]]]:
    """Load and preprocess the training data for the given cell names."""
    processed = []
    for cell_name in cell_names:
        cell_shapes = []
        for spike_type in ['AISAP', 'RAAP']:
            file_name = f'{cell_name}_juxta{spike_type}.mat'
            recording = MatRecording(Path(data_path) / file_name)
            spike_shapes = extract_triggered_segments(
                recording, **cut_params
            )
            cell_shapes.append(
                SpikeShapes(
                    spike_shapes.times * 1000,
                    spike_shapes.segments.values,
                    spike_shapes.timestamps,
                    spike_shapes.dt
                )
            )
        processed.append(
            tuple(
                preprocess_shapes(shapes, **proc_params)
                for shapes in cell_shapes
            )
        )
    return processed

def mean_shapes_rising(
    shapes: list[ProcessedSpikeShape],
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> MeanSpikeShape:
    """Average the rising phase of a list of processed spike shapes."""
    rising_shapes = [
        get_rising_phase(s, start_ratio, end_ratio, keep_size=True)
        for s in shapes
    ]
    mean_t = rising_shapes[0].times
    mean_v = np.nanmean(np.stack([s.values for s in rising_shapes], axis=0), axis=0)
    return MeanSpikeShape(
        mean_t,
        mean_v,
        rising_shapes[0].dt,
        rising_shapes[0].peak_i,
        np.where(np.logical_not(np.isnan(mean_v)))[0][0]
    )

def load_test_data(
    cell: str,
    raw_path: str,
    label_path: str,
    cut_params: dict,
) -> SpikeShapes:
    labels = pd.read_csv(Path(label_path) / f'{cell}.csv')
    shapes = []
    recording = MatRecording(Path(raw_path) / f'{cell}.mat')
    spike_shapes = extract_triggered_segments(
        recording,
        signal_name=cut_params['signal_name'],
        event_name=cut_params['event_name'],
        cut_window=cut_params['cut_window'],
        fs=cut_params['fs']
    )
    shapes = SpikeShapes(
            spike_shapes.times * 1000,
            spike_shapes.segments.values,
            spike_shapes.timestamps,
            spike_shapes.dt
        )
    return shapes

def preprocess_shapes(
    shapes: SpikeShapes,
    smooth_params: dict,
    dv_threshold: float,
) -> list[ProcessedSpikeShape]:
    """Preprocess spike shapes by applying a Savitzky-Golay filter."""
    return [
        proc_shape for shape in shapes
        if (proc_shape := process_shape(shape, smooth_params, dv_threshold)) is not None
    ]

def process_shape(
    shape: SpikeShape,
    smooth_params: dict,
    dv_threshold: float,
) -> ProcessedSpikeShape | None:
    """Preprocess a single spike shape by applying a Savitzky-Golay filter."""
    t, v = shape.times, shape.values
    peak_i = np.argmin(np.abs(t - 0)).astype(int)
    v = savgol_filter(v, **smooth_params)
    start_i = find_start_index(
        v[:peak_i + 1],
        {'start_threshold': dv_threshold}
    )

    if start_i is None:
        return None

    amplitude = v[peak_i] - v[start_i]
    v = (v - v[start_i]) / amplitude
    return ProcessedSpikeShape(
        t, v, shape.timestamp, shape.dt, peak_i, start_i
    )

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
