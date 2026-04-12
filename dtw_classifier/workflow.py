"""This module contains the workflow steps."""

import sys

from pathlib import Path


sys.path.append('..')

import pandas as pd
import numpy as np

from scipy.signal import savgol_filter

from signalizer.model.experiment.recordings import MatRecording
from analyses.slowdown_descriptors import find_start_index
from data.spikes import SpikeShape, SpikeShapes, extract_triggered_segments

from dtw_classifier.dataclasses import ProcessedSpikeShape

SPIKE_TYPES = ['AISAP', 'RAAP']

def load_training_dataset(
    names: list[str],
    data_path: str,
    cut_params: dict,
    proc_params: dict
) -> list[dict[str, list[ProcessedSpikeShape]]]:
    """Load and preprocess the training data for the given cell names."""
    processed = [
        load_training_data(name, data_path, cut_params, proc_params)
        for name in names
    ]
    return processed

def load_training_data(
    name: str,
    data_path: str,
    cut_params: dict,
    proc_params: dict
) -> dict[str, list[ProcessedSpikeShape]]:
    """Load and preprocess the training data for the given cell name."""
    shapes = {}
    for spike_type in SPIKE_TYPES:
        file_name = f'{name}_juxta{spike_type}.mat'
        recording = MatRecording(Path(data_path) / file_name)
        shapes[spike_type] = load_spikes(recording, cut_params, proc_params)
    return shapes

def load_spikes(
    recording: MatRecording,
    cut_params: dict,
    proc_params: dict
) -> list[ProcessedSpikeShape]:
    """Load and preprocess the spikes from a recording."""
    raw_spikes = extract_triggered_segments(
        recording,
        **cut_params
    )
    return [
        proc_spike for raw_spike in raw_spikes
        if (proc_spike := process_spike(raw_spike, **proc_params))
        is not None
    ]

def process_spike(
    spike: SpikeShape,
    smooth_params: dict,
    dv_threshold: float,
) -> ProcessedSpikeShape | None:
    """Preprocess a single spike shape by applying a Savitzky-Golay filter."""
    t = spike.times
    v = savgol_filter(spike.values, **smooth_params)

    peak_i = int(np.argmin(np.abs(t - 0)))
    start_i = find_start_index(
        v[:peak_i + 1],
        {'start_threshold': dv_threshold}
    )

    if start_i is None:
        return None

    amplitude = v[peak_i] - v[start_i]
    v = (v - v[start_i]) / amplitude

    return ProcessedSpikeShape(
        t * 1000, v, spike.timestamp, spike.dt * 1000, peak_i, start_i
    )
