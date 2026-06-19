"""
This module contains the classes for representing spike likes, spike shapes,
and spike rising parts.
"""

from pathlib import Path

from dataclasses import dataclass

import numpy as np

import raap.data.signals as sig
import raap.data.recording as rec

@dataclass
class SpikeShape:
    """
    Represents the shape of a spike, including its time points, segments,
    timestamps, and time step (dt).
    """
    t: np.ndarray
    v: np.ndarray
    si: float
    timestamps: float

def load_spike_shapes(
    path: str | Path,
    signal_name: str,
    event_name: str,
    cut_window: tuple[float, float],
    fs: int
) -> list[SpikeShape]:
    """
    Extracts the spikes from the recording.
    """
    signal = rec.get_signal(path, signal_name).resample(fs)
    timestamps = rec.get_markers(path, event_name)

    shapes = []
    for timestamp in timestamps:
        segment = sig.cut_at(signal, timestamp, cut_window)
        shapes.append(SpikeShape(segment.t, segment.v, signal.si, timestamp))

    return shapes
