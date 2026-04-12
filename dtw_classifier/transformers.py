"""This module contains functions that transforms spikes shapes."""


import numpy as np

from dtw_classifier.dataclasses import (
    FirstDeriv,
    ProcessedSpikeShape,
    PhaseCurve
)


def rising_phase(
    shape: ProcessedSpikeShape,
    start: float = 0.0,
    end: float = 1.0,
    keep_size: bool = False
) -> ProcessedSpikeShape:
    """Extract the rising phase of a spike shape."""
    peak_i = shape.peak_i
    start_i = shape.start_i
    left_i = int(start * (peak_i - start_i) + start_i)
    right_i = int(end * (peak_i - start_i) + start_i)

    if keep_size:
        new_v = np.nan * np.ones_like(shape.values)
        new_v[left_i:right_i + 1] = shape.values[left_i:right_i + 1]
        new_t = shape.times
    else:
        new_v = shape.values[left_i:right_i + 1]
        new_t = shape.times[left_i:right_i + 1]

    return ProcessedSpikeShape(
        new_t,
        new_v,
        shape.timestamp,
        shape.dt,
        right_i - left_i,
        0
    )

def first_derivative(
    shape: ProcessedSpikeShape
) -> FirstDeriv:
    """Calculate the first derivative of a spike shape."""
    dv = np.diff(shape.values)
    return FirstDeriv(
        shape.times[:-1],
        dv
    )

def phase_curve(
    shape: ProcessedSpikeShape,
) -> PhaseCurve:
    """Extract a phase of a spike shape."""
    return PhaseCurve(
        shape.values[:-1],
        np.diff(shape.values)
    )

def normalize_phase_curve_range(
    phase_curve: PhaseCurve
) -> PhaseCurve:
    """Normalize the range of a phase curve."""
    v_normalized = normalize_v_range(phase_curve.v)
    dvdt_normalized = normalize_v_range(phase_curve.dvdt)
    return PhaseCurve(v_normalized, dvdt_normalized)

def resample_norm_phase_curve(
    phase_curve: PhaseCurve,
    n_points: int
) -> PhaseCurve:
    """Resample a phase curve to have a fixed number of points."""
    new_v = np.linspace(0, 1, n_points)
    new_dvdt = np.interp(new_v, phase_curve.v, phase_curve.dvdt)
    return PhaseCurve(new_v, new_dvdt)

def normalize_v_range(
    v: np.ndarray,
) -> np.ndarray:
    """Calculate the range of the vector norm of a spike shape."""
    v_min = np.nanmin(v)
    v_max = np.nanmax(v)
    return (v - v_min) / (v_max - v_min)
