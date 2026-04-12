"""This module contains data classes used for DTW based classifier."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessedSpikeShape:
    """A dataclass to hold the processed spike shape and its label."""
    times: np.ndarray
    values: np.ndarray
    timestamp: float
    dt: float
    peak_i: int
    start_i: int

@dataclass
class FirstDeriv:
    """A dataclass to hold the first derivative of a spike shape."""
    times: np.ndarray
    dv: np.ndarray

@dataclass
class PhaseCurve:
    """A dataclass to hold the spike phase for DTW classification."""
    v: np.ndarray
    dvdt: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return the phase curve as a 2D array for DTW."""
        return np.stack([self.v, self.dvdt], axis=1)

@dataclass
class MeanSpikeShape:
    """A dataclass to hold the mean spike shape for a class."""
    times: np.ndarray
    values: np.ndarray
    dt: float
    peak_i: int
    start_i: int

