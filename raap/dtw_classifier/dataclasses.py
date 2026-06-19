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

    @property
    def x(self) -> np.ndarray:
        """Return the times as the x values for plotting."""
        return self.times
    @property
    def y(self) -> np.ndarray:
        """Return the values as the y values for plotting."""
        return self.values

@dataclass
class FirstDeriv:
    """A dataclass to hold the first derivative of a spike shape."""
    times: np.ndarray
    dv: np.ndarray

    @property
    def x(self) -> np.ndarray:
        """Return the times as the x values for plotting."""
        return self.times
    @property
    def y(self) -> np.ndarray:
        """Return the first derivative values as the y values for plotting."""
        return self.dv

@dataclass
class SecondDeriv:
    """A dataclass to hold the second derivative of a spike shape."""
    times: np.ndarray
    ddv: np.ndarray

    @property
    def x(self) -> np.ndarray:
        """Return the times as the x values for plotting."""
        return self.times

    @property
    def y(self) -> np.ndarray:
        """Return the second derivative values as the y values for plotting."""
        return self.ddv


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

