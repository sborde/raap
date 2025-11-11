

import abc

from collections import namedtuple
from dataclasses import dataclass
import functools
from typing import Self, Protocol

import numpy as np
import pandas as pd

from pandas import DataFrame

from scipy.ndimage import gaussian_filter1d

from param_classes.workflows.peritrigger_analysis import (
    PeriTriggerAnalysisParams
)

class Histogram:
    """A class that represents a histogram."""
    def __init__(
        self,
        centers: np.ndarray,
        values: np.ndarray,
        data_unit: str = '',
        value_unit: str = ''
    ) -> None:
        self._centers = centers
        self._values = values
        self._data_unit = data_unit
        self._value_unit = value_unit

    @property
    def centers(self) -> np.ndarray:
        """Get the centers of the histogram bins."""
        return self._centers

    @property
    def values(self) -> np.ndarray:
        """Get the values of the histogram bins."""
        return self._values

    @property
    def data_unit(self) -> str:
        """Get the data unit of the histogram."""
        return self._data_unit

    @property
    def value_unit(self) -> str:
        """Get the value unit of the histogram."""
        return self._value_unit

    def with_values(
        self,
        values: np.ndarray,
        value_unit: str = ''
    ) -> Self:
        """Create a new Histogram with the given values."""
        return type(self)(
            centers=self._centers,
            values=values,
            data_unit=self._data_unit,
            value_unit=(value_unit or self._value_unit)
        )


class MultiTrialHistogram:
    """A class that represents a histogram with multiple trials.
    It contains multiple Histogram objects, each representing a trial.
    """

    def __init__(
        self,
        centers: np.ndarray,
        values: np.ndarray,
        data_unit: str = '',
        value_unit: str = ''
    ) -> None:

        centers = centers.squeeze()
        if centers.ndim != 1:
            raise ValueError("Centers must be a 1D array.")

        if centers.shape[0] != values.shape[0]:
            raise ValueError("Centers and values must have the same length.")

        self._centers = centers
        self._values = values
        self._data_unit = data_unit
        self._value_unit = value_unit

    @property
    def centers(self) -> np.ndarray:
        """Get the centers of the histogram bins."""
        return self._centers

    @property
    def values(self) -> np.ndarray:
        """Get the values of the histogram bins."""
        return self._values

    @property
    def data_unit(self) -> str:
        """Get the data unit of the histogram."""
        return self._data_unit

    @property
    def value_unit(self) -> str:
        """Get the value unit of the histogram."""
        return self._value_unit

    @property
    def n_trials(self) -> int:
        """Get the number of trials in the histogram."""
        return self._values.shape[1]

    @classmethod
    def from_list(cls, data: list[Histogram]) -> Self:
        """Create a MultiTrialHistogram from a list of dictionaries."""
        return cls(
            data[0].centers,
            np.stack([item.values for item in data], axis=1),
            data[0].data_unit,
            data[0].value_unit
        )


class HistogramCalculatorLike(Protocol):
    """Protocol for histogram calculator-like classes."""
    @abc.abstractmethod
    def __call__(self, data: np.ndarray) -> Histogram:
        """Calculate the histogram."""
    @property
    @abc.abstractmethod
    def bin_range(self) -> tuple[float, float]:
        """Get the range of the histogram bins."""
    @property
    @abc.abstractmethod
    def data_unit(self) -> str:
        """Get the data unit of the histogram."""
    @property
    @abc.abstractmethod
    def data_multiplier(self) -> float:
        """Get the data multiplier."""


class HistogramCalculator(HistogramCalculatorLike):
    """
    A class that calculates a histogram according the specified parameters.
    """

    def __init__(
        self,
        bin_range: tuple[float, float],
        bin_size: float,
        value_unit: str = 'count',
        data_unit: str = ''
    ) -> None:
        self._bin_size = bin_size
        self._range = bin_range
        self._value_unit = value_unit
        self._data_unit = data_unit

        self._edges = np.arange(
            bin_range[0],
            bin_range[1] + bin_size,
            bin_size
        )
        self._centers = np.mean(
            [self._edges[:-1], self._edges[1:]], axis=0
        )
        self._data_multiplier = 1000 if data_unit == 'ms' else 1

    @property
    def centers(self) -> np.ndarray:
        """Get the centers of the histogram bins."""
        return self._centers

    @property
    def value_unit(self) -> str:
        """Get the value unit of the histogram."""
        return self._value_unit

    @property
    def data_unit(self) -> str:
        """Get the data unit of the histogram."""
        return self._data_unit

    @property
    def bin_range(self) -> tuple[float, float]:
        """Get the range of the histogram bins."""
        return self._range

    @property
    def data_multiplier(self) -> float:
        """Get the data multiplier."""
        return self._data_multiplier

    def __call__(self, data: np.ndarray) -> Histogram:
        """Calculate the histogram."""
        data *= self._data_multiplier
        values, _ = np.histogram(data, bins=self._edges, range=self._range)
        value_unit = 'count'
        if self._value_unit == 'freq':
            values = (values / self._bin_size) * self._data_multiplier
            value_unit = 'Hz'
        elif self._value_unit == 'prob':
            if values.sum() > 0:
                values = values / values.sum()
            value_unit = 'norm. count'

        return Histogram(
            centers=self._centers,
            values=values,
            data_unit=self._data_unit,
            value_unit=value_unit
        )


class HistogramCalculatorGaussSmoother:
    """Adapter class to smooth histograms after calculation."""
    def __init__(
        self, calculator: HistogramCalculatorLike, sigma: float
    ):
        self._calculate = calculator
        self._sigma = sigma
        self._smooth = functools.partial(
            gaussian_filter1d,
            sigma=sigma,
            axis=0
        )

    @property
    def bin_range(self) -> tuple[float, float]:
        """Get the range of the histogram bins."""
        return self._calculate.bin_range

    @property
    def data_unit(self) -> str:
        """Get the data unit of the histogram."""
        return self._calculate.data_unit

    @property
    def data_multiplier(self) -> float:
        """Get the data multiplier."""
        return self._calculate.data_multiplier

    def __call__(self, data: np.ndarray) -> Histogram:
        """Calculate and smooth the histogram."""
        raw_histogram = self._calculate(data)
        smoothed_values = self._smooth(raw_histogram.values)
        return raw_histogram.with_values(smoothed_values)


SpikeTypeResults = namedtuple("SpikeTypeResults", ['spike_type', 'result'])


@dataclass
class EventAnalysisResult:
    """A class that represents the result of event analysis for a spike type."""
    histogram: MultiTrialHistogram
    trial_events: list[np.ndarray]

def calculate_ptth_for_events(
    triggers: np.ndarray,
    events: np.ndarray,
    calculator: HistogramCalculatorLike,
) -> EventAnalysisResult:
    """
    Calculate the peritrigger time-histogram for a set of triggers and events.
    """
    triggered_events = [(events - trigger) for trigger in triggers]
    trial_events = [
        te[
            (te >= (calculator.bin_range[0] / calculator.data_multiplier)) &
            (te <= (calculator.bin_range[1] / calculator.data_multiplier))
        ]
        for te in triggered_events
    ]
    histograms = [calculator(trial_event) for trial_event in trial_events]
    histogram = MultiTrialHistogram.from_list(histograms)
    return EventAnalysisResult(histogram=histogram, trial_events=trial_events)

def calculate_ptth_for_spike_types(
    spikes: pd.DataFrame,
    triggers: np.ndarray,
    spike_types: list[str],
    calculator: HistogramCalculatorLike
) -> list[SpikeTypeResults]:
    """Calculate triggered histograms around each trigger."""
    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=calculate_ptth_for_events(
                triggers,
                select_spikes(spikes, spike_type),
                calculator
            )
        ) for spike_type in spike_types
    ]

def create_per_spike_type_calculator(
    params: PeriTriggerAnalysisParams
) -> functools.partial:
    """Create a partial function to calculate per-spike-type histograms."""

    calculator = HistogramCalculator(
        params.histogram.bin_range,
        params.histogram.bin_size,
        params.histogram.value_unit,
        params.histogram.data_unit
    )
    if params.smooth_size > 0:
        calculator = HistogramCalculatorGaussSmoother(
            calculator, params.smooth_size
        )

    return functools.partial(
        calculate_ptth_for_spike_types,
        spike_types=params.spike_types,
        calculator=calculator
    )

def select_spikes(spikes: DataFrame, type_name: str) -> np.ndarray:
    """Select spikes of the given type."""
    return spikes[spikes['spike_type'] == type_name]['timestamp'].to_numpy()

