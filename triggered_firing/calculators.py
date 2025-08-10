


from collections import namedtuple
from typing import Self

import numpy as np
import pandas as pd

SpikeTypeResults = namedtuple("SpikeTypeResults", ['spike_type', 'result'])

class Histogram:

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
        return self._centers

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def data_unit(self) -> str:
        return self._data_unit

    @property
    def value_unit(self) -> str:
        return self._value_unit


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
        return self._centers

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def data_unit(self) -> str:
        return self._data_unit

    @property
    def value_unit(self) -> str:
        return self._value_unit

    @property
    def n_trials(self) -> int:
        """Get the number of trials in the histogram."""
        return self._values.shape[1]

    @classmethod
    def from_list(cls, data: list[Histogram]) -> Self:
        """Create a MultiTrialHistogram from a list of dictionaries."""
        centers = data[0].centers
        values = np.stack([item.values for item in data], axis=1)
        centers_unit = data[0].data_unit
        value_unit = data[0].value_unit
        return cls(centers, values, centers_unit, value_unit)


class HistogramCalculator:
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
            [self._edges[:-1], self._edges[1:]],
            axis=0
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

    def __call__(self, data: np.ndarray) -> Histogram:
        data *= self._data_multiplier
        values, _ = np.histogram(
            data,
            bins=self._edges,
            range=self._range
        )
        value_unit = 'count'
        if self._value_unit == 'freq':
            values = (values / self._bin_size) * self._data_multiplier
            value_unit = 'Hz'
        elif self._value_unit == 'prob':
            values = values / values.sum()
            value_unit = 'norm. count'

        return Histogram(
            centers=self._centers,
            values=values,
            data_unit=self._data_unit,
            value_unit=value_unit
        )

def calculate_ptth(
    triggers: np.ndarray,
    events: np.ndarray,
    calculator: HistogramCalculator,
) -> MultiTrialHistogram:
    """
    Calculate the peritrigger time-histogram for a set of triggers and events.
    """
    histogram_values = [
        calculator(events - trigger).values
        for trigger in triggers
    ]
    return MultiTrialHistogram(
        centers=calculator.centers,
        values=np.stack(histogram_values, axis=1),
        data_unit=calculator.data_unit,
        value_unit=calculator.value_unit
    )

def calculate_ptth_for_events(
    triggers: np.ndarray,
    events: pd.DataFrame,
    calculator: HistogramCalculator,
) -> MultiTrialHistogram:
    """
    Calculate the peritrigger time-histogram for a set of triggers and events.
    It's an adapter for the `calculate_ptth` function.
    """
    return calculate_ptth(triggers, events['timestamp'].to_numpy(), calculator)

def calculate_ptth_by_spike_type(
    triggers: np.ndarray,
    spikes: pd.DataFrame,
    calculator: HistogramCalculator,
    params: dict
) -> list[SpikeTypeResults]:
    """
    Calculate triggered histograms around each trigger.
    """

    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=calculate_ptth_for_events(
                triggers,
                spikes[spikes['spike_type'] == spike_type],
                calculator
            )
        ) for spike_type in params['spike_types']
    ]

def calculate_ptth_for_spike_types(
    spikes: pd.DataFrame,
    triggers: np.ndarray,
    spike_types: list[str],
    calculator: HistogramCalculator
) -> list[SpikeTypeResults]:
    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=calculate_ptth_for_events(
                triggers,
                spikes[spikes['spike_type'] == spike_type],
                calculator
            )
        ) for spike_type in spike_types
    ]