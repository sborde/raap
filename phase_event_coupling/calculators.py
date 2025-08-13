


from collections import namedtuple
from typing import Optional, Self

import numpy as np
import pandas as pd

from triggered_firing.calculators import SpikeTypeResults

class NormalizedHistogram:

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


class MultiTrialNormalizedHistogram:
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
    def from_list(cls, data: list[NormalizedHistogram]) -> Self:
        """Create a MultiTrialHistogram from a list of dictionaries."""
        centers = data[0].centers
        values = np.stack([item.values for item in data], axis=1)
        centers_unit = data[0].data_unit
        value_unit = data[0].value_unit
        return cls(centers, values, centers_unit, value_unit)

FloatRange = tuple[float, float]

class NormalizedHistogramCalculator:
    """
    A class that calculates a histogram according the specified parameters.
    """

    def __init__(
        self,
        n_bins: int,
        target_range:  FloatRange = (0, 1),
        value_unit: str = 'count',
        data_unit: str = 'norm'
    ) -> None:
        self._n_bins = n_bins
        self._value_unit = value_unit
        self._data_unit = data_unit

        self._target_range = target_range
        self._target_range_size = target_range[1] - target_range[0]
        self._edges = np.linspace(target_range[0], target_range[1], n_bins + 1)
        self._centers = np.mean(
            [self._edges[:-1], self._edges[1:]],
            axis=0
        )

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

    def __call__(
        self,
        data: np.ndarray,
        norm_range: Optional[FloatRange] = None
    ) -> NormalizedHistogram:



        # if data.size == 0:
        #     return NormalizedHistogram(
        #         centers=self._centers,
        #         values=np.zeros_like(self._centers),
        #         data_unit=self._data_unit,
        #         value_unit=value_unit
        #     )

        if norm_range is None:
            norm_range = (data.min(), data.max())

        bin_size = (norm_range[1] - norm_range[0]) / self._n_bins
        data_range = norm_range[1] - norm_range[0]
        norm_data = (data - norm_range[0]) / data_range if data_range > 0 else 0
        norm_data *= self._target_range_size + self._target_range[0]

        values, _ = np.histogram(
            norm_data,
            bins=self._edges
        )
        if self._value_unit == 'freq':
            values = values / bin_size
        elif self._value_unit == 'prob':
            if values.sum() > 0:
                values = values / values.sum()

        value_unit = 'count'
        if self._value_unit == 'freq':
            value_unit = 'Hz'
        elif self._value_unit == 'prob':
            value_unit = 'norm. count'

        return NormalizedHistogram(
            centers=self._centers,
            values=values,
            data_unit=self._data_unit,
            value_unit=value_unit
        )

def calculate_pec(
    cycles: np.ndarray,
    events: np.ndarray,
    calculator: NormalizedHistogramCalculator,
) -> MultiTrialNormalizedHistogram:
    """
    Calculate the normalized cycle phase histogram for a given
    oscillation episode, dividing the events between the cycle markers.
    """
    histograms = []
    for cycle in cycles:
        cycle_events = events[(events >= cycle[0]) & (events <= cycle[1])]
        histograms.append(calculator(cycle_events, cycle))

    histogram_values = [hist.values for hist in histograms]
    return MultiTrialNormalizedHistogram(
        centers=histograms[0].centers,
        values=np.stack(histogram_values, axis=1),
        data_unit=histograms[0].data_unit,
        value_unit=histograms[0].value_unit
    )

def calculate_pec_for_spike_types(
    spikes: pd.DataFrame,
    cycles: np.ndarray,
    spike_types: list[str],
    calculator: NormalizedHistogramCalculator
) -> list[SpikeTypeResults]:
    spikes_by_type = [
        (spike_type, spikes[spikes['spike_type'] == spike_type])
        for spike_type in spike_types
    ]
    return [
        SpikeTypeResults(
            spike_type=spike_type,
            result=calculate_pec(
                cycles,
                spikes_of_type['timestamp'].to_numpy(),
                calculator
            )
        ) for spike_type, spikes_of_type in spikes_by_type
    ]