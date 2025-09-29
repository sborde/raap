"""A module for spike description related service functions."""
from pathlib import Path
from tqdm import tqdm

import numpy as np
from pandas import DataFrame

from workflows.common.models.cell import (
    CellInfo,
    CellWrapper
)
from workflows.common.models.spike import (
    SpikeShapes
)
from workflows.calculators.slowdown_descriptors import DescribedSpikeShape, calculate_rising_features

FEATURES = [
    {
        'name': 'relative_dv_drop',
        'label': 'Relative dV drop'
    },
    {
        'name': 'relative_rebound_time',
        'label': 'Relative rebound time'
    }
]


class SpikeShapeDescriptor:
    """A class that calculates defined features for spike shapes."""

    def __init__(self, config: dict):
        self._config = config

    def __call__(self, spikes: SpikeShapes) -> list[DescribedSpikeShape]:
        features = [
            calculate_rising_features(spike, self._config)
            for spike in tqdm(spikes, desc=f'Describe spikes')
        ]
        features = [f for f in features if f is not None]

        if len(features) < len(spikes):
            n_errors = len(spikes) - len(features)
            print(f"[INFO]: {n_errors} spike's start couldn't be detected.")

        return features


class SpikeShapeCategorizer:
    """A class that categorizes spikes based on their features."""

    def __init__(self, config: dict):
        self._groups = config['groups']
        self._features = config['features']

    def __call__(
        self,
        features: list[DescribedSpikeShape]
    ) -> list[DescribedSpikeShape]:

        result = []
        for spike in features:
            predicted = self._groups[0]['name']
            for group in self._groups[1:]:
                flag = True
                for feature, threshold  in zip(self._features, group['bottom-left']):
                    flag = flag and threshold < getattr(spike, feature)
                if not flag:
                    break
                predicted = group['name']
            spike.category = predicted
            result.append(spike)
        return result

def select_detected_spikes(
    features: list[DescribedSpikeShape]
) -> list[DescribedSpikeShape]:
    """ Select valid spikes based on the calculated features.
    Args:
        features (list[SpikeShapeFeatures]): The features of the cell spikes.
    Returns:
        CellWrapper[list[SpikeShapeFeatures]]: A DataFrame of valid spike features.
    """
    return [
        spike_features for spike_features in features
        if (
            spike_features.slowdown_features.relative_dv_drop > 0 or
            spike_features.slowdown_features.relative_rebound_time > 0
        )
    ]
