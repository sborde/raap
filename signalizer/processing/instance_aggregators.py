from typing import Sequence

import numpy as np

from signalizer.model.analysis.categorizations import CategorizationMatrix
from signalizer.model.experiment.references import TimeReferenced
from signalizer.model.experiment.segments import MatrixLike
from signalizer.model.experiment.sequences import TimeReferencedVector
from signalizer.model.wrappers import InstanceMatrix


class TopologicalHistogram:

    def __init__(self, values: np.ndarray):
        self._values = values

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def shape(self) -> Sequence[int]:
        return self._values.shape


class NormalizedAggregator[R: TimeReferencedVector]:

    """
    Normalizes the input data by dividing each value by the maximum value in the data.
    """

    def __init__(self):
        pass

    def aggregate(self, data: InstanceMatrix[R, CategorizationMatrix]) -> TopologicalHistogram:
        assignments = data.data.as_matrix()
        profile_values = np.sum(assignments, axis=0) / np.sum(assignments)
        return TopologicalHistogram(profile_values)
