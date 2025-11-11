"""This module contains classes that contains analysis results."""

from dataclasses import dataclass

import numpy as np

from triggered_firing.calculators import MultiTrialHistogram
from workflows.normalized_inter_raap import Interval

@dataclass
class EventSequenceList:
    """Class that contains a list of event sequences."""
    sequences: list[np.ndarray]

@dataclass
class IntervalRelatedEvents:
    """Class that contains event sequences related to intervals."""
    interval: Interval
    sequence: np.ndarray

@dataclass
class FiringAnalysisResults:
    """Class that contains firing analysis results for a single cell."""
    histograms: MultiTrialHistogram
    sequences: EventSequenceList

@dataclass
class AnalysisResult[T]:
    """Class that contains analysis results for a single cell."""
    result: T
    attributes: dict
