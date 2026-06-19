import abc
from typing import Iterable, TypeVar, Protocol

import h5py
import numpy as np

from signalizer.io.protocols import HDFExportable
from signalizer.model.experiment.signals import Signal, SignalMatrix
from signalizer.model.measurements import VoltageUnit, TimeUnit
from signalizer.model.temporal.data import TimeInterval
from signalizer.model.temporal.sequences import TimeIntervalSequence


class MatrixLike(Protocol):
    """Protocol for objects that can be converted to a matrix."""

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray: ...

    @classmethod
    @abc.abstractmethod
    def concat[I](cls, segment_matrices: list[I]) -> I: ...


class Segment(HDFExportable):
    """Represents a segment of a signal."""
    def __init__(self, signal: Signal):
        self._signal = signal

    def __str__(self):
        return f'segment ({self._signal.duration} {self._signal.time_unit})'

    @property
    def times(self) -> np.ndarray: return self._signal.times

    @property
    def values(self) -> np.ndarray: return self._signal.values

    @property
    def sample_count(self) -> int: return self._signal.sample_count

    @property
    def time_unit(self) -> TimeUnit: return self._signal.time_unit

    @property
    def value_unit(self) -> VoltageUnit: return self._signal.value_unit

    @property
    def signal(self) -> Signal: return self._signal

    @property
    def si(self) -> float: return self._signal.si

    @property
    def fs(self) -> int: return self._signal.fs

    def save(self, group: h5py.Group): self._signal.save(group.create_group('signal'))

    @classmethod
    def load(cls, group: h5py.Group): return Segment(Signal.load(group['signal']))


class IntervalSegment(Segment):
    """Represents a segment of a signal with an associated time interval."""

    def __init__(self, signal: Signal, interval: TimeInterval):
        super().__init__(signal)
        self._interval = interval

    @property
    def interval(self) -> TimeInterval: return self._interval

    def save(self, group: h5py.Group):
        super().save(group)
        self._interval.save(group.create_group('interval'))

    @classmethod
    def load(cls, group: h5py.Group):
        segment = super().load(group)
        interval = TimeInterval.load(group['interval'])
        return IntervalSegment(segment.signal, interval)


T = TypeVar('T', bound=Segment)


class SegmentList[T](HDFExportable):
    """Represents a collection of segments with the same sampling frequency."""

    def __init__(self, segments: Iterable[T]):

        if len(set([segment.fs for segment in segments])) != 1:
            raise ValueError('All segments must have the same sampling frequency.')

        self._segments = list(segments)

    def __len__(self) -> int: return len(self._segments)

    def __getitem__(self, item) -> T: return self._segments[item]

    def __delitem__(self, key): del self._segments[key]

    def __iter__(self): return iter(self._segments)

    @property
    def si(self) -> float: return self._segments[0].si

    @property
    def fs(self) -> int: return self._segments[0].fs

    def save(self, group: h5py.Group):
        for i, segment in enumerate(self):
            segment.save(group.create_group(f'segment_{i}'))

    @classmethod
    def load(cls, group: h5py.Group):
        segments = []
        for key in group.keys():
            segments.append(Segment.load(group[key]))
        return SegmentList(segments)


class SegmentMatrix(HDFExportable, MatrixLike):
    """A collection of segments with the same size."""

    def __init__(self, signals: SignalMatrix):
        self._signals = signals

    def __repr__(self): return f'SegmentMatrix({self._signals})'

    def __str__(self): return f'{len(self)} segments of {len(self.times)} samples'

    def __len__(self) -> int: return len(self._signals)

    def __getitem__(self, item) -> Segment: return Segment(self._signals[item])

    def __delitem__(self, key): del self._signals[key]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def times(self) -> np.ndarray: return self._signals.times

    @property
    def signals(self) -> SignalMatrix: return self._signals

    @property
    def time_unit(self) -> TimeUnit: return self._signals.time_unit

    @property
    def value_unit(self) -> VoltageUnit: return self._signals.value_unit

    @property
    def si(self) -> float: return self._signals.si

    @property
    def fs(self) -> int: return self._signals.fs

    def as_matrix(self) -> np.ndarray: return self._signals.as_matrix()

    def average(self) -> Segment:
        """Returns the average of the segments."""
        return Segment(self._signals.average())

    def save(self, group: h5py.Group):
        segment_group = group.create_group('segments')
        self._signals.save(segment_group.create_group('signals'))

    @classmethod
    def load(cls, group: h5py.Group):
        signals = SignalMatrix.load(group['segments']['signals'])
        return SegmentMatrix(signals)

    @classmethod
    def concat(cls, segment_matrices: list['SegmentMatrix']) -> 'SegmentMatrix':
        """Concatenates multiple SegmentMatrices into one."""
        signals = [segment_matrix.signals for segment_matrix in segment_matrices]
        return cls(SignalMatrix.concat(signals))


class IntervalSegmentMatrix(SegmentMatrix):
    def __init__(self, signals: SignalMatrix, intervals: TimeIntervalSequence):
        super().__init__(signals)

        if len(signals) != len(intervals):
            raise ValueError('The number of signals and intervals must be the same.')

        self._intervals = intervals

    def __repr__(self):
        return f'IntervalSegmentMatrix({self._signals}, {self._intervals})'

    def __str__(self):
        return f'collection of {len(self)} segments of {len(self.times)} samples with intervals'

    def __getitem__(self, item) -> IntervalSegment:
        return IntervalSegment(super().__getitem__(item).signal, self._intervals[item])

    def __delitem__(self, key):
        super().__delitem__(key)
        del self._intervals[key]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def save(self, group: h5py.Group):
        super().save(group)
        self._intervals.save(group.create_group('intervals'))

    @classmethod
    def load(cls, group: h5py.Group):
        signals = SignalMatrix.load(group['signals'])
        interval = TimeIntervalSequence.load(group['intervals'])
        return IntervalSegmentMatrix(signals, interval)
