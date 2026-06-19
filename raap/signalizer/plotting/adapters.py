import abc
from pathlib import Path
from typing import Protocol, Iterable

from matplotlib import pyplot as plt

from signalizer.framework.pipeline.protocols import EntityHandler
from signalizer.model.experiment.segments import SegmentMatrix
from signalizer.model.experiment.signals import SignalLike
from signalizer.model.wrappers import EntityData, WrappedData
from signalizer.plotting.primitives import VectorPairPlotter


class OnAxisPlotter[T](Protocol):
    """Plots the data of type T on the given axis."""
    @abc.abstractmethod
    def plot(self, axis: plt.Axes, data: T):
        ...


class OnFigureEntityPlotter[D](Protocol):
    """Creates a new figure for the given entity and plots on that axis."""
    def plot(self, entity: EntityData[D]) -> plt.Figure:
        """Creates a new figure for the given entity and plots on that axis."""
        ...


class SignalLikePlotter:
    """Plots the data of type SignalLike on the given axis."""
    def __init__(self, plotter: VectorPairPlotter):
        self._plotter = plotter

    def plot(self, axis: plt.Axes, signal: SignalLike):
        self._plotter.plot(axis, signal.times, signal.values)
        axis.set_xlabel(f'Time ({signal.time_unit})')
        axis.set_ylabel(f'Voltage ({signal.value_unit})')


class SuperimposedSegmentPlotter:
    """Plots all Segment in the given iterable on the same axis."""
    def __init__(self, plotter: VectorPairPlotter):
        self._plotter = SignalLikePlotter(plotter)

    def plot(self, axis: plt.Axes, segments: SegmentMatrix):
        for segment in segments:
            self._plotter.plot(axis, segment)


class AveragedSegmentPlotter:
    """Plots the average of segments of the same size in the given iterable on the same axis."""
    def __init__(self, plotter: VectorPairPlotter):
        self._plotter = SignalLikePlotter(plotter)

    def plot(self, axis: plt.Axes, segments: SegmentMatrix):
        self._plotter.plot(axis, segments.average())


class MultimodalSegmentPlotter:
    """Plots the segments in the given iterable on the same axis with different plotters."""
    def __init__(self, plotters: Iterable[OnAxisPlotter[SegmentMatrix]]):
        self._plotters = plotters

    def plot(self, axis: plt.Axes, segments: SegmentMatrix):
        for plotter in self._plotters:
            plotter.plot(axis, segments)


class WrapperPlotter[D]:
    """Plots the data of type D wrapped in a WrappedData object."""
    def __init__(self, plotter: OnAxisPlotter[D]):
        self._plotter = plotter

    def plot(self, axis: plt.Axes, wrapper: WrappedData[D]):
        self._plotter.plot(axis, wrapper.data)


class SingleOnFigureEntityPlotter[D]:
    """Creates a new figure for the given entity and plots on that axis."""
    def __init__(self, plotter: OnAxisPlotter[D]):
        self._plotter = WrapperPlotter[D](plotter)

    def plot(self, entity: EntityData[D]) -> plt.Figure:
        fig, ax = plt.subplots(1, 1)

        self._plotter.plot(ax, entity)
        ax.set_title(f'{entity.name} {entity.version}')

        return fig


class SavedSingleFigureEntityPlotter[D](EntityHandler[D]):
    """Saves the given entity on an individual figure."""

    def __init__(self, plotter: OnFigureEntityPlotter[D], path: Path, suffix: str, extension: str = 'png'):

        if path.exists() and not path.is_dir():
            raise ValueError("SavedSingleFigureEntityPlotter's argument points to an existing file.")
        if not path.exists():
            path.mkdir(parents=True)

        self._plotter = plotter
        self._path = path
        self._suffix = suffix
        self._extension = extension

    def run(self, entity: EntityData[D]) -> None:
        self.plot(entity)

    def plot(self, entity: EntityData[D]):
        fig = self._plotter.plot(entity)
        fig.savefig(self._path / f'{entity.entity.label}{self._suffix}.{self._extension}')
        plt.close(fig)