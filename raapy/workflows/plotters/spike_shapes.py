
import abc
from typing import Callable, Protocol, Sequence
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from workflows.calculators.slowdown_descriptors import DescribedSpikeShape
from workflows.common.models.cell import CellWrapper
from workflows.processors.spike_transformers import create_transformer
from workflows.common.models.group import GroupInfo, GroupWrapper
from workflows.common.models.spike import (
    SpikePhase,
    SpikeShape,
    SpikeShapes,
    PlottableSpike
)
from .spike_types import get_color

class CategorizedSpikeShapePlotter:
    """A class to plot categorized spike shapes."""
    def __init__(self, params: dict):
        self._params = params
        self._average_color = None

    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Plot the categorized spike shapes."""

        self._average_color = get_color(
            self._params['average']['color'],
            properties.info,
            self._params['palette']
        )

        groups = [
            GroupWrapper[DescribedSpikeShape](
                GroupInfo(group_name),
                [spike for spike in properties.data if spike.category == group_name],
                len(properties.data)
            ) for group_name in self._params['plot_order']
        ]

        ax_size = tuple(self._params['axis_size'])
        figsize = (ax_size[0] * len(groups), ax_size[1] * 2)
        fig, axes = plt.subplots(
            2, len(groups), figsize=figsize,
            sharey='row', sharex='row'
        )

        n_spikes = sum(group.n_members for group in groups)
        for i, group in enumerate(groups):
            ax1 = axes[0, i]
            ax2 = axes[1, i]

            ax1.set_title(
                group.name +
                f' {group.n_members}/{n_spikes}' +
                f' ({100 * group.n_members / n_spikes:.2f}%)'
            )

            ax1.set_xlim(self._params['xlim'])

            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('V (mV)')
            ax2.set_xlabel('V (mV)')
            ax2.set_ylabel('dV/dt (mV/ms)')

            if group.n_members == 0:
                continue

            superimposed_axis_plotter = SingleSpikeShapeAndPhasePlotter(
                self._plot_superimposed, ax1, ax2
            )
            average_axis_plotter = SingleSpikeShapeAndPhasePlotter(
                self._plot_average, ax1, ax2
            )

            members = [self._transform(spike) for spike in group.members]

            MeanAndSuperimposedSpikeListPlotter(
                average_axis_plotter,
                superimposed_axis_plotter
            )(members)

        plt.tight_layout(pad=1.0)
        return CellWrapper[Figure](properties.info, fig)

    def _transform(self, spike: DescribedSpikeShape) -> SpikeShape:
        if self._params['mode'] == 'raw':
            return spike.spike
        return SpikeShape(
            spike.spike.times,
            spike.spike.values - spike.spike.values[spike.start_i],
            spike.spike.timestamp,
            spike.spike.dt
        )

    def _plot_superimposed(self, x: np.ndarray, y: np.ndarray, ax: Axes):
        ax.plot(x, y, **self._params['superimposed'])

    def _plot_average(self, x: np.ndarray, y: np.ndarray, ax: Axes):
        ax.plot(
            x,
            y,
            linewidth=self._params['average']['linewidth'],
            alpha=self._params['average']['alpha'],
            color=self._average_color
        )


class SpikeShapePlotter(Protocol):
    """A protocol for plotting spike shapes."""
    @abc.abstractmethod
    def __call__(self, spike: PlottableSpike):
        """Plot the given spike shape."""


class SpikeShapeListPlotter(Protocol):
    """A protocol for plotting a list of spike shapes."""
    @abc.abstractmethod
    def __call__(self, spikes: list[SpikeShape]) -> None:
        """Plot the given list of spike shapes."""


class MeanAndSuperimposedSpikeListPlotter(SpikeShapeListPlotter):
    """A class to plot the mean of a list of spike shapes."""
    def __init__(
        self,
        mean_plotter: SpikeShapePlotter,
        superimposed_plotter: SpikeShapePlotter
    ):
        self._mean_plotter = mean_plotter
        self._superimposed_plotter = superimposed_plotter

    def __call__(self, spikes: list[SpikeShape]):
        """Plot the mean of the given list of spike shapes on the provided axes."""
        for spike in spikes:
            self._superimposed_plotter(spike)
        self._mean_plotter(SpikeShape.average(spikes))


class LinePlotter(Protocol):
    """A protocol for plotting lines on a given axes."""

    def __call__(self, x: np.ndarray, y: np.ndarray, ax: Axes) -> None:
        """Plot a line on the provided axes with x and y data."""


class SingleSpikeShapeAndPhasePlotter:
    """A class to plot a single spike shape and its phase."""

    def __init__(self, plotter: LinePlotter, shape_ax: Axes, phase_ax: Axes):
        self._plotter = plotter
        self._shape_ax = shape_ax
        self._phase_ax = phase_ax

    def __call__(self, spike: PlottableSpike):
        """Plot a single spike shape and its phase on the provided axes."""
        spike_phase = spike.get_phase()
        self._plotter(spike.times * 1000, spike.values, self._shape_ax)
        self._plotter(spike_phase.v, spike_phase.dvdt / 1000, self._phase_ax)
