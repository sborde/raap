
import abc
from collections import namedtuple
from typing import Optional, Protocol
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from common.data_utils import GroupWrapper
from data.spikes import SpikePhase, SpikeShape

class PlottableSpike(Protocol):
    """A protocol for objects that can be plotted as spikes."""
    
    @property
    @abc.abstractmethod
    def times(self) -> np.ndarray:
        """Get the time points of the spike."""

    @property
    @abc.abstractmethod
    def values(self) -> np.ndarray:
        """Get the values of the spike."""
    
    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Get the time step of the spike."""

    @abc.abstractmethod
    def get_phase(self) -> SpikePhase:
        """Get the phase representation of the spike."""


def plot_grouped_spikes(
    groups: list[GroupWrapper[SpikeShape]],
    config: dict
) -> Figure:
    """Plot the average spike shape and its phase categorized by spike type."""

    figsize = (config['axis_size'][0] * len(groups), config['axis_size'][1] * 2)
    fig, axes = plt.subplots(
        2, len(groups), figsize=figsize,
        sharey='row', sharex='row'
    )

    def superimposed_plotter(ax, x, y):
        ax.plot(x, y, **config['superimposed'])
    def average_plotter(ax, x, y):
        ax.plot(x, y, **config['average'])

    n_spikes = sum(group.n_members for group in groups)
    for i, group in enumerate(groups):
        ax1 = axes[0, i]
        ax2 = axes[1, i]

        ax1.set_title(
            group.name +
            f' {group.n_members}/{n_spikes}' +
            f' ({100 * group.n_members / n_spikes:.2f}%)'
        )

        ax1.set_xlim(config['xlim'])

        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('V (mV)')
        ax2.set_xlabel('V (mV)')
        ax2.set_ylabel('dV/dt (mV/ms)')

        if group.n_members == 0:
            continue

        members = [config['transformer'](spike) for spike in group.members]

        for spike in members:
            plot_single_spike(
                spike,
                ax1,
                ax2,
                superimposed_plotter
            )

        plot_single_spike(
            SpikeShape.average(members),
            ax1,
            ax2,
            average_plotter
        )

    plt.tight_layout(pad=1.0)
    return fig


class LinePlotter(Protocol):
    """A protocol for plotting lines on a given axes."""

    def __call__(self, ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
        """Plot a line on the provided axes with x and y data."""


def plot_single_spike(
    spike: PlottableSpike,
    shape_ax: Axes,
    phase_ax: Axes,
    plotter: LinePlotter
):
    """Plot a single spike shape and its phase on the provided axes."""
    spike_phase = spike.get_phase()
    plotter(shape_ax, spike.times * 1000, spike.values)
    plotter(phase_ax, spike_phase.v, spike_phase.dvdt / 1000)
