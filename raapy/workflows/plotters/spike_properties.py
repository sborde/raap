"""Plotting functions for spike properties."""

import abc

from typing import Optional, Protocol

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
from matplotlib.figure import Figure

from workflows.calculators.slowdown_descriptors import DescribedSpikeShape
from workflows.plotters.base import IndividualSpikeFigureSaver
from workflows.common.models.cell import CellWrapper
from workflows.plotters.base import calculate_sparse_2d_histogram


class SpikePropertyPlotter(Protocol):
    """A protocol for classes that plot spike shape properties."""
    @abc.abstractmethod
    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Describe spikes for a cell spread across multiple files."""


class ResidualHistogramPlotter:
    """Create a histogram of the residuals of the spike shapes."""
    def __init__(self, params: dict):
        self._params = params

    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Create a histogram of the residuals of the spike shapes."""

        noise_power = np.array([
            f.noise_properties.noise for f in properties.data
        ])
        raw_power = np.array([
            f.noise_properties.raw for f in properties.data
        ])
        noise_ratio = (noise_power / raw_power) * 100

        fig, ax = plt.subplots(figsize=tuple(self._params['axis_size']))
        ax.hist(
            noise_ratio,
            bins=self._params['bins'],
            range=self._params['limits'],
            color=self._params['color'],
            alpha=self._params['alpha']
        )
        ax.set_xlabel('Residual/Raw power ratio (%)')

        median = np.median(noise_ratio)
        q1 = np.percentile(noise_ratio, 25)
        q3 = np.percentile(noise_ratio, 75)
        ax.set_title(
            f'{properties.name}\n' +
            f'Median residual/raw power ratio: {median:.2f}%' +
            f'Q1-Q3: {q1:.2f}-{q3:.2f}%'
        )
        return properties.with_data(fig)


class ScatterHistogramPlotter:
    """Create a scatter histogram of the spike features."""
    def __init__(self, params: dict):
        self._params = params

    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Create a scatter histogram of the spike features."""

        binned_features = calculate_sparse_2d_histogram(
            np.stack([
                np.array([
                    f.slowdown_features.relative_dv_drop,
                    f.slowdown_features.relative_rebound_time
                ]) for f in properties.data
            ]),
            np.zeros((len(properties.data),)),
            self._params['histogram']['bin_size']
        )

        fig, ax = plt.subplots(figsize=self._params['axis_size'])
        sns.scatterplot(
            data=binned_features,
            x='x1',
            y='x2',
            size='ratio',
            color=self._params['palette'][properties.group],
            sizes=tuple(self._params['point_sizes']),
            ax=ax
        )
        ax.hlines(
            self._params['threshold'][1],
            self._params['threshold'][0],
            ax.get_xlim()[1],
            color='black',
            linestyle='--'
        )
        ax.vlines(
            self._params['threshold'][0],
            self._params['threshold'][1],
            ax.get_ylim()[1],
            color='black',
            linestyle='--'
        )

        ax.set_xlabel(self._params['axis']['x']['label'])
        ax.set_ylabel(self._params['axis']['y']['label'])
        ax.set_xlim(self._params['axis']['x']['limits'])
        ax.set_ylim(self._params['axis']['y']['limits'])
        ax.get_legend().remove()
        ax.set_title(properties.name)
        return properties.with_data(fig)


class ValleyTimeDeviationHistogramPlotter:
    """Create a histogram of the valley time deviations."""
    def __init__(self, params: dict):
        self._params = params

    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Create a histogram of the valley time deviations."""
        valley_times = [
            f.valley_time * 1000 for f in properties.data
        ]

        fig, ax = plt.subplots(figsize=tuple(self._params['axis_size']))
        ax.hist(
            valley_times,
            bins=self._params['bins'],
            range=self._params['limits'],
            color=self._params['color'],
            alpha=self._params['alpha']
        )
        ax.set_xlabel(self._params['axis']['x']['label'])
        ax.set_title(properties.name)

        return CellWrapper[Figure](properties.info, fig)

def make_slowdown_debug_images(
    results: list[DescribedSpikeShape],
    figure_saver: IndividualSpikeFigureSaver
) -> None:
    """
    Plots each individual spike with the calculated features and some
    auxiliary information about the description process.
    """
    fig_size = (12, 20)
    n_spike_on_figure = 10
    n_rows, n_cols = 10, 3

    # Sort the results by relative_dv_drop
    results = sorted(results, key=lambda x: x.slowdown_features.relative_dv_drop)

    def create_fig():
        """Create a new figure with the specified size."""
        return plt.subplots(n_rows, n_cols, figsize=fig_size)

    fig, axes = create_fig()
    for i, feature in enumerate(results):
        row_i = i % n_spike_on_figure

        plot_spike(feature, axes[row_i, :])

        if (i+1) % n_spike_on_figure == 0 or i == len(results) - 1:
            plt.tight_layout(pad=1.0)
            figure_saver(fig, f'spike_shape_{i // n_spike_on_figure + 1}.png')

            if i < len(results) - 1:
                fig, axes = create_fig()

def plot_spike(features: DescribedSpikeShape, axes: list[Axes]) -> None:
    """Plot a single spike shape on the given axes."""
    plot_spike_shape_with_points(features, axes[0])
    plot_spike_derivative_with_features(features, axes[1])
    plot_spike_phase_with_features(features, axes[2])

def plot_spike_shape_with_points(
    features: DescribedSpikeShape,
    ax: Axes
):
    """Plot a single spike shape with its features on the given axis."""
    t = features.spike.times
    v = features.spike.values

    ax.plot(t, v, '-', color='blue')
    ax.plot(t[features.start_i], v[features.start_i], 'o', color='red')
    ax.set_xticks(np.linspace(t[0], t[-1], 4))

    search_interval = features.search_interval
    if search_interval is not None:
        ax.plot(
            t[search_interval.left_i:search_interval.right_i+1],
            v[search_interval.left_i:search_interval.right_i+1],
            '--',
            color='red'
        )

def plot_spike_derivative_with_features(
    feature: DescribedSpikeShape,
    ax: Axes
):
    """Plot the derivative of a single spike shape with its features on the given axis."""
    derivative = feature.spike.get_derivative()
    dvt = derivative.times
    dv = derivative.dv

    ax.plot(dvt, dv, '-', color='green')
    ax.plot(dvt[np.argmax(dv)], dv[np.argmax(dv)], 's', color='blue')
    ax.set_xticks(np.linspace(dvt[0], dvt[-1], 4))
    if feature.search_interval is not None:
        ax.plot(
            dvt[feature.search_interval.left_i:feature.search_interval.right_i+1],
            dv[feature.search_interval.left_i:feature.search_interval.right_i+1],
            '--',
            color='red'
        )

    if feature.slowdown_features is None:
        return
    slowdown_features = feature.slowdown_features
    if slowdown_features.dv_min_peak is None:
        return

    left_peak = slowdown_features.left_peak
    right_peak = slowdown_features.right_peak
    valley_i = slowdown_features.valley_i

    ax.plot(dvt[left_peak], dv[left_peak], 'kx', markersize=5)
    ax.plot(dvt[right_peak], dv[right_peak], 'kx', markersize=5)

    ax.plot(dvt[valley_i], dv[valley_i], 'v', color='green')

    ax.vlines(
        dvt[valley_i],
        dv[valley_i],
        slowdown_features.dv_min_peak,
        linestyle='--',
        color='black'
    )
    ax.hlines(
        slowdown_features.dv_min_peak,
        dvt[slowdown_features.rebound_left_i],
        dvt[slowdown_features.rebound_right_i],
        linestyle='--',
        color='black'
    )
    ax.set_title(
        f't={feature.spike.timestamp:.3f}s ' +
        f'rdv={slowdown_features.relative_dv_drop:.3f} ' +
        f'rrt={slowdown_features.relative_rebound_time:.3f}'
    )

def plot_spike_phase_with_features(
    feature: DescribedSpikeShape,
    ax: Axes
):
    """Plot the phase of a single spike shape with its features on the given axis."""
    v = feature.spike.get_phase().v
    dvdt = feature.spike.get_phase().dvdt

    ax.plot(v, dvdt, '-', color='blue')
    ax.plot(
        v[feature.search_interval.left_i:feature.search_interval.right_i+1],
        dvdt[feature.search_interval.left_i:feature.search_interval.right_i+1],
        '--',
        color='red'
    )
    ax.plot(
        v[np.argmax(dvdt)],
        dvdt[np.argmax(dvdt)],
        's',
        color='blue'
    )

    if feature.slowdown_features is None:
        return
    slowdown_features = feature.slowdown_features

    left_peak = slowdown_features.left_peak
    right_peak = slowdown_features.right_peak
    valley_i = slowdown_features.valley_i

    if left_peak is not None and right_peak is not None:
        ax.plot(v[left_peak], dvdt[left_peak], 'kx', markersize=5)
        ax.plot(v[right_peak], dvdt[right_peak], 'kx', markersize=5)
        ax.plot(v[valley_i], dvdt[valley_i], 'v', color='green')
