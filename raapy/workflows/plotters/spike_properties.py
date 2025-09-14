"""Plotting functions for spike properties."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
from matplotlib.figure import Figure

from workflows.common.models.cell import CellWrapper
from workflows.plotters.base import calculate_sparse_2d_histogram

def make_residual_histogram(
    features: CellWrapper[DataFrame],
    params: dict
) -> CellWrapper[Figure]:
    """
    Create a histogram of the residuals of the spike shapes.
    """
    noise_power = features.data['noise_power']
    raw_power = features.data['raw_power']
    noise_ratio = (noise_power / raw_power) * 100
    noise_median = np.median(noise_ratio)
    noise_q1 = np.percentile(noise_ratio, 25)
    noise_q3 = np.percentile(noise_ratio, 75)

    fig, ax = plt.subplots(figsize=tuple(params['axis_size']))
    ax.hist(
        noise_ratio,
        bins=params['bins'],
        range=params['limits'],
        color=params['color'],
        alpha=0.7
    )
    ax.set_xlabel('Residual-Raw power ratio (%)')
    ax.set_title(
        f'{features.name}\n' +
        f'Noise ratio median: {noise_median:.2f} %' +
        f'Q1-Q3: {noise_q1:.2f}-{noise_q3:.2f} %'
    )

    return CellWrapper[Figure](features.info, fig)

def make_valley_time_deviation_histogram(
    features: CellWrapper[DataFrame],
    params: dict
) -> CellWrapper[Figure]:
    """Create a histogram of the valley time deviations."""

    valley_times = features.data['valley_t'] * 1000

    fig, ax = plt.subplots(figsize=tuple(params['axis_size']))
    ax.hist(
        valley_times,
        bins=params['bins'],
        range=params['limits'],
        color=params['color'],
        alpha=0.7
    )
    ax.set_xlabel('Valley time (ms)')
    ax.set_title(f'{features.name} Valley Time Histogram')

    return CellWrapper[Figure](features.info, fig)

def make_scatter_histogram(
    features: CellWrapper[DataFrame],
    plot_features: list[dict],
    params: dict,
    threshold: tuple[float, float]
) -> CellWrapper[Figure]:
    """
    Create a scatter histogram of the spike features.
    """

    binned_features = calculate_sparse_2d_histogram(
        features.data[[f['name'] for f in plot_features]].values,
        np.zeros((features.data.shape[0],)),
        params['scatter']['bin_size']
    )

    fig_size = tuple(params['scatter']['axis_size'])
    fig, ax = plt.subplots(figsize=fig_size)
    sns.scatterplot(
        data=binned_features,
        x='x1',
        y='x2',
        size='ratio',
        color=params['palette'][features.cell_type],
        sizes=tuple(params['scatter']['point_sizes']),
        ax=ax
    )
    ax.set_xlabel(plot_features[0]['label'])
    ax.set_ylabel(plot_features[1]['label'])
    ax.set_xlim(params['scatter']['axis_limits']['x'])
    ax.set_ylim(params['scatter']['axis_limits']['y'])

    ax.hlines(
        threshold[1],
        threshold[0],
        ax.get_xlim()[1],
        color='black',
        linestyle='--'
    )
    ax.vlines(
        threshold[0],
        threshold[1],
        ax.get_ylim()[1],
        color='black',
        linestyle='--'
    )

    ax.get_legend().remove()
    ax.set_title(f'{features.name}')

    return CellWrapper[Figure](features.info, fig)


def make_slowdown_debug_images(
    results: list[dict],
    output_dir: Path,
    output_dpi: int = 72
) -> None:
    """
    Plots each individual spike with the calculated features and some
    auxiliary information about the description process.
    """
    fig_size = (12, 20)
    n_spike_on_figure = 10
    n_rows, n_cols = 10, 3

    # Sort the results by relative_dv_drop
    results = sorted(results, key=lambda x: x['relative_dv_drop'])

    def create_fig():
        """Create a new figure with the specified size."""
        return plt.subplots(n_rows, n_cols, figsize=fig_size)

    fig, axes = create_fig()
    for i, feature in enumerate(results):
        row_i = i % n_spike_on_figure

        feature = feature

        t = feature['t']
        dvt = t[:-1]
        v = feature['v']
        dv = feature['dv']

        start_i = feature['start_i']
        search_start_i = feature['search_start_i']
        search_end_i = feature['search_end_i']

        left_peak = feature['left_peak']
        right_peak = feature['right_peak']
        valley_i = feature['valley_i']

        ax1 = axes[row_i, 0]
        ax1.plot(t, v, '-', color='blue')
        ax1.plot(t[start_i], v[start_i], 'o', color='red')
        ax1.plot(
            t[search_start_i:search_end_i+1],
            v[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        # ax1.set_xlim(-0.0005, 0.0005)
        ax1.set_xticks(np.linspace(t[0], t[-1], 4))

        ax2 = axes[row_i, 1]
        ax2.plot(dvt, dv, '-', color='green')
        ax2.plot(
            dvt[search_start_i:search_end_i+1],
            dv[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        ax2.plot(t[np.argmax(dv)], dv[np.argmax(dv)], 's', color='blue')
        # ax2.set_xlim(-0.0005, 0.0005)
        ax2.set_xticks(np.linspace(t[0], t[-1], 4))


        if left_peak is not None and right_peak is not None:
            ax2.plot(t[left_peak], dv[left_peak], 'kx', markersize=5)
            ax2.plot(t[right_peak], dv[right_peak], 'kx', markersize=5)

            ax2.plot(
                t[valley_i],
                dv[valley_i],
                'v',
                color='green'
            )
            ax2.vlines(
                t[valley_i],
                dv[valley_i],
                feature['dv_min_peak'],
                linestyle='--',
                color='black'
            )
            ax2.hlines(
                feature['dv_min_peak'],
                dvt[feature['rebound_i'][0]],
                dvt[feature['rebound_i'][1]],
                linestyle='--',
                color='black'
            )

        ax2.set_title(
            f't={feature["timestamp"]:.3f}s ' +
            f'rdv={feature["relative_dv_drop"]:.3f} ' +
            f'rrt={feature["relative_rebound_time"]:.3f}'
        )

        ax3 = axes[row_i, 2]
        ax3.plot(v[:-1], dv, '-', color='blue')
        ax3.plot(
            v[search_start_i:search_end_i+1],
            dv[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        ax3.plot(
            v[np.argmax(dv)],
            dv[np.argmax(dv)],
            's',
            color='blue'
        )
        if left_peak is not None and right_peak is not None:
            ax3.plot(v[left_peak], dv[left_peak], 'kx', markersize=5)
            ax3.plot(v[right_peak], dv[right_peak], 'kx', markersize=5)
            ax3.plot(v[valley_i], dv[valley_i], 'v', color='green')

        if (i+1) % n_spike_on_figure == 0 or i == len(results) - 1:
            plt.tight_layout(pad=1.0)
            plt.savefig(
                output_dir / f'spike_shape_{i // n_spike_on_figure + 1}.png',
                dpi=output_dpi
            )
            plt.close(fig)
            del fig
            if i < len(results) - 1:
                fig, axes = create_fig()
