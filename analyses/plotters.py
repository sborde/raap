
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_sparse_2d_histogram(x, labels, bin_size) -> pd.DataFrame:
    """
    Calculates a sparse 2D histogram of the given data points.
    The histogram bins are defined by the bin size in the config.
    The function returns a DataFrame with the bin centers and counts.
    """

    def generate_bins(values):
        """
        Divides a value set into bins of a given size.
        """
        value_range = max(values) - min(values)
        n_bins = int(np.ceil(value_range / bin_size)) if value_range > 0 else 1
        padding = (n_bins * bin_size - value_range) / 2
        return np.linspace(
            min(values) - padding,
            max(values) + padding,
            n_bins + 1
        )

    x1_bins = generate_bins(x[:, 0])
    x2_bins = generate_bins(x[:, 1])
    x1_centers = (x1_bins[:-1] + x1_bins[1:]) / 2
    x2_centers = (x2_bins[:-1] + x2_bins[1:]) / 2

    counts = []
    for label in np.unique(labels):
        group_data = x[labels == label, :]
        group_size = group_data.shape[0]
        for i in range(len(x1_bins) - 1):
            for j in range(len(x2_bins) - 1):
                count = np.sum(
                    (group_data[:, 0] >= x1_bins[i]) &
                    (group_data[:, 0] < x1_bins[i + 1]) &
                    (group_data[:, 1] >= x2_bins[j]) &
                    (group_data[:, 1] < x2_bins[j + 1])
                )

                if count == 0:
                    continue

                counts.append({
                    'x1': x1_centers[i],
                    'x2': x2_centers[j],
                    'count': count,
                    'ratio': count / group_size,
                    'label': label
                })

    return pd.DataFrame(counts).sort_values(by=['label'], ascending=True)


def create_debug_images(
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

