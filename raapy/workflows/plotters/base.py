"""A module for general plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Figure
from pandas import DataFrame

from workflows.common.models.cell import CellWrapper

def calculate_sparse_2d_histogram(x, labels, bin_size) -> DataFrame:
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

    return DataFrame(counts).sort_values(by=['label'], ascending=True)


def save_figure(
    cell_figure: CellWrapper[Figure],
    suffix: str,
    config: dict
) -> str:
    """
    Save the figure to the specified output path.
    Args:
        cell_figure (CellWrapper[Figure]): The figure to save.
        output_path (Path): The path where the figure will be saved.
    """
    output_path = Path(config['path'])
    figure_dir = output_path / 'Figures' / cell_figure.cell_type
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = cell_figure.data
    file_name = f'{cell_figure.name}_{suffix}'
    for format in config['formats']:
        figure_path = figure_dir / f'{file_name}.{format}'
        fig.savefig(figure_path, dpi=config['dpi'])
    plt.close(fig)
    del fig  # Clean up to avoid memory issues

    return str(figure_dir / f'{file_name}.png').replace('\\', '/')
