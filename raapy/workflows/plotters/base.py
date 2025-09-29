"""A module for general plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Figure
from pandas import DataFrame

from workflows.common.models.cell import CellInfo, CellWrapper

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


class IndividualSpikeFigureSaver:
    """Helper class to save individual spike feature figures."""
    def __init__(self, figure_dir: Path, dpi: int = 100):
        self._figure_dir = figure_dir
        self._dpi = dpi
        self._figure_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig: Figure, name: str):
        plt.savefig(self._figure_dir / name, dpi=self._dpi)
        plt.close(fig)
        del fig


class IndividualSpikeFigureSaverBuilder:
    """Helper class to build individual figure savers."""
    def __init__(self, figure_dir: str, dpi: int):
        self._figure_dir = Path(figure_dir)
        self._dpi = dpi
        self._figure_dir.mkdir(parents=True, exist_ok=True)

    def build(self, cell: CellInfo) -> IndividualSpikeFigureSaver:
        return IndividualSpikeFigureSaver(
            (self._figure_dir / cell.group / cell.name / 'individual'),
            self._dpi
        )

def create_individual_figure_saver(params: dict) -> IndividualSpikeFigureSaverBuilder:
    """Creates a figure saver for individual spike figures."""
    return IndividualSpikeFigureSaverBuilder(
        params['path'],
        params['dpi']
    )

class CellwiseFigureSaver:
    """Helper class to save cell-wise figures."""

    def __init__(
        self,
        path: str,
        suffix: str = '',
        make_svg: bool = False,
        dpi: int = 100
    ):
        self._figure_dir = Path(path) / 'figures'
        self._figure_dir.mkdir(parents=True, exist_ok=True)
        self._suffix = f'_{suffix}' if suffix else ''
        self._make_svg = make_svg
        self._dpi = dpi

    def __call__(self, figure: CellWrapper[Figure]):
        """Saves the given figure to the appropriate directory."""
        figure_dir = self._figure_dir / figure.group
        figure_dir.mkdir(parents=True, exist_ok=True)

        fig = figure.data
        figure_path = figure_dir / f'{figure.name}{self._suffix}'

        fig.savefig(
            figure_path.with_suffix('.png'),
            dpi=self._dpi,
            format='png',
        )
        if self._make_svg:
            fig.savefig(
                figure_path.with_suffix('.svg'),
                format='svg',
            )

        plt.close(fig)
        del fig

