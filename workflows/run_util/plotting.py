
from typing import Optional
import numpy as np
import seaborn as sns

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

class CategoricalDistribution:
    """Represents a categorical distribution with named categories."""
    def __init__(self, values: np.ndarray, names: list[str], category_name: str):
        self.values = values
        self.names = names
        self.category_name = category_name


SpikeDistribution = tuple[str, CategoricalDistribution]

def create_stacked_area_chart(
    stats: list[SpikeDistribution],
    ax: Axes,
    params: dict
):

    for i, (_, distribution) in enumerate(stats):
        _make_bar(
            i + 1,
            np.cumsum(distribution.values),
            ax,
            distribution.names,
            params['palette'],
            barwidth=0.5
        )
    plt.xticks(np.arange(len(stats)) + 1, [s[0] for s in stats])

    # Get unique handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handle_dict = {}
    for handle, label in zip(handles, labels):
        if label not in handle_dict:
            handle_dict[label] = handle

    unique_handles = list(handle_dict.values())
    unique_labels = list(handle_dict.keys())

    ax.legend(
        unique_handles,
        unique_labels,
        title=distribution.category_name,
        loc='upper left',
        bbox_to_anchor=(1.0, 1),
        borderaxespad=0.
    )

def _make_bar(
    x: int,
    y: np.ndarray,
    ax: Axes,
    labels: list[str],
    palette: Optional[list[str]] = None,
    barwidth: float = 0.5
):
    """
    Creates a stacked bar from a cumulative distribution at the given x
    coordinate. Labels and colors can be passed through arguments.
    """

    if palette is None:
        palette = sns.color_palette("Set2", n_colors=len(labels)).as_hex()

    bar_edges = np.concatenate(([0], y))
    for i, category in enumerate(labels):
        ax.fill_between(
            [x - barwidth / 2, x + barwidth / 2],
            bar_edges[i],
            bar_edges[i + 1],
            color=palette[i],
            label=category
        )
