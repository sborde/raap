"""Module for functions to create spike-type related plots."""

import matplotlib.pyplot as plt

from pandas import DataFrame
from matplotlib.figure import Figure

from workflows.common.models.cell import CellWrapper

def make_spike_type_piechart(
    features: CellWrapper[DataFrame],
    config: dict
) -> CellWrapper[Figure]:
    """
    Create a pie chart of the spike types.
    """
    group_counts = features.data['spike_type'].value_counts()
    labels = group_counts.index.tolist()
    sizes = group_counts.values.tolist()
    fillcolors = [config['fill_color'][label] for label in labels]
    edgecolors = [config['edge_color'][label] for label in labels]
    hatches = [config['hatch'][label] for label in labels]

    fig, ax = plt.subplots(figsize=tuple(config['axis_size']))
    # Matplotlib's pie chart does not support setting hatch colors directly.
    # Hatches will use the edgecolor of each wedge.
    wedges, _, _ = ax.pie( # type: ignore
        sizes,
        labels=labels,
        autopct='%1.2f%%',
        startangle=90,
        colors=fillcolors,
        wedgeprops=dict(linewidth=1.5),
    )

    # Apply hatches and edgecolors manually
    for i, wedge in enumerate(wedges):
        if hatches[i]:
            wedge.set_hatch(hatches[i])
        wedge.set_edgecolor(edgecolors[i])

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return CellWrapper[Figure](features.info, fig)
