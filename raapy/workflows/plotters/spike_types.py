"""Module for functions to create spike-type related plots."""

import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from workflows.calculators.slowdown_descriptors import DescribedSpikeShape
from workflows.plotters.spike_properties import SpikePropertyPlotter
from workflows.common.models.cell import CellInfo, CellWrapper

class SpikeTypePiechartPlotter(SpikePropertyPlotter):
    """Plotter for creating pie charts of spike types."""

    def __init__(self, params: dict) -> None:
        self._params = params

    def __call__(
        self,
        properties: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[Figure]:
        """Create a pie chart of the spike types."""
        styles = self._params['styles']

        counts = {style['group']: 0 for style in styles}
        for spike in properties.data:
            spike_category = spike.category
            if spike_category not in counts:
                raise ValueError(f'Unknown spike category: {spike_category}')
            counts[spike_category] += 1

        cell_info = properties.info
        palette = self._params.get('palette', {})
        labels = [style['group'] for style in styles]
        values = [counts[style['group']] for style in styles]
        fill_colors = [
            get_color(style['fill_color'], cell_info, palette) for style in styles
        ]
        edge_colors = [
            get_color(style['edge_color'], cell_info, palette) for style in styles
        ]
        hatches = [style['pattern'] for style in styles]

        fig, ax = plt.subplots(figsize=tuple(self._params['axis_size']))
        wedges, _, _ = ax.pie( # type: ignore
            values,
            labels=labels,
            autopct='%1.2f%%',
            startangle=90,
            colors=fill_colors,
            wedgeprops=dict(linewidth=1.5),
        )

        for i, wedge in enumerate(wedges):
            if hatches[i]:
                wedge.set_hatch(hatches[i])
            wedge.set_edgecolor(edge_colors[i])

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        return CellWrapper[Figure](properties.info, fig)

def get_color(definition: str, info: CellInfo, palette: dict) -> str:
    """Get the color for a given spike type definition."""
    if definition.startswith('#'):
        return definition
    if definition == 'cell_type' or definition == 'spike_type':
        return palette[info.group]
    raise ValueError(f'Unknown color definition: {definition}')