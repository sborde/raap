"""Module for plot related classes and functions."""


from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from common.data_utils import CellWrapper
from phase_event_coupling.calculators import SpikeTypeResults

class SpikeTypePaletteBuilder:
    """
    A builder class that creates a color palette
    for spike types for a given cell type.
    """

    def __init__(self, palette: dict, coloring: dict):
        self._palette = palette
        self._coloring = coloring

    def __call__(self, cell_type: str) -> dict[str, str]:
        """
        Create a color palette for spike types.

        Args:
            info (CellInfo): CellInfo for the current cell
        Returns:
            dict[str, str]: A dictionary mapping spike types to colors for the current cell type.
        """
        spike_colors = {}
        for spike_type, color_mode in self._coloring.items():
            if color_mode == 'spike_type':
                reference_value = spike_type
            elif color_mode == 'cell_type':
                reference_value = cell_type
            base_palette = self._palette[color_mode]
            spike_colors[spike_type] = base_palette[reference_value]
        return spike_colors

def make_ylabel(value_unit: str) -> str:
    """
    Create a y-axis label based on the value type.
    """
    if value_unit == 'Hz':
        text = "Firing Frequency (Hz)"
    elif value_unit == 'norm. count':
        text = "Distribution (norm. count)"
    else:
        text = "Count"
    return "Mean spike " + text

def make_pec_figure(
    cell_results: CellWrapper[list[SpikeTypeResults]],
    spike_palette: dict[str, str],
    line_width: float
) -> CellWrapper[Figure]:
    """
    Create a Phase-Event Coupling Histogram (PEC) figure.
    """
    info = cell_results.info
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in cell_results.data:
        spike_type = result.spike_type
        histogram = result.result
        ax.plot(
            np.concatenate([histogram.centers, histogram.centers + histogram.centers[-1]]),
            np.concatenate([histogram.values.mean(axis=1)] * 2),
            label=spike_type,
            color=spike_palette[spike_type],
            linewidth=line_width
        )
    ax.set_ylabel(make_ylabel(cell_results.data[0].result.value_unit))
    ax.set_xlabel(f"Cycle phase ({cell_results.data[0].result.data_unit})")
    return CellWrapper(cell_results.info, fig)

def create_pec_plotter(
    params: dict
) -> Callable[
    [CellWrapper[list[SpikeTypeResults]], dict], CellWrapper[Figure]
]:
    """
    Create a function to plot PEC figures for a specific cell.
    """
    def plot(
        cell_result: CellWrapper[list[SpikeTypeResults]],
        spike_palette: dict
    ) -> CellWrapper[Figure]:
        """Creates a PEC plot for the given cell result."""
        cell_figure = make_pec_figure(
            cell_results=cell_result,
            spike_palette=spike_palette,
            line_width=params['line_width']
        )
        return cell_figure
    return plot
