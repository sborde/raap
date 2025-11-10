
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from common.data_utils import CellInfo, CellWrapper
from triggered_firing.calculators import SpikeTypeResults


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

def create_spike_palette_for_celltype(
    cell_type: str,
    coloring: dict,
    palette: dict
) -> dict[str, str]:
    """
    Create a color palette for spike types.

    Args:
        info (CellInfo): CellInfo for the current cell
        palette (dict): The color palette.
        coloring_mode (dict): coloring mode for each spike type for the current cell
    Returns:
        dict[str, str]: A dictionary mapping spike types to colors.
    """
    spike_color_map = {}
    for spike_type, color_mode in coloring.items():
        if color_mode == 'spike_type':
            spike_color_map[spike_type] = palette['spike_type'][spike_type]
        elif color_mode == 'cell_type':
            spike_color_map[spike_type] = palette['cell_type'][cell_type]
    return spike_color_map


def make_ptth_figure(
    cell_results: CellWrapper[list[SpikeTypeResults]],
    spike_palette: dict[str, str],
    trigger: str,
    line_width: float,
    raster_tick_size: float = 0.5
) -> CellWrapper[Figure]:
    """
    Create a Peri-Event Time Histogram (PETH) figure.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10),
        sharex=True, constrained_layout=True
    )
    raster_ax = axes[0]
    hist_ax = axes[1]
    for result in cell_results.data:
        spike_type = result.spike_type
        events = result.result.trial_events
        histogram = result.result.histogram
        spike_color = spike_palette[spike_type]
        for i, trial_events in enumerate(events):
            raster_ax.vlines(
                trial_events,
                i - raster_tick_size/2,
                i + raster_tick_size/2,
                color=spike_color
            )
        hist_ax.plot(
            histogram.centers,
            histogram.values.mean(axis=1),
            label=spike_type,
            color=spike_color,
            linewidth=line_width
        )
    hist_ax.set_ylabel(make_ylabel(cell_results.data[0].result.histogram.value_unit))
    hist_ax.set_xlabel(f"Time relative to trigger ({cell_results.data[0].result.histogram.data_unit})")
    plt.suptitle(f"{cell_results.info.name} {trigger}-triggered Histogram")
    return CellWrapper(cell_results.info, fig)

def save_figure(
    cell_figure: CellWrapper[Figure],
    path: str,
    suffix: str
) -> None:
    """
    Save the figure to a file.
    """

    info, fig = cell_figure.info, cell_figure.data

    figure_path = (
        Path(path) /
        'figures' /
        info.cell_type
    )
    figure_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(figure_path / f'{info.name}_{suffix}_histogram.png')
    plt.close(fig)


def create_ptth_plotter(
    trigger: str, params: dict
) -> Callable[
    [CellWrapper[list[SpikeTypeResults]], dict], CellWrapper[Figure]
]:
    """
    Create a function to plot PETH figures for a specific cell.
    """
    def plot(
        cell_result: CellWrapper[list[SpikeTypeResults]],
        spike_palette: dict
    ) -> CellWrapper[Figure]:
        """Creates a PETH plot for the given cell result."""
        cell_figure = make_ptth_figure(
            cell_results=cell_result,
            spike_palette=spike_palette,
            trigger=trigger,
            line_width=params['line_width'],
            raster_tick_size=params.get('raster_tick_size', 0.5)
        )
        return cell_figure
    return plot

def create_figure_saver(path: str, suffix: str) -> Callable[[CellWrapper[Figure]], None]:
    """
    Create a function to save figures with a specific path and suffix.
    """
    def save(cell_figure: CellWrapper[Figure]) -> None:
        save_figure(cell_figure, path, suffix)
    return save
