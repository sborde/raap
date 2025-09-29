"""
This script runs a full slowdown analysis on each cell of the dataset.

It iterates over the given input cells and for each cell:
 - calculates the slowdown features,
 - visualizes the distribution of the spike features
 - plots the spikes grouped by the features according to a given threshold
 - estimates and plots the SNR using the provided Savitzky-Golay filter

Versions:
   - 2025. 07. 19. - initial version
   - 2025. 07. 29. - way of starting the analysis changed
"""

import abc

from tqdm import tqdm

from workflows.plotters.spike_shapes import CategorizedSpikeShapePlotter
from services.data_managers.output import (
    create_exporter
)
from services.data_managers.input import (
    list_input_files,
    make_grouped_path
)

from workflows.plotters.base import (
    IndividualSpikeFigureSaverBuilder,
    CellwiseFigureSaver,
    create_individual_figure_saver
)
from workflows.common.models.cell import (
    CellInfo,
    CellWrapper
)
from workflows.common.models.spike import (
    SpikeShapes,
    load_spike_shapes
)

from workflows.plotters.spike_properties import (
    ResidualHistogramPlotter,
    ScatterHistogramPlotter,
    SpikePropertyPlotter,
    ValleyTimeDeviationHistogramPlotter,
    make_slowdown_debug_images
)
from workflows.plotters.spike_types import SpikeTypePiechartPlotter

from workflows.calculators.slowdown_descriptors import DescribedSpikeShape
from services.spike_description import (
    SpikeShapeCategorizer,
    SpikeShapeDescriptor,
    select_detected_spikes
)

def run_spike_shape_description(config: dict) -> None:
    """
    Runs the full spike shape description analysis on the dataset 
    based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary for the analysis.
    """

    workflow = SpikeShapeDescriptionFullWorkflow(config)

    cells = list_input_files(config)
    [
        workflow.run(cell) for cell in
        tqdm(cells, total=len(cells), desc="Describe spikes")
    ]


class SpikeShapeDescriptorWorkflow:
    """
    A class to describe spike shapes using wavelet transforms and other features.
    """

    def __init__(self, config: dict):
        self._config = config
        self._descriptor = SpikeShapeDescriptor(config['params']['descriptor'])
        self._exporter = create_exporter(config['data']['features'])
        self._figure_saver_builder = IndividualSpikeFigureSaverBuilder(
            config['data']['figures']['path'],
            config['plot']['global']['dpi']
        )
        self._save_individual = config['params']['descriptor']['single_spike_images']['do']

    def run(self, cell: CellInfo) -> CellWrapper[list[DescribedSpikeShape]]:
        """ Runs the full spike shape description analysis on a single cell."""
        cell_spikes = self._load(cell)
        cell_spikes = self._describe(cell_spikes)
        self._exporter.export(cell_spikes)
        self._plot_single_spike_features(cell_spikes)
        return cell_spikes

    def _load(self, cell: CellInfo) -> CellWrapper[SpikeShapes]:
        """ Loads the spikes for the given cell using the provided configuration."""
        path = make_grouped_path(cell, self._config['data']['raw_data'])
        spikes = load_spike_shapes(path, self._config['params']['extractor'])
        return CellWrapper(cell, spikes)

    def _describe(self, cell: CellWrapper[SpikeShapes]) -> CellWrapper[list[DescribedSpikeShape]]:
        """ Loads the spikes for the given cell using the provided configuration."""
        spike_features = self._descriptor(cell.data)
        return CellWrapper(cell.info, spike_features)

    def _plot_single_spike_features(self, cell: CellWrapper[list[DescribedSpikeShape]]):
        """Plots the features for a single spike."""
        if self._save_individual:
            figure_saver = self._figure_saver_builder.build(cell.info)
            plot_features = (
                select_detected_spikes(cell.data)
                if self._config['params']['descriptor']['single_spike_images']['show'] == 'detected'
                else cell.data
            )
            make_slowdown_debug_images(plot_features, figure_saver)


class SpikeShapeCategorizerWorkflow:
    """
    A class to categorize spike shapes using their features.
    """

    def __init__(self, config: dict):
        self._categorizer = SpikeShapeCategorizer(config['params']['categorization'])

    def run(
        self,
        cell_features: CellWrapper[list[DescribedSpikeShape]]
    ) -> CellWrapper[list[DescribedSpikeShape]]:
        """ Runs the full spike shape description analysis on a single cell."""
        categorized_features = self._categorizer(cell_features.data)
        return CellWrapper(cell_features.info, categorized_features)


class SpikeShapeFeaturePlotterWorkflow:
    """
    A class to plot spike shape features using wavelet transforms and other features.
    """

    def __init__(self, config: dict):
        self._config = config
        self._plotters = [
            decorate_plotter_with_saver(
                create_spike_property_plotter(plot_type, config['plot']),
                config['data']['figures']['path'],
                config['plot'][plot_type].get('suffix', ''),
                config['plot']['global']
            )
            for plot_type, selector_flag in config['params']['plot']['selector'].items()
            if selector_flag
        ]

    def run(self, cell: CellWrapper[list[DescribedSpikeShape]]) -> CellWrapper[list[DescribedSpikeShape]]:
        """ Runs the full spike shape description analysis on a single cell."""
        for plotter in self._plotters:
            plotter(cell)
        return cell


class SpikeShapeDescriptionFullWorkflow:
    """
    A protocol that defines the interface for a full spike shape description workflow.
    """
    def __init__(self, config: dict):
        self._descriptor = SpikeShapeDescriptorWorkflow(config)
        self._categorizer = SpikeShapeCategorizerWorkflow(config)
        self._plotter = SpikeShapeFeaturePlotterWorkflow(config)

    @abc.abstractmethod
    def run(self, cell: CellInfo) -> CellWrapper[list[DescribedSpikeShape]]:
        """ Runs the full spike shape description analysis on a single cell."""
        spike_features = self._descriptor.run(cell)
        spike_features = self._categorizer.run(spike_features)
        return self._plotter.run(spike_features)


class SavedSpikePropertyPlotter(SpikePropertyPlotter):
    """
    A class that plots the spike shape properties specified by the
    plotter object and saves the resulting figure.
    """
    def __init__(self, plotter: SpikePropertyPlotter, saver: CellwiseFigureSaver):
        self._plotter = plotter
        self._save_figure = saver
    
    def __call__(self, properties: CellWrapper[list[DescribedSpikeShape]]):
        """Describe spikes for a cell spread across multiple files."""
        cell_figure = self._plotter(properties)
        self._save_figure(cell_figure)

def decorate_plotter_with_saver(
    plotter: SpikePropertyPlotter,
    path: str,
    suffix: str,
    params: dict
) -> SavedSpikePropertyPlotter:
    """A helper function to decorate a spike property plotter with a figure saver."""
    figure_saver = CellwiseFigureSaver(
        path,
        suffix=suffix,
        make_svg=params.get('make_svg', False),
        dpi=params.get('dpi', 100)
    )
    return SavedSpikePropertyPlotter(plotter, figure_saver)

def create_spike_property_plotter(
    plot_type: str,
    params: dict
) -> SpikePropertyPlotter:
    """A factory function that creates a spike property plotter."""
    if plot_type == 'residual_histogram':
        return ResidualHistogramPlotter(params[plot_type] | {'palette': params['palette']})
    if plot_type == 'scatter':
        return ScatterHistogramPlotter(params[plot_type] | {'palette': params['palette']})
    if plot_type == 'valley_time_deviation':
        return ValleyTimeDeviationHistogramPlotter(params[plot_type] | {'palette': params['palette']})
    if plot_type == 'category_piechart':
        return SpikeTypePiechartPlotter(params[plot_type] | {'palette': params['palette']})
    if plot_type == 'category_averages':
        return CategorizedSpikeShapePlotter(params[plot_type] | {'palette': params['palette']})
    raise ValueError(f'Unknown spike property plotter type: {plot_type}')



