

import abc
import sys
from pathlib import Path
from typing import Protocol

# pylint: disable=wrong-import-position
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from signalizer.model.experiment.recordings import MatRecording
from data.spike.descriptors import (
    RisingFeatureCalculator,
    SpikeCategorizer,
    SpikeIntervalAssignerFactory
)
from data.spikes import SpikeShapeLoader
from common.wrappers import CellPath
from perform_full_wavelet import calculateTFDPower

from common.run_utils import (
    load_config,
    create_default_argument_parser,
    setup_result_directory,
    collect_inputs
)

class Analysis(Protocol):
    """
    A protocol for analysis classes.
    Each analysis class should implement a run method that takes a CellPath
    and returns a DataFrame with the analysis results.
    """
    @abc.abstractmethod
    def run(self, cell_path: CellPath) -> pd.DataFrame:
        """Run the analysis on the given cell path and return the results."""


class SpikeDescriptorAnalysis(Analysis):
    """
    Class to perform the full spike descriptor analysis.
    It loads the spikes, calculates the features, categorizes them,
    and assigns them to episodes.
    """
    def __init__(self, params: dict, paths: dict):
        self._load_spikes = SpikeShapeLoader(params['extractor'])
        self._describe_spikes = RisingFeatureCalculator(params['descriptor'])
        self._categorize_spikes = SpikeCategorizer(**params['classification'])
        self._create_assigner = SpikeIntervalAssignerFactory(**params['episode_assignment'])
        self._save = params['save_results']
        self._overwrite = params['overwrite']
        self._paths = paths

    def run(self, cell_path: CellPath) -> pd.DataFrame:
        """
        Perform the full slowdown analysis on a single cell.
        """
        assign_spikes = self._create_assigner(MatRecording(cell_path.path))

        spikes = self._load_spikes(cell_path.path)
        spikes = self._describe_spikes(spikes)
        spikes = self._categorize_spikes(spikes)
        spikes = assign_spikes(spikes.copy())

        if self._save:
            save_path = (
                self._paths['processed'] /
                cell_path.cell_type /
                f'{cell_path.name}_spikes.csv'
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if self._overwrite or not save_path.exists():
                spikes.to_csv(save_path, index=False)
        return spikes


class SpikeDescriptorAnalysisLoader(Analysis):
    """
    Class to load the spike descriptor analysis results.
    It loads the spikes from the processed directory.
    """
    def __init__(self, params: dict, paths: dict):
        self._paths = paths
        self._spike_types = [
            group['name'] for group in params['classification']['groups']
        ]

    def run(self, cell_path: CellPath) -> pd.DataFrame:
        """
        Load the spikes from the processed directory.
        """
        load_path = (
            self._paths['processed'] /
            cell_path.cell_type /
            f'{cell_path.name}_spikes.csv'
        )
        if not load_path.exists():
            raise FileNotFoundError(f'Spikes not found for {cell_path.name}')
        spikes = pd.read_csv(load_path, index_col=False)
        spikes['spike_type'] = pd.Categorical(
            spikes['spike_type'],
            categories=self._spike_types
        )
        return spikes

def create_analysis(config: dict, paths: dict) -> Analysis:
    """
    Create an instance of SpikeDescriptorAnalysis with the given parameters and paths.
    """
    if config['start_phase'] == 'raw':
        return SpikeDescriptorAnalysis(config, paths)
    elif config['start_phase'] == 'processed':
        return SpikeDescriptorAnalysisLoader(config, paths)
    raise ValueError(
        f"Invalid start phase '{config['start_phase']}' in configuration."
    )

if __name__ == "__main__":
    parser = create_default_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    result_dirs = setup_result_directory(config['output']['path'])
    inputs = collect_inputs(config['input'])

    analysis = create_analysis(config['analysis'], result_dirs)

    for cell_path in inputs:
        recording = MatRecording(cell_path.path)
        spikes = analysis.run(cell_path)

        spike_types = spikes['spike_type'].cat.categories
        fig, axes = plt.subplots(
            1 + len(spike_types), 1, figsize=(20, 6),
            sharex=True, layout='constrained'
        )

        wavelet_params = config['plot']['sleep_distribution']['spectrogram']['params']
        signal = recording.get_waveform('wideband')
        signal = signal.resampled(1000)
        spectrogram = calculateTFDPower(
            signal.values,
            si=signal.si,
            params=wavelet_params
        )
        im = axes[0].pcolormesh(
            signal.times,
            np.arange(wavelet_params['low'], wavelet_params['high'] + wavelet_params['step'], wavelet_params['step']),
            spectrogram,
            vmin=wavelet_params['clim'][0],
            vmax=wavelet_params['clim'][1],
            shading='auto',
            cmap='jet'
        )
        fig.colorbar(
            im,
            ax=axes[0],
            label='Power',
            pad=0.02,
            fraction=0.05,
            location='right',
            anchor=(0, 0.5)
        )
        axes[0].set_title('Time-Frequency Distribution (Spectrogram)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')

        event_plot_params = config['plot']['sleep_distribution']['event_plot']
        for i, spike_type in enumerate(spike_types):
            spike_subset = spikes[spikes['spike_type'] == spike_type]
            if spike_subset.empty:
                continue
            timestamps = spike_subset['timestamp'].to_numpy()
            if event_plot_params['mode'] == 'raster':
                axes[1 + i].vlines(
                    timestamps,
                    np.full(len(spike_subset), -0.25),
                    np.full(len(spike_subset), 0.25),
                    color=sns.color_palette('Set2')[i],
                    alpha=0.5
                )
                axes[1 + i].set_ylim((-1, 1))
                axes[1 + i].set_yticks([0])
                axes[1 + i].set_yticklabels([spike_type])
            elif event_plot_params['mode'] == 'instantaneous':
                axes[1 + i].plot(
                    timestamps[1:],
                    1/np.diff(timestamps),
                    color=sns.color_palette('Set2')[i],
                    alpha=0.5
                )
                axes[1 + i].set_xlabel('Time (s)')

        plt.show()


