
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.run_utils import (
    load_config,
    collect_inputs
)
from common.wrappers import CellPath
from perform_full_wavelet import calculateTFDPower
from signalizer.model.experiment.recordings import MatRecording

def load_spike_features(cell_path: CellPath, params: dict) -> pd.DataFrame:
    """
    Load the spikes from the processed directory.
    """
    spike_types = [
        group['name'] for group in params['classification']['groups']
    ]

    load_path = cell_path.path
    print(load_path)
    spikes = pd.read_csv(load_path, index_col=False)
    spikes['spike_type'] = pd.Categorical(
        spikes['spike_type'],
        categories=spike_types
    )
    return spikes

if __name__ == "__main__":

    config = load_config(
        'config/spike_distribution_over_sleep.yaml'
    )

    raw_inputs = collect_inputs(config['raw_input'])
    inputs = collect_inputs(config['preprocessed_input'])

    for (raw_cell_path, preprocessed_cell_path) in zip(raw_inputs, inputs):
        recording = MatRecording(raw_cell_path.path)
        spikes = load_spike_features(preprocessed_cell_path, config['analysis'])

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