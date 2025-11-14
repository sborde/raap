"""A module for utility function of the recording wavelet analysis."""

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from signalizer.model.experiment.signals import Signal
from signalizer.model.experiment.recordings import MatRecording

from perform_full_wavelet import calculate_tfd_power, Spectrogram



def get_signal(recording: MatRecording, params: dict) -> Signal:
    """Retrieve and resample a signal from a recording."""
    signal = recording.get_waveform(params['name'])
    return signal.resampled(params['sampling_rate'])


def calculate_spectrogram(signal: Signal, params: dict) -> Spectrogram:
    """Calculate the spectrogram of a signal using wavelet transform."""
    power = calculate_tfd_power(signal.values, signal.si, params)
    freqs = np.arange(
        params['low'], params['high'] + params['step'], params['step']
    )
    return Spectrogram(
        signal.times,
        freqs,
        power,
        signal.fs
    )

def plot_spectrogram(
    spectrogram: Spectrogram,
    clim: tuple,
    fig: Figure,
    ax: Axes
):
    """Plot the spectrogram of a signal."""

    im = ax.imshow(
        spectrogram.power,
        aspect='auto',
        extent=(
            spectrogram.times[0],
            spectrogram.times[-1],
            spectrogram.freqs[0],
            spectrogram.freqs[-1]
        ),
        origin='lower',
        vmin=clim[0],
        vmax=clim[1],
        cmap='jet',
        interpolation='bicubic'
    )

    cb = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        fraction=0.05,
        location='right',
        anchor=(0, 0.5)
    )

    cb.set_label('Power')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    ax.set_title('Time-Frequency Distribution (Spectrogram)')
