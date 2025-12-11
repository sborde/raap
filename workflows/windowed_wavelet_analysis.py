"""This module contains classes and function to run spike autocorrelation analysis workflows."""

import abc


from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pandas import DataFrame

from analyses.model import (
    RecordingWithSpikes,
    select_spikes
)
from signalizer.model.experiment.signals import Signal
from common.data_utils import CellWrapper
from param_classes.workflows.wavelet_analysis import (
    SpikeTriggeredWaveletWorkflowParams,
    WindowedWaveletAnalysisParams,
    WaveletParams,
    WorkflowParams
)

from notebook_utils.recording_wavelet import plot_spectrogram
from perform_full_wavelet import (
    calculate_spectrogram,
    Spectrogram,
    WaveletCalculator,
    save_spectrogram_hdf,
    aggregate_spectrograms
)

from workflows.run_util.io import ResultPaths
from workflows.spike_ac_analysis import (
    save_figure,
)

class SignalExtractor(Protocol):
    """Protocol for callables that extracts signal from a recording"""
    @abc.abstractmethod
    def __call__(self, data: RecordingWithSpikes) -> Signal:
        """Return a signal from the recording with the given sampling rate"""

def calculate_recording_spectrogram(
    data: RecordingWithSpikes,
    *,
    extract_signal: SignalExtractor,
    calculate_wavelet: WaveletCalculator
) -> Spectrogram:
    """Calculate the spectrogram of a signal."""
    return calculate_wavelet(extract_signal(data))

@dataclass
class SpectrogramSpikesPair:
    """A class representing a spectrogram and associated spikes."""
    spectrogram: Spectrogram
    spike_counts: dict[str, int]

    @property
    def window_size(self) -> float:
        """Return the size of the spectrogram window."""
        return self.spectrogram.times[-1] - self.spectrogram.times[0]

    def get_firing_rate(self, name: str) -> float:
        """Return the firing rate of the spikes in Hz."""
        return self.spike_counts.get(name, 0) / self.window_size

def segment_to_windows_with_firing(
    spectrogram: Spectrogram,
    spikes: DataFrame,
    *,
    window_size: float,
    step_size: float
) -> list[SpectrogramSpikesPair]:
    """Calculate the normalized inter RA-AP interval firing histogram."""
    segments = []
    window_start = spectrogram.times[0]
    while window_start <= spectrogram.times[-1] - window_size:
        spikes_subset = spikes[
            (spikes['timestamp'] >= window_start) &
            (spikes['timestamp'] < window_start + window_size)
        ]
        sub_spectrogram = spectrogram.cut_around(
            trigger=window_start, window=(0.0, window_size)
        )
        window_start += step_size
        if sub_spectrogram is None:
            continue
        segments.append(
            SpectrogramSpikesPair(
                spectrogram=sub_spectrogram,
                spike_counts={
                    name: len(spikes_subset[spikes_subset['spike_type'] == name])
                    for name in spikes['spike_type'].unique()
                }
            )
        )
    return segments

def plot_spectrogram_with_power(
    spectrogram: Spectrogram,
    clim: tuple[float, float],
    color: str = 'blue',
    *,
    fig_size: tuple[float, float] = (10, 6)
) -> Figure:
    """Plot the spectrogram with average power over time."""
    fig, axes = plt.subplots(
        1, 2, figsize=fig_size,
        constrained_layout=True,
        width_ratios=[9, 1]
    )
    spec_ax, power_ax = axes

    plot_spectrogram(
        spectrogram,
        clim=clim,
        fig=fig,
        ax=spec_ax
    )

    plot_time_average_spectrogram(
        spectrogram,
        ax=power_ax,
        color=color
    )

    y_limit = spectrogram.freqs[0], spectrogram.freqs[-1]
    spec_ax.set_ylim(y_limit)
    spec_ax.set_xlim(spectrogram.times[0], spectrogram.times[-1])
    power_ax.set_ylim(y_limit)
    return fig

def plot_time_average_spectrogram(
    spectrogram: Spectrogram,
    ax: Axes,
    color: str = 'blue',
):
    """Plot the time-averaged spectrogram power."""
    mean_spectrum = np.mean(spectrogram.power, axis=1)
    std_spectrum = np.std(spectrogram.power, axis=1)
    sem_spectrum = std_spectrum / np.sqrt(spectrogram.power.shape[1])
    ax.fill_betweenx(
        spectrogram.freqs,
        mean_spectrum - sem_spectrum,
        mean_spectrum + sem_spectrum,
        color=color,
        edgecolor='none',
        alpha=0.3
    )
    ax.plot(
        mean_spectrum,
        spectrogram.freqs,
        '-',
        linewidth=2,
        color=color,
    )

def save_spectrogram_figure(
    fig: Figure,
    name: str,
    path: str
):
    """Save the spectrogram figure to the specified path."""
    save_path = Path(path) / '02_figures'
    save_path.mkdir(parents=True, exist_ok=True)
    save_figure(
        save_path,
        fig,
        name,
        to_svg=False
    )

def create_windowed_segmenter(
    params: WindowedWaveletAnalysisParams
) -> Callable[[Spectrogram, DataFrame], list[SpectrogramSpikesPair]]:
    """Return a function that segments spectrogram into windows with associated spikes"""
    return partial(
        segment_to_windows_with_firing,
        window_size=params.window_size,
        step_size=params.step_size
    )

def create_recording_wavelet_calculator(
    params: WorkflowParams[WindowedWaveletAnalysisParams]
):
    """Return a function that calculates wavelet spectrogram for a recording"""
    signal_extractor = create_signal_extractor(
        name=params.analysis.signal.name,
        fs=params.analysis.signal.fs
    )
    wavelet_calculator = create_wavelet_calculator(params.analysis.wavelet)
    return partial(
        calculate_recording_spectrogram,
        extract_signal=signal_extractor,
        calculate_wavelet=wavelet_calculator
    )

def create_signal_extractor(
    name: str, fs: int | None = None
) -> SignalExtractor:
    """Return a function that extracts signal from a recording"""
    def extract(data: RecordingWithSpikes) -> Signal:
        """Return a signal from the recording with the given sampling rate"""
        return data.get_signal(name=name, fs=fs)
    return extract

def create_wavelet_calculator(params: WaveletParams) -> WaveletCalculator:
    """Return a wavelet spectrogram calculator"""
    return partial(
        calculate_spectrogram,
        params=params
    )

def create_wavelet_saver(path: str):
    """Return a wavelet spectrogram saver"""
    save_path = Path(path) / '0_cache' / 'spectrograms'
    save_path.mkdir(parents=True, exist_ok=True)
    return partial(
        save_spectrogram_hdf,
        path=str(save_path)
    )
