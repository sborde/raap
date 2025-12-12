"""This module contains classes and function to run spike autocorrelation analysis workflows."""

import abc


from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol, Self
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

@dataclass
class SpectrogramSpikesPair:
    """A class representing a spectrogram and associated spikes."""
    trigger: float
    spectrogram: Spectrogram
    spikes: DataFrame
    spike_counts: dict[str, int]

    @property
    def window_size(self) -> float:
        """Return the size of the spectrogram window."""
        return self.spectrogram.times[-1] - self.spectrogram.times[0]

    def with_spectrogram(self, spectrogram: Spectrogram) -> Self:
        """Return a new SpectrogramSpikesPair with the given spectrogram."""
        return type(self)(
            trigger=self.trigger,
            spectrogram=spectrogram,
            spikes=self.spikes,
            spike_counts=self.spike_counts
        )

    def get_firing_rate(self, name: str) -> float:
        """Return the firing rate of the spikes in Hz."""
        return self.spike_counts.get(name, 0) / self.window_size

def extract_spectrogram(
    data: RecordingWithSpikes,
    *,
    extract_signal: SignalExtractor,
    calculate_wavelet: WaveletCalculator
) -> Spectrogram:
    """Calculate the spectrogram of a signal."""
    return calculate_wavelet(extract_signal(data))

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
        spike_subsequence = spikes[
            (spikes['timestamp'] >= window_start) &
            (spikes['timestamp'] < window_start + window_size)
        ].copy()
        spike_subsequence['timestamp'] -= window_start
        spectrogram_segment = spectrogram.cut_around(
            trigger=window_start, window=(0.0, window_size)
        )
        if spectrogram_segment is not None:
            segments.append(
                SpectrogramSpikesPair(
                    trigger=window_start - step_size,
                    spectrogram=spectrogram_segment,
                    spikes=spike_subsequence,
                    spike_counts={
                        name: len(spike_subsequence[spike_subsequence['spike_type'] == name])
                        for name in spikes['spike_type'].unique()
                    }
                )
            )
        window_start += step_size
    return segments

def plot_spectrogram_with_power_and_raster(
    segments: list[SpectrogramSpikesPair],
    clim: tuple[float | None, float | None],
    color: str = 'blue',
    *,
    palette: dict[str, str],
    raster_tick_size: float = 1.0,
    fig_size: tuple[float, float] = (10, 6)
) -> Figure:
    """Plot the spectrogram with average power over time."""
    fig, axes = plt.subplots(
        2, 2, figsize=fig_size,
        constrained_layout=True,
        width_ratios=[9, 1]
    )

    spectrogram = aggregate_spectrograms(
        [seg.spectrogram for seg in segments]
    )
    spec_ax, power_ax = axes[0, 0], axes[0, 1]
    raster_ax = axes[1, 0]
    axes[1, 1].axis('off')

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

    plot_spike_raster(
        spikes=[seg.spikes for seg in segments],
        ax=raster_ax,
        palette=palette,
        tick_size=raster_tick_size
    )
    raster_ax.set_xlim(spectrogram.times[0], spectrogram.times[-1])
    return fig

def plot_spike_raster(
    spikes: list[DataFrame],
    ax: Axes,
    palette: dict[str, str],
    tick_size: float = 1.0
):
    """Plot the spike raster."""
    for i, spike_df in enumerate(spikes):
        for spike_type, group in spike_df.groupby('spike_type'):
            ax.vlines(
                group['timestamp'],
                ymin=i - tick_size / 2,
                ymax=i + tick_size / 2,
                color=palette[str(spike_type)],
            )



def plot_spectrogram_with_power(
    spectrogram: Spectrogram,
    clim: tuple[float | None, float | None],
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

def create_spectrogram_segment_extractor(
    params: WindowedWaveletAnalysisParams
) -> Callable[[RecordingWithSpikes], list[SpectrogramSpikesPair]]:
    """Return a function that extracts spectrogram segments from a recording."""
    spectrogram_calculator = create_recording_wavelet_calculator(params)
    segmenter = create_windowed_segmenter(params)
    return partial(
        extract_spectrogram_segments,
        get_spectrogram=spectrogram_calculator,
        cut_segments=segmenter
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
    params: WindowedWaveletAnalysisParams
):
    """Return a function that calculates wavelet spectrogram for a recording"""
    signal_extractor = create_signal_extractor(
        name=params.signal.name,
        fs=params.signal.fs
    )
    wavelet_calculator = create_wavelet_calculator(params.wavelet)
    return partial(
        extract_spectrogram,
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

def extract_spectrogram_segments(
    recording: RecordingWithSpikes,
    *,
    get_spectrogram: Callable[[RecordingWithSpikes], Spectrogram],
    cut_segments: Callable[[Spectrogram, DataFrame], list[SpectrogramSpikesPair]]
) -> list[SpectrogramSpikesPair]:
    """Extract spectrogram segments from a recording."""
    spectrogram = get_spectrogram(recording)
    return cut_segments(spectrogram, recording.spikes)

def calculate_baseline_spectrogram(
    segments: list[SpectrogramSpikesPair],
    spike_type: str,
) -> Spectrogram:
    """Calculate the baseline spectrogram from non-spiking segments."""
    non_spiking_segments = [
        segment.spectrogram for segment in segments
        if segment.spike_counts.get(spike_type, 0) == 0
    ]
    return aggregate_spectrograms(non_spiking_segments)

def collect_spiking_segments(
    segments: list[SpectrogramSpikesPair],
    spike_type: str,
) -> list[SpectrogramSpikesPair]:
    """Divide segments into two groups based on the presence of spikes of a given type."""
    spiking = []
    for segment in segments:
        if segment.spike_counts.get(spike_type, 0) > 0:
            spiking.append(segment)
    return spiking

def create_spectrogram_normalizer(
    baseline: Spectrogram,
    mode: str = 'mean'
) -> Callable[[Spectrogram], Spectrogram]:
    """Return a function that normalizes spectrogram segments by a baseline."""
    return partial(
        normalize_spectrogram_by_baseline,
        baseline=baseline,
        mode=mode
    )

def normalize_spectrogram_by_baseline(
    spectrogram: Spectrogram,
    baseline: Spectrogram,
    *,
    mode: str = 'mean'
) -> Spectrogram:
    """Normalize a spectrogram by a baseline spectrogram."""
    if mode == 'mean':
        baseline_power = np.mean(baseline.power, axis=1).reshape(-1, 1)
    elif mode == 'median':
        baseline_power = np.median(baseline.power, axis=1).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    normalized_power = spectrogram.power / baseline_power
    return spectrogram.with_power(normalized_power)

def normalize_spectrogram_segment(
    segment: SpectrogramSpikesPair,
    normalize: Callable[[Spectrogram], Spectrogram]
) -> SpectrogramSpikesPair:
    """Normalize a spectrogram segment by a baseline spectrogram."""
    return segment.with_spectrogram(normalize(segment.spectrogram))

def plot_and_save_spectrogram(
    spectrogram: Spectrogram,
    name: str,
    path: str,
    *,
    clim: tuple[float | None, float | None] = (None, None),
    fig_size: tuple[float, float] = (5, 3)
):
    """Plot and save a spectrogram figure."""
    fig = plot_spectrogram_with_power(
        spectrogram,
        clim=clim,
        color='blue',
        fig_size=fig_size
    )
    save_spectrogram_figure(
        fig, name, path,
    )
    plt.close(fig)

@dataclass
class PowerSegmentMatrix:
    """A class representing a matrix of power segments."""
    x: np.ndarray
    y: np.ndarray
    power: np.ndarray

def create_segment_power_matrix(
    segments: list[SpectrogramSpikesPair],
    mode: str = 'mean'
) -> PowerSegmentMatrix:
    """Create a matrix of average power values for each segment."""
    if mode == 'mean':
        aggfunc = partial(np.mean, axis=1)
    elif mode == 'median':
        aggfunc = partial(np.median, axis=1)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    power_matrix = np.stack([
        aggfunc(segment.spectrogram.power) for segment in segments
    ], axis=1)
    return PowerSegmentMatrix(
        x=np.array([segment.get_firing_rate('RA-AP') for segment in segments]),
        y=segments[0].spectrogram.freqs,
        power=power_matrix.T
    )
