"""This module contains functions for band power analysis around RA-AP."""

import numpy as np
import pandas as pd

from analyses.model import RecordingWithSpikes

from common.data_utils import CellWrapper
from param_classes.workflows.wavelet_analysis import WindowedWaveletAnalysisParams
from perform_full_wavelet import Spectrogram
from workflows.run_util.io import ResultPaths
from workflows.windowed_wavelet_analysis import (
    create_recording_wavelet_calculator,
    normalize_spectrogram_by_baseline,
    aggregate_spectrograms,
    plot_spectrogram_with_power
)

def calculate_mean_band_power_for_cell(
    input_file: CellWrapper[ResultPaths],
    params: WindowedWaveletAnalysisParams,
    frequency_bands: list[tuple],
    time_intervals: list[tuple[float, float]],
) -> pd.DataFrame:
    """Calculates the mean power around RA-AP in specified bands."""
    cell_data = input_file.data.load()

    spectrogram = extract_signal_spectrogram(
        cell_data,
        params
    )
    spectrogram = normalize_spectrogram_by_baseline(
        spectrogram,
        baseline=spectrogram,
        mode='zscore'
    )
    segments = cut_spike_triggered_spectrogram_segments(
        spectrogram,
        cell_data.spikes,
        window_size=params.window_size,
        spike_type='RA-AP'
    )
    mean_spectrogram = aggregate_spectrograms(segments)

    mean_band_powers = []
    for  freq_band in frequency_bands:
        for time_interval in time_intervals:
            band_power = aggregate_band_power(
                mean_spectrogram,
                freq_band=freq_band,
                time_interval=time_interval
            )
            mean_band_powers.append({
                'band': '_'.join(str(freq) for freq in freq_band),
                'part': '_'.join(str(time) for time in time_interval),
                'power': band_power
            })
    result_df = pd.DataFrame(mean_band_powers)
    result_df.insert(0, 'group', input_file.cell_type)
    result_df.insert(0, 'name', input_file.name)

    return result_df

def extract_signal_spectrogram(
    data: RecordingWithSpikes,
    params: WindowedWaveletAnalysisParams
) -> Spectrogram:
    """Returns the calculated spectrogram of the specified signal"""
    calculate_spectrogram = create_recording_wavelet_calculator(params)
    return calculate_spectrogram(data)

def cut_spike_triggered_spectrogram_segments(
    spectrogram: Spectrogram,
    spikes: pd.DataFrame,
    *,
    window_size: float,
    spike_type: str
) -> list[Spectrogram]:
    """Calculate the normalized inter RA-AP interval firing histogram."""
    selected_spikes = spikes[spikes['spike_type'] == spike_type]
    segments = []
    for _, row in selected_spikes.iterrows():
        segment = spectrogram.cut_around(
            trigger=row['timestamp'],
            window=(-window_size, window_size)
        )
        if segment is not None:
            segments.append(segment)

    return segments

def aggregate_band_power(
    spectrogram: Spectrogram,
    freq_band: tuple[float, float],
    time_interval: tuple[float, float]
) -> float:
    """
    Returns the mean power before and after the trigger in the specified band
    """
    times = spectrogram.times
    freqs = spectrogram.freqs
    freq_flag = (freq_band[0] <= freqs) & (freqs <= freq_band[1])
    time_flag = (time_interval[0] <= times) & (times <= time_interval[1])
    band_power = np.mean(spectrogram.power[freq_flag, :], axis=0)
    return float(np.mean(band_power[time_flag]))
