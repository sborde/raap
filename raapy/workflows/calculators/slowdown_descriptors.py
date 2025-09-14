"""Functions to calculate rising phase features of individual spikes."""

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, find_peaks

from workflows.common.models.spike import SpikeShape
from workflows.processors.spike_transformers import find_start_index

FEATURE_PROPERTIES = [
    {
        'name': 'relative_dv_drop',
        'label': 'Relative dV drop'
    },
    {
        'name': 'relative_rebound_time',
        'label': 'Relative rebound time'
    }
]

def calculate_rising_features(spike: SpikeShape, params: dict) -> dict | None:
    """
    Calculate the features of the spike rising phase.
    """

    spike_peak_index = np.argmin(np.abs(spike.times - 0)).astype(int)
    start_i = find_start_index(spike.values[:spike_peak_index + 1], params)

    # if the start cannot be detected, return None
    if start_i is None:
        return None

    noise_properties = {}
    v = spike.values
    if 'smooth' in params and params['smooth']:
        v = savgol_filter(v, **params['smooth'])

        noise_properties = {
            'raw_power': np.mean(spike.values ** 2),
            'smoothed_power': np.mean(v ** 2),
            'noise_power': np.mean((spike.values - v) ** 2)
        }
    else:
        noise_properties = {
            'raw_power': np.nan,
            'smoothed_power': np.nan,
            'noise_power': np.nan
        }

    peak_i = np.argmin(np.abs(spike.times - 0)).astype(int)

    # search on the rising phase by default
    search_start_i = start_i
    # else on the rising phase from the given proportion of the amplitude
    if params['search_start']['mode'] == 'amplitude':
        search_start_i = find_rise_start_index(
            v,
            start_i,
            peak_i,
            params['search_start']['param']['threshold']
        )
    elif params['search_start']['mode'] == 'time':
        rising_sample_count = peak_i - start_i
        search_start_i = np.round(
            rising_sample_count * params['search_start']['param']['threshold']
        ).astype(int) + start_i

    # Find the peaks of the first derivative
    dv = np.diff(v)
    # search until the peak by default
    search_end_i = peak_i
    # else until the global maximum of the first derivative
    if params['search_end']['mode'] == 'dv_max':
        search_end_i = np.argmax(dv)
    elif params['search_end']['mode'] == 'time':
        rising_sample_count = peak_i - start_i
        search_end_i = peak_i - np.round(
            rising_sample_count * params['search_end']['param']['threshold']
        ).astype(int)


    # Calculate the rising phase duration
    dt = np.mean(np.diff(spike.times))
    rising_duration = (search_end_i - search_start_i) * dt

    # find peaks and keep only the valid ones
    peaks, _ = find_peaks(dv, prominence=(None, None))
    peaks = [peak for peak in peaks if search_start_i <= peak <= search_end_i]

    # if there are no secondary peaks, return default values
    if len(peaks) < 2:
        return {
            'timestamp': spike.timestamp,
            'relative_dv_drop': 0,
            'relative_rebound_time': 0,
            't': spike.times,
            'v': v,
            'dv': dv,
            'start_i': start_i,
            'search_start_i': search_start_i,
            'search_end_i': search_end_i,
            'left_peak': None,
            'right_peak': None,
            'valley_i': None,
            'rebound_i': [],
            'rebound_time': None,
            'dv_min_peak': None
        } | noise_properties

    features = []
    dv_max_i = np.argmax(dv[:peak_i])
    for peak in peaks:
        if peak == dv_max_i:
            # skip the global maximum peak
            continue

        left_peak, right_peak = (
            (peak, dv_max_i) if peak < dv_max_i else (dv_max_i, peak)
        )
        dv_interpeak = dv[left_peak:right_peak + 1]
        dv_min_peak = min(dv_interpeak[0], dv_interpeak[-1])

        # Find the valley between the peaks
        dv_valley_i = np.argmin(dv_interpeak)
        dv_valley = dv_interpeak[dv_valley_i]

        # Calculate the relative drop in dv
        dv_drop = dv_min_peak - dv_valley

        # find the first point left of the valley where dv reaches the min peak
        left_level_i = np.where(dv_interpeak <= dv_min_peak)[0][0]
        # find the first point right of the valley where dv reaches the min peak
        right_level_i = np.where(dv_interpeak <= dv_min_peak)[0][-1]

        # calculate the real rebound time
        rebound_time = (right_level_i - left_level_i) * dt
        if dv_interpeak[left_level_i] < dv_min_peak:
            alpha = (
                (dv_min_peak - dv_interpeak[left_level_i]) /
                (dv_interpeak[left_level_i-1] - dv_interpeak[left_level_i])
            )
            rebound_time += alpha * dt
        if dv_interpeak[right_level_i] < dv_min_peak:
            alpha = (
                (dv_min_peak - dv_interpeak[right_level_i]) /
                (dv_interpeak[right_level_i+1] - dv_interpeak[right_level_i])
            )
            rebound_time += (1 - alpha) * dt

        features.append({
            'timestamp': spike.timestamp,
            'dv_drop': dv_drop,
            'relative_dv_drop': dv_drop / dv[dv_max_i],
            'relative_rebound_time': rebound_time / rising_duration,
            't': spike.times,
            'v': v,
            'dv': dv,
            'start_i': start_i,
            'search_start_i': search_start_i,
            'search_end_i': search_end_i,
            'left_peak': left_peak,
            'right_peak': right_peak,
            'valley_i': dv_valley_i + left_peak,
            'rebound_i': [left_level_i + left_peak, right_level_i + left_peak],
            'rebound_time': rebound_time,
            'dv_min_peak': dv_min_peak
        } | noise_properties)

    return sorted(features, key=lambda x: x['dv_drop'])[-1]

def find_rise_start_index(
    v: np.ndarray,
    start_i: int,
    peak_i: int,
    threshold: float
) -> int:
    """
    Find the start index of the rising phase of a spike.
    The rising phase is defined as the first point after the start index
    where the signal exceeds a certain proportion of its amplitude.
    """
    # align rising phase to the start
    rising_phase = v[start_i:peak_i + 1] - v[start_i]
    threshold_level = threshold * rising_phase[-1]

    # find the first index where the rising phase exceeds the threshold
    return np.where(rising_phase > threshold_level)[0][0] + start_i

def convert_features_to_dataframe(features: list[dict]) -> pd.DataFrame:
    """
    Convert the list of features to a pandas DataFrame.
    """

    rows = []
    for f in features:
        rows.append({
            'timestamp': f['timestamp'],
            'relative_dv_drop': f['relative_dv_drop'],
            'relative_rebound_time': f['relative_rebound_time'],
            'raw_power': f['raw_power'],
            'smoothed_power': f['smoothed_power'],
            'noise_power': f['noise_power'],
            'start_t': f['t'][f['start_i']],
            'start_v': f['v'][f['start_i']],
            'peak_t': f['t'][np.argmin(np.abs(f['t']))],
            'peak_v': f['v'][np.argmin(np.abs(f['t']))],
            'search_start_t': f['t'][f['search_start_i']],
            'search_start_v': f['v'][f['search_start_i']],
            'search_end_t': f['t'][f['search_end_i']],
            'search_end_v': f['v'][f['search_end_i']],
            'left_peak_t': f['t'][f['left_peak']] if f['left_peak'] is not None else None,
            'left_peak_v': f['v'][f['left_peak']] if f['left_peak'] is not None else None,
            'right_peak_t': f['t'][f['right_peak']] if f['right_peak'] is not None else None,
            'right_peak_v': f['v'][f['right_peak']] if f['right_peak'] is not None else None,
            'valley_t': f['t'][f['valley_i']] if f['valley_i'] is not None else None,
            'valley_v': f['v'][f['valley_i']] if f['valley_i'] is not None else None,
            'rebound_start_t': f['t'][f['rebound_i'][0]] if f['rebound_i'] else None,
            'rebound_end_t': f['t'][f['rebound_i'][1]] if f['rebound_i'] else None
        })

    return pd.DataFrame(rows)
