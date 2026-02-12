"""This module contains components for the burst analysis workflow."""

import numpy as np
import pandas as pd

def analyze_burst_probabilities(
    spikes: pd.DataFrame,
    max_isi: float,
    min_count: int = 2,
) -> dict:
    """Returns the burst RA-AP probabilities"""
    spikes = assign_bursts(
        spikes,
        max_isi=max_isi,
        min_count=min_count
    )
    return describe_bursting(spikes)

def assign_bursts(
    spikes: pd.DataFrame,
    max_isi: float,
    min_count: int = 2,
) -> pd.DataFrame:
    """Returns a list of spikes belonging to bursts."""

    isi = np.diff(spikes['timestamp'])
    is_within_burst = isi <= max_isi
    state_change = np.diff(is_within_burst.astype(int))
    starts = np.where(state_change == 1)[0] + 1
    ends = np.where(state_change == -1)[0] + 1

    assigned_spikes = spikes.copy()
    assigned_spikes['burst_id'] = -1
    for i, (start, end) in enumerate(zip(starts, ends)):
        if end - start + 1 < min_count:
            continue
        assigned_spikes.loc[start:end + 1, 'burst_id'] = i

    return assigned_spikes

def describe_bursting(
    spikes: pd.DataFrame,
) -> dict:
    """Returns a DataFrame with burst properties."""
    bursting_spikes = spikes[spikes['burst_id'] != -1]

    n_spikes = len(spikes)
    n_bursting_spikes = len(bursting_spikes)

    n_raap = int(np.sum(spikes['spike_type'] == 'RA-AP'))
    n_raap_in_bursts = int(np.sum(bursting_spikes['spike_type'] == 'RA-AP'))
    p_bursting = n_bursting_spikes / n_spikes
    p_raap = n_raap / n_spikes
    p_raap_and_burst = n_raap_in_bursts / n_spikes
    p_raap_if_burst = p_raap_and_burst / p_bursting if p_bursting > 0 else np.nan

    return {
        'n_spikes': n_spikes,
        'n_bursting_spikes': n_bursting_spikes,
        'n_raap': n_raap,
        'n_raap_and_burst': n_raap_in_bursts,
        'p_bursting_spikes': p_bursting,
        'p_raap': p_raap,
        'p_raap_and_burst': p_raap_and_burst,
        'p_raap_if_burst': p_raap_if_burst,
        'enrichment_ratio': p_raap_if_burst / p_raap
    }

def analyze_local_density(
    spikes: pd.DataFrame,
    window_size: float,
    spike_type: str
) -> dict:
    """Analyzes the local density of spikes of a given type."""
    neighbor_stats = describe_raap_neighbors(
        spikes,
        window_size=window_size,
        spike_type=spike_type
    )
    return neighbor_stats.mean().to_dict()

def describe_raap_neighbors(
    spikes: pd.DataFrame,
    window_size: float,
    spike_type: str
) -> pd.DataFrame:
    """Returns the neighboring spike statistics for each of the defined spike type"""
    stats = []
    raap_indices = np.where(spikes['spike_type'] == spike_type)[0]
    for idx in raap_indices:
        current_time = spikes.iloc[idx]['timestamp']
        other_spikes = spikes.drop(spikes.index[idx])
        neighbor_spikes = other_spikes[
            (other_spikes['timestamp'] >= current_time - window_size) &
            (other_spikes['timestamp'] <= current_time + window_size)
        ]
        stats.append({
            'n_neighbors': len(neighbor_spikes),
            f'n_{spike_type}': np.sum(neighbor_spikes['spike_type'] == spike_type),
            f'n_non_{spike_type}': np.sum(neighbor_spikes['spike_type'] != spike_type),
        })
    return pd.DataFrame(stats)
