"""This module contains components for the burst analysis workflow."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from workflows.spike_ac_analysis import save_figure

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

    n_aisap = int(np.sum(spikes['spike_type'] == 'AIS-AP'))
    n_undef = int(np.sum(spikes['spike_type'] == 'UNKNOWN'))
    n_raap = int(np.sum(spikes['spike_type'] == 'RA-AP'))

    n_raap_in_bursts = int(np.sum(bursting_spikes['spike_type'] == 'RA-AP'))
    n_aisap_in_bursts = int(np.sum(bursting_spikes['spike_type'] == 'AIS-AP'))
    n_undef_in_bursts = int(np.sum(bursting_spikes['spike_type'] == 'UNKNOWN'))
    p_bursting = n_bursting_spikes / n_spikes
    p_aisap = n_aisap / n_spikes
    p_undef = n_undef / n_spikes
    p_raap = n_raap / n_spikes

    p_aisap_and_burst = n_aisap_in_bursts / n_spikes
    p_undef_and_burst = n_undef_in_bursts / n_spikes
    p_raap_and_burst = n_raap_in_bursts / n_spikes

    p_aisap_if_burst = p_aisap_and_burst / p_bursting if p_bursting > 0 else np.nan
    p_undef_if_burst = p_undef_and_burst / p_bursting if p_bursting > 0 else np.nan
    p_raap_if_burst = p_raap_and_burst / p_bursting if p_bursting > 0 else np.nan

    enrichment_ratio = p_raap_if_burst / p_raap if p_raap > 0 else np.nan
    log_enrichment = np.log2(enrichment_ratio) if enrichment_ratio > 0 else np.nan

    return {
        'n_spikes': n_spikes,
        'n_bursting_spikes': n_bursting_spikes,
        'n_aisap': n_aisap,
        'n_undef': n_undef,
        'n_raap': n_raap,
        'n_aisap_in_burst': n_aisap_in_bursts,
        'n_undef_in_burst': n_undef_in_bursts,
        'n_raap_and_burst': n_raap_in_bursts,
        'p_bursting_spikes': p_bursting,
        'p_aisap': p_aisap,
        'p_undef': p_undef,
        'p_raap': p_raap,
        'p_aisap_and_burst': p_aisap_and_burst,
        'p_undef_and_burst': p_undef_and_burst,
        'p_raap_and_burst': p_raap_and_burst,
        'p_aisap_if_burst': p_aisap_if_burst,
        'p_undef_if_burst': p_undef_if_burst,
        'p_raap_if_burst': p_raap_if_burst,
        'raap_enrichment_ratio': enrichment_ratio,
        'log_raap_enrichment': log_enrichment
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

def analyze_spike_neighborhood(
    spikes: pd.DataFrame,
    window_size: float
) -> dict:
    """Analyzes spike counts in the spike neighborhood"""
    all_neighbor = count_neighboring_spikes(
        spikes,
        window_size=window_size,
        spike_type=None
    )
    raap_neighbor = count_neighboring_spikes(
        spikes,
        window_size=window_size,
        spike_type='RA-AP'
    )
    return (
        all_neighbor.mean().to_dict() |
        raap_neighbor.mean().to_dict()
    )

def count_neighboring_spikes(
    spikes: pd.DataFrame,
    window_size: float,
    spike_type: str | None = None
) -> pd.DataFrame:
    """Returns the spike count in the neighborhood of all spikes of the given type"""
    if spike_type is None:
        indices = list(range(len(spikes)))
        prefix = ''
    else:
        indices = np.where(spikes['spike_type'] == spike_type)[0]
        prefix = f'{spike_type}_'

    stats = []
    for idx in indices:
        current_time = spikes.iloc[idx]['timestamp']
        other_spikes = spikes.drop(spikes.index[idx])
        neighbor_spikes = other_spikes[
            (other_spikes['timestamp'] >= current_time - window_size) &
            (other_spikes['timestamp'] <= current_time + window_size)
        ]
        stats.append({
            f'{prefix}neighbor_count': len(neighbor_spikes),
            f'{prefix}neighbor_count_AIS-AP': np.sum(neighbor_spikes['spike_type'] == 'AIS-AP'),
            f'{prefix}neighbor_count_UNKNOWN': np.sum(neighbor_spikes['spike_type'] == 'UNKNOWN'),
            f'{prefix}neighbor_count_RA-AP': np.sum(neighbor_spikes['spike_type'] == 'RA-AP'),
        })
    return pd.DataFrame(stats)

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

def plot_prob_change_per_cell(
    data: pd.DataFrame,
    group_names: list[str],
    palette: dict[str, str],
    save_path: str
):
    """Plots the probabilities of RA-AP and RA-AP|Burst for each cell."""
    n_cells = len(data)
    fig, ax = plt.subplots(1, 1, figsize=(20, n_cells * 0.3))
    y_labels = []
    for group in group_names:
        color = palette[group]
        sub_data = data[data['group'] == group]
        y_labels += sub_data['name'].to_list()
        for fill_flag, feature in zip([True, False], ['p_raap', 'p_raap_if_burst']):
            ax.scatter(
                sub_data[feature],
                sub_data.index,
                marker='o',
                edgecolors=color if not fill_flag else None,
                facecolors=color if fill_flag else 'none',
            )
    ax.set_yticks(
        ticks=list(range(len(y_labels))),
        labels=y_labels
    )
    ax.set_title('P(RA-AP) (filled) vs P(RA-AP|Burst) (empty) - per cell')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_figure(
        Path(save_path),
        fig,
        'conditional_probability_change_per_cell',
        to_svg=True
    )
    plt.close()

def plot_prob_change_per_group(
    data: pd.DataFrame,
    group_names: list[str],
    palette: dict[str, str],
    save_path: str
):
    """Plots the probability change per group with a box and scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    for i, group_name in enumerate(group_names):
        sub_data = data[data['group'] == group_name]
        color = palette[group_name]
        for fill_flag, feature in zip([-1, 1], ['p_raap', 'p_raap_if_burst']):
            _plot_box_strip_at_position(
                sub_data[feature].to_numpy(),
                position=i + fill_flag * 0.1,
                ax=ax,
                color=color,
                fill_markers=fill_flag == -1
            )
        for _, row in sub_data.iterrows():
            ax.plot(
                [i - 0.1, i + 0.1],
                [row['p_raap'], row['p_raap_if_burst']],
                color='gray',
                linewidth=0.5
            )
    ax.set_xticks(
        ticks=list(range(len(group_names))),
        labels=group_names
    )
    ax.set_title('P(RA-AP) (filled) vs P(RA-AP|Burst) (empty) - per group')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_figure(
        Path(save_path),
        fig,
        'conditional_probability_change_per_group_boxplot',
        to_svg=True
    )
    plt.close()

def plot_spiketype_prob_diff_per_group(
    data: pd.DataFrame,
    group_names: list[str],
    palette: dict[str, dict[str, str]],
    save_path: str,
    suffix: str = ''
):
    """Plots the probability change per group with a box and scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    for i, group_name in enumerate(group_names):
        sub_data = data[data['group'] == group_name]
        group_color = palette['cell_type'][group_name]
        raap_color = palette['spike_type']['RA-AP']
        for j, (color, feature) in enumerate(zip(
            [group_color, raap_color],
            ['p_aisap_if_burst', 'p_raap_if_burst']
        )):
            _plot_box_strip_at_position(
                sub_data[feature].to_numpy(),
                position=i + (- 0.1 + j * 0.2),
                ax=ax,
                color=color,
            )
        for _, row in sub_data.iterrows():
            ax.plot(
                [i - 0.1, i + 0.1],
                [row['p_aisap_if_burst'], row['p_raap_if_burst']],
                color='gray',
                linewidth=0.5
            )
    ax.set_xticks(
        ticks=list(range(len(group_names))),
        labels=group_names
    )
    ax.set_title('P(AISAP|Burst) vs P(RA-AP|Burst) - per group')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_figure(
        Path(save_path),
        fig,
        f'aisap_if_burst_vs_raap_if_burst_per_group_boxplot{suffix}',
        to_svg=True
    )
    plt.close()

def plot_box_and_strip_per_group(
    data: pd.DataFrame,
    group_names: list[str],
    palette: dict[str, str],
    feature_name: str,
    save_path: str,
    plot_title: str = '',
):
    """Plots the bursting probability per group with a box and scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    for i, group_name in enumerate(group_names):
        sub_data = data[data['group'] == group_name]
        color = palette[group_name]
        _plot_empty_box(
            sub_data[feature_name].to_numpy(),
            position=i,
            ax=ax
        )
        _plot_points_vertical(
            sub_data[feature_name].to_numpy(),
            position=i,
            ax=ax,
            color=color
        )
    ax.set_xticks(
        ticks=list(range(len(group_names))),
        labels=group_names
    )
    if not plot_title:
        plot_title = feature_name.capitalize()
    ax.set_title(f'{plot_title} - per group')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_figure(
        Path(save_path),
        fig,
        f'{feature_name}_per_group_boxplot',
        to_svg=True
    )
    plt.close()


def plot_neighborhood_count_per_group(
    data: pd.DataFrame,
    group_names: list[str],
    palette: dict[str, str],
    save_path: str
):
    """Plots the neighborhood count per group with a box and scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    feature_pair = ('neighbor_count', 'RA-AP_neighbor_count')
    for i, group_name in enumerate(group_names):
        sub_data = data[data['group'] == group_name].dropna(axis=0)
        color = palette[group_name]
        for fill_flag, feature in zip([-1, 1], feature_pair):
            _plot_box_strip_at_position(
                sub_data[feature].to_numpy(),
                position=i + fill_flag * 0.1,
                ax=ax,
                color=color,
                fill_markers=fill_flag == -1
            )
        for _, row in sub_data.iterrows():
            ax.plot(
                [i - 0.1, i + 0.1],
                [row[feature_pair[0]], row[feature_pair[1]]],
                color='gray',
                linewidth=0.5
            )
    ax.set_xticks(
        ticks=list(range(len(group_names))),
        labels=group_names
    )
    ax.set_title(
        f'{feature_pair[0]} vs {feature_pair[1]} - per group'
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_figure(
        Path(save_path),
        fig,
        f'{feature_pair[0]}_vs_{feature_pair[1]}_per_group_boxplot',
        to_svg=True
    )
    plt.close()

def _plot_box_strip_at_position(
    values: np.ndarray,
    position: float,
    ax: Axes,
    color: str,
    fill_markers: bool = True,
    width: float = 0.15
):
    """Plots a box and strip plot at the specified position."""
    _plot_empty_box(values, position, ax, width)
    _plot_points_vertical(values, position, ax, color, fill=fill_markers)

def _plot_empty_box(
    values: np.ndarray,
    position: float,
    ax: Axes,
    width: float = 0.15
):
    """Draws an unfilled boxplot for the given values at the specified position."""
    is_non_nan = ~np.isnan(values)
    ax.boxplot(
        values[is_non_nan],
        positions=[position],
        widths=width,
        patch_artist=True,
        flierprops={'markersize': 0},
        boxprops={'facecolor': 'none', 'edgecolor': 'gray'},
        medianprops={'color': 'gray'},
        whiskerprops={'color': 'gray'},
        capprops={'linewidth': 0}
    )

def _plot_points_vertical(
    values: np.ndarray,
    position: float,
    ax: Axes,
    color: str,
    marker: str = 'o',
    fill: bool = True,
):
    """Plots points vertically at the specified position."""
    ax.scatter(
        [position] * len(values),
        values,
        marker=marker,
        edgecolors=color if not fill else None,
        facecolors=color if fill else 'none',
        color=color
    )

def test_enrichment_ratio(
    data: pd.DataFrame,
    test_groups: list[str],
    save_path: str
):
    """Tests the enrichment ratio between groups using a Mann-Whitney U test."""
    p_raws = []
    for group_name in test_groups:
        group_data = data[data['group'] == group_name]['log_enrichment'].dropna()
        _, p = wilcoxon(group_data)
        p_raws.append(p)
    _, p_adjs, _, _ = multipletests(p_raws, method='holm')

    result_df = pd.DataFrame([
        {
            'group': group_name,
            'p_raw': p_raw,
            'p_adj': p_adj,
        }
        for (group_name, p_raw, p_adj) in zip(test_groups, p_raws, p_adjs)
    ])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    result_df.to_excel(
        f'{save_path}\\enrichment_ratio_statistical_tests.xlsx',
        index=False
    )
