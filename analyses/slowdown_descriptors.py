"""Functions to calculate rising phase features of individual spikes."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from common.data_utils import PathEntry
from data.spikes import SpikeShape, load_spike_shapes

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

def find_start_index(v: np.ndarray, params: dict) -> int | None:
    """
    Find the start index of a spike.

    It first smooths the signal, then calculates the first derivative.
    We consider threshold dV value as the given fraction of the maximum
    derivative value before the maximum. If the signal never reaches this
    threshold, it returns None. Else, it returns the first index where the
    derivative is at or above the threshold.
    """

    if 'smooth' in params and params['smooth']:
        v = savgol_filter(v, **params['smooth'])

    peak_i = np.argmax(v)
    dv = np.diff(v[:peak_i])
    if len(dv) == 0:
        # if there is no derivative, return None
        return None
    max_dv_i = np.argmax(dv)
    dv_ratio = params['start_threshold']
    threshold_dv = dv_ratio * dv[max_dv_i]

    below_index = np.where(dv[:max_dv_i] < threshold_dv)[0]

    return below_index[-1] + 1 if len(below_index) > 0 else None

def align_spike_to_zero(spike: SpikeShape, params: dict) -> SpikeShape:
    """
    Align the spike to zero time.
    The spike is shifted so that the peak is at zero time.
    """
    spike_peak_index = np.argmin(np.abs(spike.times - 0)).astype(int)
    start_i = find_start_index(spike.values[:spike_peak_index + 1], params)

    # if the start cannot be detected, return None
    if start_i is None:
        raise ValueError(
            f'Spike {spike.timestamp} start can\'t be detected.'
        )

    start_v = spike.values[start_i]

    return SpikeShape(
        spike.times,
        spike.values - start_v,
        spike.timestamp,
        spike.dt
    )

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

def calculate_rising_features(spike: SpikeShape, params: dict) -> dict | int:
    """
    Calculate the features of the spike rising phase.
    """

    spike_peak_index = np.argmin(np.abs(spike.times - 0)).astype(int)
    start_i = find_start_index(spike.values[:spike_peak_index + 1], params)

    # if the start cannot be detected, return None
    if start_i is None:
        return 1

    noise_properties = {}
    v = spike.values
    if 'smooth' in params and params['smooth']:
        v = savgol_filter(v, **params['smooth'])

        raw_power = np.mean(spike.values ** 2)
        noise_power = np.mean((spike.values - v) ** 2)
        noise_properties = {
            'raw_power': raw_power,
            'smoothed_power': np.mean(v ** 2),
            'noise_power': noise_power
        }

        noise_limit = params.get('noise_limit')
        if noise_limit is not None:
            noise_ratio = (noise_power / raw_power) * 100
            if noise_ratio > noise_limit:
                print(
                    f"[INFO] Spike {spike.timestamp}'s noise ratio "
                    "is above limit, skip spike."
                )
                return 2
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

    try:
        half_amplitude_left, half_amplitude_right = _calculate_halfwidth(
            spike.times, v, start_i, peak_i
        )
    except ValueError:
        print("[INFO] Half-width not found, skip spike.")
        return 3

    amplitude = v[peak_i] - v[start_i]
    risetime = spike.times[peak_i] - spike.times[start_i]
    base_v = float(np.mean(spike.values[:start_i + 1]))
    start_v = v[start_i]
    rising_slope = amplitude / risetime if risetime > 0 else np.nan

    if start_i < 2:
        baseline_slope = np.nan
    else:
        X = spike.times[:start_i+1].reshape(-1, 1)
        y = spike.values[:start_i+1]
        model = LinearRegression().fit(X, y)
        baseline_slope = float(model.coef_[0])

    basic_features = {
        'amplitude': amplitude,
        'risetime': risetime,
        'base_v': base_v,
        'start_v': start_v,
        'rising_slope': rising_slope,
        'baseline_slope': baseline_slope,
        'half_width': half_amplitude_right[0] - half_amplitude_left[0]
    }

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
        } | noise_properties | basic_features

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
        } | noise_properties | basic_features)

    return sorted(features, key=lambda x: x['dv_drop'])[-1]

def _calculate_halfwidth(
    t: np.ndarray,
    v: np.ndarray,
    start_i: int,
    peak_i: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    # Calculate the half-width of the spike
    half_amplitude = (v[peak_i] - v[start_i]) / 2 + v[start_i]

    i = start_i
    while i < peak_i and v[i + 1] < half_amplitude:
        i += 1
    left_time = _calculate_exact_value(
        (v[i], v[i + 1]), half_amplitude, (t[i], t[i + 1])
    )

    i = peak_i
    while i < len(v) - 1 and v[i + 1] > half_amplitude:
        i += 1


    if i == len(v) - 1:
        raise ValueError('Half-width right side not found.')

    right_time = _calculate_exact_value(
        (v[i], v[i + 1]), half_amplitude, (t[i], t[i + 1])
    )

    return (
        (left_time, half_amplitude),
        (right_time, half_amplitude)
    )

def _calculate_exact_value(
    x: tuple[float, float],
    x_target: float,
    y: tuple[float, float]
) -> float:
    # Linear interpolation to find the exact value at x_target
    x_left, x_right = x
    y_left, y_right = y
    delta_y = y_right - y_left
    alpha = (x_target - x_left) / (x_right - x_left)
    return y_left + delta_y * alpha

def convert_features_to_dataframe(features: list[dict]) -> pd.DataFrame:
    """
    Convert the list of features to a pandas DataFrame.
    """
    rows = []
    for f in features:
        t = f['t']
        v = f['v']
        rows.append({
            'timestamp': f['timestamp'],
            'amplitude': f['amplitude'],
            'risetime': f['risetime'],
            'base_v': f['base_v'],
            'rising_slope': f['rising_slope'],
            'baseline_slope': f['baseline_slope'],
            'half_width': f['half_width'],
            'relative_dv_drop': f['relative_dv_drop'],
            'relative_rebound_time': f['relative_rebound_time'],
            'raw_power': f['raw_power'],
            'smoothed_power': f['smoothed_power'],
            'noise_power': f['noise_power'],
            'start_t': t[f['start_i']],
            'start_v': v[f['start_i']],
            'peak_t': t[np.argmin(np.abs(f['t']))],
            'peak_v': v[np.argmin(np.abs(f['t']))],
            'search_start_t': t[f['search_start_i']],
            'search_start_v': v[f['search_start_i']],
            'search_end_t': t[f['search_end_i']],
            'search_end_v': v[f['search_end_i']],
            'left_peak_t': t[f['left_peak']] if f['left_peak'] is not None else None,
            'left_peak_v': v[f['left_peak']] if f['left_peak'] is not None else None,
            'right_peak_t': t[f['right_peak']] if f['right_peak'] is not None else None,
            'right_peak_v': v[f['right_peak']] if f['right_peak'] is not None else None,
            'valley_t': t[f['valley_i']] if f['valley_i'] is not None else None,
            'valley_v': v[f['valley_i']] if f['valley_i'] is not None else None,
            'rebound_start_t': t[f['rebound_i'][0]] if f['rebound_i'] else None,
            'rebound_end_t': t[f['rebound_i'][1]] if f['rebound_i'] else None
        })

    return pd.DataFrame(rows)

def calculate_sparse_2d_histogram(x, labels, bin_size) -> pd.DataFrame:
    """
    Calculates a sparse 2D histogram of the given data points.
    The histogram bins are defined by the bin size in the config.
    The function returns a DataFrame with the bin centers and counts.
    """

    def generate_bins(values):
        """
        Divides a value set into bins of a given size.
        """
        value_range = max(values) - min(values)
        n_bins = int(np.ceil(value_range / bin_size)) if value_range > 0 else 1
        padding = (n_bins * bin_size - value_range) / 2
        return np.linspace(
            min(values) - padding,
            max(values) + padding,
            n_bins + 1
        )

    x1_bins = generate_bins(x[:, 0])
    x2_bins = generate_bins(x[:, 1])
    x1_centers = (x1_bins[:-1] + x1_bins[1:]) / 2
    x2_centers = (x2_bins[:-1] + x2_bins[1:]) / 2

    counts = []
    for label in np.unique(labels):
        group_data = x[labels == label, :]
        group_size = group_data.shape[0]
        for i in range(len(x1_bins) - 1):
            for j in range(len(x2_bins) - 1):
                count = np.sum(
                    (group_data[:, 0] >= x1_bins[i]) &
                    (group_data[:, 0] < x1_bins[i + 1]) &
                    (group_data[:, 1] >= x2_bins[j]) &
                    (group_data[:, 1] < x2_bins[j + 1])
                )

                if count == 0:
                    continue

                counts.append({
                    'x1': x1_centers[i],
                    'x2': x2_centers[j],
                    'count': count,
                    'ratio': count / group_size,
                    'label': label
                })

    return pd.DataFrame(counts).sort_values(by=['label'], ascending=True)

def create_debug_images(
    results: list[dict],
    output_dir: Path,
    output_dpi: int = 72
) -> None:
    """
    Plots each individual spike with the calculated features and some
    auxiliary information about the description process.
    """
    fig_size = (12, 20)
    n_spike_on_figure = 10
    n_rows, n_cols = 10, 3

    # Sort the results by relative_dv_drop
    results = sorted(results, key=lambda x: x['relative_dv_drop'])

    def create_fig():
        """Create a new figure with the specified size."""
        return plt.subplots(n_rows, n_cols, figsize=fig_size)

    fig, axes = create_fig()
    for i, feature in enumerate(results):
        row_i = i % n_spike_on_figure

        feature = feature

        t = feature['t']
        dvt = t[:-1]
        v = feature['v']
        dv = feature['dv']

        start_i = feature['start_i']
        search_start_i = feature['search_start_i']
        search_end_i = feature['search_end_i']

        left_peak = feature['left_peak']
        right_peak = feature['right_peak']
        valley_i = feature['valley_i']

        ax1 = axes[row_i, 0]
        ax1.plot(t, v, '-', color='blue')
        ax1.plot(t[start_i], v[start_i], 'o', color='red')
        ax1.plot(
            t[search_start_i:search_end_i+1],
            v[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        # ax1.set_xlim(-0.0005, 0.0005)
        ax1.set_xticks(np.linspace(t[0], t[-1], 4))

        ax2 = axes[row_i, 1]
        ax2.plot(dvt, dv, '-', color='green')
        ax2.plot(
            dvt[search_start_i:search_end_i+1],
            dv[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        ax2.plot(t[np.argmax(dv)], dv[np.argmax(dv)], 's', color='blue')
        # ax2.set_xlim(-0.0005, 0.0005)
        ax2.set_xticks(np.linspace(t[0], t[-1], 4))


        if left_peak is not None and right_peak is not None:
            ax2.plot(t[left_peak], dv[left_peak], 'kx', markersize=5)
            ax2.plot(t[right_peak], dv[right_peak], 'kx', markersize=5)

            ax2.plot(
                t[valley_i],
                dv[valley_i],
                'v',
                color='green'
            )
            ax2.vlines(
                t[valley_i],
                dv[valley_i],
                feature['dv_min_peak'],
                linestyle='--',
                color='black'
            )
            ax2.hlines(
                feature['dv_min_peak'],
                dvt[feature['rebound_i'][0]],
                dvt[feature['rebound_i'][1]],
                linestyle='--',
                color='black'
            )

        ax2.set_title(
            f't={feature["timestamp"]:.3f}s ' +
            f'rdv={feature["relative_dv_drop"]:.3f} ' +
            f'rrt={feature["relative_rebound_time"]:.3f}'
        )

        ax3 = axes[row_i, 2]
        ax3.plot(v[:-1], dv, '-', color='blue')
        ax3.plot(
            v[search_start_i:search_end_i+1],
            dv[search_start_i:search_end_i+1],
            '--',
            color='red'
        )
        ax3.plot(
            v[np.argmax(dv)],
            dv[np.argmax(dv)],
            's',
            color='blue'
        )
        if left_peak is not None and right_peak is not None:
            ax3.plot(v[left_peak], dv[left_peak], 'kx', markersize=5)
            ax3.plot(v[right_peak], dv[right_peak], 'kx', markersize=5)
            ax3.plot(v[valley_i], dv[valley_i], 'v', color='green')

        if (i+1) % n_spike_on_figure == 0 or i == len(results) - 1:
            plt.tight_layout(pad=1.0)
            plt.savefig(
                output_dir / f'spike_shape_{i // n_spike_on_figure + 1}.png',
                dpi=output_dpi
            )
            plt.close(fig)
            del fig
            if i < len(results) - 1:
                fig, axes = create_fig()

def describe_spikes(entry: PathEntry, config: dict) -> pd.DataFrame:
    """
    Calculate the slowdown features for spikes of a given file.
    """

    spikes = load_spike_shapes(entry.path, config['extractor'])

    features = []
    for i, spike in enumerate(spikes):
        print(f'Describing spike {i + 1}/{len(spikes)}', end='\r')
        feature = calculate_rising_features(spike, config['descriptor'])
        if feature is None:
            print(f'Spike {i + 1}\'s start can\'t be detected.')
            continue
        features.append(feature)
    print()

    output_path = Path(config['output_dir'])
    if config['descriptor']['single_spike_images']['do']:
        debug_dir = output_path / 'Figures' / 'Individual'
        if entry.subdir:
            debug_dir = debug_dir / entry.subdir
        debug_dir = debug_dir / entry.path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        plot_features = features
        if config['descriptor']['single_spike_images']['skip_zero']:
            plot_features = [
                f for f in features 
                if (
                    f['relative_dv_drop'] > 0 or
                    f['relative_rebound_time'] > 0
                )
            ]
        create_debug_images(
            plot_features,
            debug_dir,
            output_dpi=config['descriptor']['single_spike_images']['dpi']
        )
        print('Individual images created.')

    return convert_features_to_dataframe(features)

def describe_spikes_per_cell(cell_data: tuple[str, list[PathEntry]], config: dict) -> None:
    """Describe spikes for a cell spread across multiple files."""

    cell_name = cell_data[0]
    entries = cell_data[1]
    output_path = Path(config['output_dir'])

    all_features = []
    for entry in entries:
        print(f'Processing {entry.path.stem}...')
        features = describe_spikes(entry, config)
        features['cell_name'] = cell_name

        features['spike_type'] = entry.spike_type
        all_features.append(features)

    all_features_df = pd.concat(all_features, ignore_index=True)

    if entries[0].subdir is not None:
        all_features_df['cell_type'] = entries[0].subdir

    all_features_df.to_csv(
        output_path / 'Preprocessed' / f'{cell_name}_features.csv',
        index=False
    )

    binned_features = calculate_sparse_2d_histogram(
        all_features_df[['relative_dv_drop', 'relative_rebound_time']].values,
        all_features_df['spike_type'].values,
        bin_size=0.005
    )

    palette = config['palette']
    if config['coloring'] == 'cell_type':
        if entries[0].subdir is None:
            palette = None
        binned_features['label'] = entries[0].subdir

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        data=binned_features,
        x='x1',
        y='x2',
        size='ratio',
        hue='label',
        sizes=(10, 1000),
        palette=palette,
        ax=ax
    )
    ax.set_xlabel('Relative dV Drop')
    ax.set_ylabel('Relative Rebound Time')

    handles, labels = ax.get_legend_handles_labels()
    components_to_keep = [
        (handle, label) for handle, label
        in zip(handles, labels)
        if label in ['RAAP', 'AISAP']
    ]
    style_handles = [h for h, _ in components_to_keep]
    style_labels = [l for _, l in components_to_keep]
    ax.legend(
        style_handles,
        style_labels,
        title='Spike Type',
        loc='lower right'
    )
    ax.set_title(f'{cell_name} spike features')

    fig.savefig(
        output_path / 'Figures' / f'{cell_name}_features.png', dpi=300
    )
    plt.close(fig)



