
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.data_utils import CellInfo, CellPath
from signalizer.model.experiment.sequences import EpisodeSequence
from utils.file_utils import load_config, enumerate_input_files
from perform_full_wavelet import calculate_tfd_power
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

def get_preprocessed_path(
    preprocessed_dir: Path,
    cell_info: CellInfo
) -> CellPath:
    """
    Creates the preprocessed path for the given raw cell.
    Args:
        raw_cell (CellInfo): The raw cell information.
        preprocessed_dir (Path): The directory where preprocessed files are stored.
    Returns:
        CellPath: The path to the preprocessed file.
    """
    return CellPath(
        cell_info,
        (
            preprocessed_dir /
            cell_info.cell_type / f"{cell_info.name}.csv"
        )
    )

def create_episode_mask(
    timestamps: np.ndarray,
    episodes: EpisodeSequence
) -> np.ndarray:
    """
    Returns a boolean mask indicating for each timestamp whether
    it falls within any episode in the episode sequence.

    Args:
        timestamps (np.ndarray): The array containing spike timestamps.
        episodes (EpisodeSequence): The episode sequence to check against.

    Returns:
        np.ndarray: A boolean mask indicating whether each timestamp is within any episode.
    """
    masks = []
    for episode in episodes:
        masks.append(
            (episode.interval.start <= timestamps) &
            (timestamps <= episode.interval.end)
        )
    return np.logical_or.reduce(masks)

def mark_episode(
    timestamps: np.ndarray,
    episode_name: str,
    recording: MatRecording
) -> np.ndarray:
    """
    Returns a binary integer array indicating whether each timestamp
    belongs to the specified episode.

    Args:
        timestamps (np.ndarray): The array containing spike timestamps.
        episode_name (str): The name of the episode to mark.
        recording (MatRecording): The MatRecording object containing episode data.
    Returns:
        np.ndarray: a binary array indicating episode membership or NaN
        if the episode is not present.
    """
    if f'{episode_name}_st' not in recording.list_markers():
        return np.full(len(timestamps), np.nan)
    episodes = recording.get_episode(episode_name)
    return create_episode_mask(timestamps, episodes).astype(int)

def assign_interval_to_spike(
    recording: MatRecording,
    spikes: pd.DataFrame,
    episode_groups: list[dict]
) -> pd.DataFrame:
    """
    Assigns the interval's name in which the spike occurs.
    Args:
        recording (MatRecording): The MatRecording object containing episode data.
        spikes (pd.DataFrame): The DataFrame containing spike features.
    Returns:
        pd.DataFrame: The DataFrame with episode memberships added.
    """
    timestamps = spikes['timestamp'].to_numpy()
    for episode_group in episode_groups:
        group_name = episode_group['name']
        spikes[group_name] = 'none'
        for episode_name in episode_group['episodes']:
            mask = mark_episode(
                timestamps,
                episode_name,
                recording
            )
            spikes[episode_name] = mask
            if np.isnan(mask).all():
                continue
            spikes.loc[mask == 1, group_name] = episode_name
    return spikes

if __name__ == "__main__":

    config = load_config(
        'spike_distribution_over_sleep.yaml'
    )

    raw_inputs = enumerate_input_files(config)

    output_path = Path(config['output']['path'])
    results_path = output_path / 'Results'
    results_path.mkdir(parents=True, exist_ok=True)

    figures_path = output_path / 'Figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    episode_palette = config['plot']['spiketype_sleep_barchart']['palette']

    for raw_cell_path in raw_inputs:
        print(f"Processing cell: {raw_cell_path.name}")
        preprocessed_cell_path = get_preprocessed_path(
            Path(config['preprocessed']['path']),
            raw_cell_path.info
        )

        recording = MatRecording(raw_cell_path.data)
        spikes = load_spike_features(preprocessed_cell_path, config['analysis'])

        spike_types = spikes['spike_type'].cat.categories

        spikes = assign_interval_to_spike(
            recording,
            spikes.copy(),
            config['analysis']['episode_assignment']['episode_groups']
        )
        spikes.to_csv(
            results_path / f'{raw_cell_path.info.name}_with_episodes.csv',
            index=False
        )

        episode_groups = config['analysis']['episode_assignment']['episode_groups']
        for episode_group in episode_groups:
            group_name = episode_group['name']
            episode_names = episode_group['episodes'] + ['none']
            stats = []
            for spike_type in spike_types:
                subset = spikes[spikes['spike_type'] == spike_type]
                for episode_name in episode_names:
                    count = np.sum(subset[group_name] == episode_name)
                    stats.append({
                        'spike_type': spike_type,
                        'episode': episode_name,
                        'ratio': (
                            count / len(subset) if len(subset) > 0 else 0
                        )
                    })
            stats_df = pd.DataFrame(stats)
            fig, ax = plt.subplots(figsize=(7.5, 5))

            for i, spike_type in enumerate(spike_types):
                spike_stats = stats_df[stats_df['spike_type'] == spike_type]
                bottom = 0
                for j, episode_name in enumerate(episode_names):
                    episode_stats = spike_stats[spike_stats['episode'] == episode_name]
                    if episode_stats.empty:
                        continue
                    ratio = episode_stats['ratio'].values
                    ax.fill_between(
                        [i + 0.75, i + 1.25],
                        bottom,
                        bottom + ratio,
                        color=episode_palette[episode_name],
                        label=f'{episode_name}'
                    )
                    bottom += ratio

            handles, labels = ax.get_legend_handles_labels()
            handle_dict = {}
            for handle, label in zip(handles, labels):
                if label not in handle_dict:
                    handle_dict[label] = handle

            # Get unique handles and labels
            unique_handles = list(handle_dict.values())
            unique_labels = list(handle_dict.keys())

            ax.legend(
                unique_handles,
                unique_labels,
                title=group_name.capitalize(),
                loc='upper left',
                bbox_to_anchor=(1.0, 1),
                borderaxespad=0.
            )

            plt.xticks(
                np.arange(len(spike_types)) + 1,
                spike_types.tolist()
            )
            plt.xlabel('Spike Type')
            plt.ylabel('Ratio')
            plt.title(f'{group_name.capitalize()} ratio per Spike Type')
            plt.tight_layout()
            fig.savefig(
                figures_path /
                f"{raw_cell_path.info.name}_spiketype_{group_name}_ratio.png",
                dpi=100
            )
            plt.close(fig)

        continue
        fig, axes = plt.subplots(
            2 + len(spike_types), 1, figsize=(15, 8),
            sharex=True, layout='constrained'
        )

        wavelet_params = config['plot']['sleep_distribution']['spectrogram']['params']
        signal = recording.get_waveform('wideband')
        signal = signal.resampled(1000)
        spectrogram = calculate_tfd_power(
            signal.values,
            si=signal.si,
            params=wavelet_params
        )
        im = axes[0].pcolormesh(
            signal.times,
            np.arange(
                wavelet_params['low'],
                wavelet_params['high'] + wavelet_params['step'],
                wavelet_params['step']
            ),
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

        event_styles = config['plot']['sleep_distribution']['styles']
        for i, style in enumerate(event_styles):
            subset = spikes[spikes['spike_type'] == style['name']]
            if style['color'] == 'cell_type':
                color = config['plot']['palette'][raw_cell_path.info.cell_type]
            else:
                color = style['color']

            y_values = np.full(len(subset), i)
            if subset.empty:
                continue
            timestamps = subset['timestamp'].to_numpy()
            axes[1].vlines(
                timestamps,
                y_values - 0.45,
                y_values + 0.45,
                color=color,
                alpha=1.0
            )

            if config['plot']['sleep_distribution']['histogram']['mode'] == 'instantaneous':
                axes[-1 - i].plot(
                    timestamps[1:],
                    1 / np.diff(timestamps),
                    color=color,
                    alpha=1.0
                )
                axes[-1 - i].set_ylabel('Inst. Freq. (Hz)')
        axes[1].set_yticks(range(len(event_styles)))
        axes[1].set_yticklabels([style['name'] for style in event_styles])
        axes[1].set_ylim((-1, len(event_styles)))
        

        axes[-1].set_xlabel('Time (s)')
        fig.savefig(
            figures_path /
            f"{raw_cell_path.info.name}_sleep_distribution.png",
            dpi=100
        )
        plt.close(fig)
