
from pathlib import Path

import yaml
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.file_utils import load_config



if __name__ == "__main__":

    config = load_config(
        'plot_spike_type_distribution_over_episodes_config.yaml'
    )

    episode_palette = config['plot']['episode_palette']
    spike_types = config['plot']['spike_types']
    figures_path = Path(config['output']['path'])
    input_dir = Path(config['input']['path'])

    for file_path in input_dir.glob('*_averages.csv'):
        cell_type = file_path.stem.replace('_averages', '')
        stats_df = pd.read_csv(file_path, index_col=None)

        for episode_group in config['plot']['episode_groups']:


            parent = episode_group.get('parent', 'all')
            group_df = stats_df.copy()
            if parent != 'all':
                group_df = group_df[group_df['episode'] == parent]

            episode_group_name = episode_group['name']

            episode_names = episode_group['episodes'] + ['none']
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for i, spike_type in enumerate(spike_types):
                spike_stats = group_df[group_df['spike_type'] == spike_type]
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
                title=episode_group_name.capitalize(),
                loc='upper left',
                bbox_to_anchor=(1.0, 1),
                borderaxespad=0.
            )

            plt.xticks(
                np.arange(len(spike_types)) + 1,
                spike_types
            )
            plt.xlabel('Spike Type')
            plt.ylabel('Ratio')
            plt.title(f'{episode_group_name.capitalize()} ratio per Spike Type')
            plt.tight_layout()
            fig.savefig(
                figures_path /
                f"{cell_type}_group_spiketype_{episode_group_name}_ratio.png",
                dpi=100
            )
            plt.close(fig)
