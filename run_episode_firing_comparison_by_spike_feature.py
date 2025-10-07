
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon

from common.data_utils import CellWrapper
from triggered_firing.io_utils import list_result_files
from workflows.run_util.config import load_yaml
from workflows.run_util.io import get_cell_info_list

CONFIG_NAME = 'episode_firing_comparison'

if __name__ == "__main__":

    config = load_yaml(f'{CONFIG_NAME}.yaml')

    spike_types = config['params']['spike_types']
    dataset_name = config['input']['dataset']
    data_config = config['data'][dataset_name]

    results_dir = Path(data_config['results']['path']) / 'EpisodeFiringBySpikeType'
    input_paths = [
        CellWrapper(
            info,
            (
                results_dir /
                info.cell_type /
                f'{info.name}_episode_firing_rate_by_spiketype.xlsx'
            )
        ) for info in get_cell_info_list(config['input'])
    ]
    frames = [
        CellWrapper(input_path.info, pd.read_excel(input_path.data))
        for input_path in input_paths
    ]

    aggregation_name = config['params']["aggregation"]
    aggregate = np.nanmean if aggregation_name == 'mean' else np.nanmedian

    result_save_path = results_dir / 'GroupComparison' / 'Results'
    result_save_path.mkdir(parents=True, exist_ok=True)
    figure_save_path = results_dir / 'GroupComparison' / 'Figures'
    figure_save_path.mkdir(parents=True, exist_ok=True)

    aggregated_results = []
    for cell in frames:
        frame = cell.data
        frame_aggregates = []
        for episode in config['params']['episodes']:
            subset = frame[frame['episode'] == episode]
            episode_result = {
                'name': cell.info.name,
                'group': cell.info.cell_type,
                'episode': episode
            }
            for spike_type in spike_types:
                episode_result[spike_type] = aggregate(subset[spike_type])
            frame_aggregates.append(episode_result)

        cell_result = pd.DataFrame(frame_aggregates)
        aggregated_results.append(cell_result)

    result_df = pd.concat(aggregated_results)
    result_df.to_excel(
        result_save_path / f'cellwise_aggregate.xlsx', index=False
    )
    result_df['episode'] = pd.Categorical(
        result_df['episode'], categories=config['params']['episodes'], ordered=True
    )

    for group in result_df['group'].unique():
        group_df = result_df[result_df['group'] == group]
        fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)
        for i, spike_type in enumerate(spike_types):
            ax = axes[i]
            sns.boxplot(
                x='episode',
                y=spike_type,
                fill=False,
                fliersize=0,
                color='#808080',
                data=group_df,
                ax=ax
            )
            res = wilcoxon(
                result_df[result_df['episode'] == 'sw'][spike_type],
                result_df[result_df['episode'] == 'rem'][spike_type]
            )
            for name in group_df['name'].unique():
                subset = group_df[group_df['name'] == name]
                ax.plot(
                    subset['episode'],
                    subset[spike_type],
                    markersize=10,
                    marker='o',
                    linestyle='-',
                    label=name,
                    linewidth=2,
                )

            ax.set_title(f"{spike_type}, p={res.pvalue:.3f}") # type: ignore
            ax.set_ylabel("Mean firing rate (Hz)")
        plt.suptitle(f"{group} cells")
        plt.tight_layout()

        figure_name = f'{group}_episode_firing_comparison'
        fig.savefig(
            figure_save_path / f'{figure_name}.png',
            dpi=config['output']['dpi'],
            format='png'
        )
        if config['output']['make_svg']:
            fig.savefig(
                figure_save_path / f'{figure_name}.svg',
                format='svg'
            )
        plt.close(fig)
