"""A module for slowdown feature visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotters import calculate_sparse_2d_histogram

def visualize_slowdown_features(
    data: pd.DataFrame,
    config: dict
) -> None:
    """
    Visualize the slowdown features of the spikes in the dataset.
    """

    figure_path = (
        Path(config['output_dir']) /
        'Figures' /
        config['run_mode'].capitalize()
    )
    palette = config['palette']

    if config['modes']['distscatter']:

        save_path = figure_path / 'DistScatter'
        save_path.mkdir(parents=True, exist_ok=True)

        if config['coloring'] == 'cell_type':
            if data['cell_type'].isnull().all():
                raise ValueError(
                    'No spike type information available for coloring.'
                )
            labels = data['cell_type'].values
        elif config['coloring'] == 'spike_type':
            labels = data['spike_type'].values
        else:
            labels = np.zeros(data.shape[0], dtype=str)
            palette = None

        binned_features = calculate_sparse_2d_histogram(
            data[['relative_dv_drop', 'relative_rebound_time']].values,
            labels,
            bin_size=0.005
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(
            data=binned_features,
            x='x1',
            y='x2',
            size='ratio',
            hue='label',
            sizes=(50, 1000),
            palette=palette,
            linewidth=0,
            ax=ax
        )

        if config['axis_limits']:
            ax.set_xlim(
                config['axis_limits']['x'][0],
                config['axis_limits']['x'][1]
            )
            ax.set_ylim(
                config['axis_limits']['y'][0],
                config['axis_limits']['y'][1]
            )

        if config['thresholds']:
            th_x = config['thresholds']['x']
            th_y = config['thresholds']['y']
            ax.hlines(
                th_y,
                th_x,
                ax.get_xlim()[1],
                color='black',
                linestyle='--'
            )
            ax.vlines(
                th_x,
                th_y,
                ax.get_ylim()[1],
                color='black',
                linestyle='--'
            )

        ax.set_xlabel('Relative dV Drop')
        ax.set_ylabel('Relative Rebound Time')

        if config['coloring'] == 'spike_type':
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
        else:
            ax.get_legend().remove()

        ax.set_title(config['plot_title'])

        for output_format in config['output_format']:
            plt.savefig(
                (
                    save_path /
                    f'{config['plot_title']}.{output_format}'
                ),
                dpi=300
            )
        plt.close(fig)

    if config['modes']['tileplot']:
        fig, ax = plt.subplots(
            1, 2, figsize=(10, 5), sharex=True, sharey=True
        )
        for i, spike_type in enumerate(['AISAP', 'RAAP']):
            sns.histplot(
                data=data[data['spike_type'] == spike_type],
                x='relative_dv_drop',
                y='relative_rebound_time',
                binwidth=0.005,
                stat='probability',
                cbar=True,
                vmin=0,
                vmax=1.0,
                color=palette[spike_type],
                ax=ax[i]
            )

            ax[i].set_title(spike_type)
            ax[i].set_xlabel('Relative dV Drop')
            ax[i].set_ylabel('Relative Rebound Time')

        plt.savefig(figure_path / 'slowdown_features_tileplot.png', dpi=300)
        plt.close(fig)

    if config['modes']['contour']:
        fig, ax = plt.subplots(
            1, 2, figsize=(10, 5), sharex=True, sharey=True
        )
        for i, spike_type in enumerate(['AISAP', 'RAAP']):
            sns.kdeplot(
                data=data[data['spike_type'] == spike_type],
                x='relative_dv_drop',
                y='relative_rebound_time',
                color=palette[spike_type],
                fill=False,
                levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                ax=ax[i]
            )

            ax[i].set_title(spike_type)
            ax[i].set_xlabel('Relative dV Drop')
            ax[i].set_ylabel('Relative Rebound Time')

        plt.savefig(figure_path / 'slowdown_features_contour.png', dpi=300)
        plt.close(fig)