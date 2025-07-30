

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

if __name__ == "__main__":
    parser = ArgumentParser(description='Plot firing rates and halfwidth scatter.')
    parser.add_argument(
        'config',
        type=str,
        help='Path to the config file.'
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load data
    data = pd.read_excel(config['input']['path'])
    data['cell_type'] = data['group'].apply(lambda x: x.split('_')[0])
    data['layer'] = data['group'].apply(lambda x: x.split('_')[1])

    group_sizes = data['group'].value_counts()
    print(f"Group sizes:\n{group_sizes}\n")

    figsize = (2 * config['plot']['axis_size'][0], config['plot']['axis_size'][1])
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for i, layer in enumerate(data['layer'].unique()):
        layer_data = data[data['layer'] == layer]
        sns.scatterplot(
            data=layer_data,
            x='halfwidth',
            y='frequency',
            hue='group',
            style='cell_type',
            ax=axs[i],
            palette=config['markers']['colors'],
            markers=config['markers']['shapes'],
            s=config['plot']['point_sizes'],
            edgecolor=None  # Remove edge color
        )

        handles, labels = axs[i].get_legend_handles_labels()
        components_to_keep = [
            (handle, label) for handle, label
            in zip(handles, labels)
            if label in config['markers']['colors']
        ]
        style_handles = [h for h, _ in components_to_keep]
        style_labels = [
            f"{l.split('_')[0]} (n={group_sizes[l]})"
            for _, l in components_to_keep
        ]

        axs[i].legend(
            style_handles,
            style_labels,
            title='Cell type',
            loc='lower right'
        )

        axs[i].set_xlim(config['plot']['axis_params']['limits']['x'])
        axs[i].set_ylim(config['plot']['axis_params']['limits']['y'])
        axs[i].set_title(f'Layer {layer}')
        axs[i].set_xlabel('Halfwidth (ms)')
        axs[i].set_ylabel('Firing Rate (Hz)')
        axs[i].set_xscale('log' if config['plot']['axis_params']['log_scale']['x'] else 'linear')
        axs[i].set_yscale('log' if config['plot']['axis_params']['log_scale']['y'] else 'linear')

    plt.tight_layout()
    extensions = ['png']
    if config['output']['do_svg']:
        extensions.append('svg')
    for ext in extensions:
        plt.savefig(
            (
                Path(config['output']['path']) /
                f"{config['output']['name']}.{ext}"
            ),
        dpi=config['output']['dpi']
    )
    plt.close(fig)
