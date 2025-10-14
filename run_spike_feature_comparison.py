
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon

from triggered_firing.io_utils import list_result_files


CONFIG_NAME = 'spike_feature_comparison'

if __name__ == "__main__":

    with open(f"{CONFIG_NAME}.yaml", 'r', encoding='utf-8') as file:
        # Load the configuration file
        config = yaml.safe_load(file)

    params = config['params']
    spike_types = params['spike_types']
    aggregation_name = params["aggregation"]
    dataset_name = config['input']['dataset']
    input_files = list_result_files(
        config['data'][dataset_name],
        config['input']
    )
    frames = [
        (input_file.name, pd.read_csv(input_file.data)) for input_file in input_files
    ]

    if aggregation_name == 'median':
        aggregate = np.nanmedian
    else:
        aggregate = np.nanmean

    result_save_path = (
        Path(config['data'][dataset_name]['results']['path']) /
        'Results'
    )
    result_save_path.mkdir(parents=True, exist_ok=True)
    figure_save_path = (
        Path(config['data'][dataset_name]['results']['path']) /
        'Figures'
    )
    figure_save_path.mkdir(parents=True, exist_ok=True)

    for feature in params['plot_features']:
        aggregated_results = []
        for name, frame in frames:
            frame_spikes = []
            for spike_type in spike_types:
                subset = frame[frame['spike_type'] == spike_type]

                if len(subset) == 0:
                    continue

                frame_spikes.append({
                    'name': name,
                    'value': aggregate(subset[feature['name']]),
                    'spike_type': spike_type
                })
            cell_result = pd.DataFrame(frame_spikes)
            if (
                'AIS-AP' in cell_result['spike_type'].values and
                'RA-AP' in cell_result['spike_type'].values
            ):
                aggregated_results.append(cell_result)
            else:
                print(f"Skipping {name} due to missing spike types")
        result_df = pd.concat(aggregated_results)
        result_df.to_excel(
            result_save_path / f'{feature["name"]}_comparison.xlsx', index=False
        )
        result_df['spike_type'] = pd.Categorical(
            result_df['spike_type'], categories=spike_types, ordered=True
        )

        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(
            x='spike_type',
            y='value',
            fill=False,
            fliersize=0,
            color='#808080',
            data=result_df
        )
        res = wilcoxon(
            result_df[result_df['spike_type'] == 'AIS-AP']['value'],
            result_df[result_df['spike_type'] == 'RA-AP']['value']
        )
        for name in result_df['name'].unique():
            subset = result_df[result_df['name'] == name]
            plt.plot(
                subset['spike_type'],
                subset['value'],
                markersize=10,
                marker='o',
                linestyle='-',
                label=name,
                linewidth=2,
            )
        plt.ylim(feature['limit'])
        plt.title(f'Comparison of {feature["name"]}, p={res.pvalue:.3f}') # type: ignore
        plt.ylabel(f'{aggregation_name.capitalize()} {feature["name"]}')
        figure_name = f'{aggregation_name}_{feature["name"]}_comparison'
        plt.savefig(
            figure_save_path / f'{figure_name}.png',
            dpi=config['output']['dpi'],
            format='png'
        )
        if config['output']['make_svg']:
            with plt.rc_context({'svg.fonttype': 'none'}):
                plt.savefig(
                    figure_save_path / f'{figure_name}.svg',
                    format='svg'
                )
        plt.close(fig)
