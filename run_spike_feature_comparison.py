
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon

from triggered_firing.io_utils import list_input_files, list_result_files, load_customized_config


CONFIG_NAME = 'spike_feature_comparison'

if __name__ == "__main__":

    with open(f"{CONFIG_NAME}.yaml", 'r') as file:
        # Load the configuration file
        run_config = yaml.safe_load(file)

    input_files = list_result_files(
        run_config['data'][run_config['input']['dataset']],
        run_config['input']
    )
    frames = [
        (input_file.name, pd.read_csv(input_file.data)) for input_file in input_files
    ]

    for feature in run_config['params']['plot_feature_list']:
        aggregated_results = []
        for name, frame in frames:
            for spike_type in frame['spike_type'].unique():
                subset = frame[frame['spike_type'] == spike_type]
                aggregated_results.append({
                    'name': name,
                    'mean': subset[feature].mean(),
                    'spike_type': spike_type
                })
        result_df = pd.DataFrame(aggregated_results)
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x='spike_type',
            y='mean',
            fill=False,
            fliersize=0,
            color='#808080',
            data=result_df
        )
        res = wilcoxon(
            result_df[result_df['spike_type'] == 'AIS-AP']['mean'],
            result_df[result_df['spike_type'] == 'RA-AP']['mean']
        )
        for name in result_df['name'].unique():
            subset = result_df[result_df['name'] == name]
            plt.plot(
                subset['spike_type'],
                subset['mean'],
                markersize=10,
                marker='o',
                linestyle='-',
                label=name,
                linewidth=2,
            )

        plt.title(f'Comparison of {feature}, p={res.pvalue:.3f}')
        plt.ylabel(f'Mean {feature}')
        plt.show()

    # if run_config['grouping']['do']:
    #     run_config['output']['suffix'] = analysis.suffix + '_histogram'
    #     run_grouped_analysis(run_config)
