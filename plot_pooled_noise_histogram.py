"""
A simple script that plots the pooled RMSD histogram
of all cells in a given directory.
 - 2025. 10. 03. - initial version
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIR_PATH = 'E:/Data/BogiSejtek/PutRAAP/Results/20250804/Preprocessed/RSI_L2-3'
BIN_RANGE = (0, 0.005)
BIN_COUNT = 100
HIST_COLOR = '#808080'

if __name__ == "__main__":

    group_name = Path(DIR_PATH).name
    save_path = Path(DIR_PATH).parent
    file_paths = list(Path(DIR_PATH).rglob('*.csv'))

    frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        df['noise_ratio'] = df['noise_power'] / df['raw_power'] * 100
        df['rmsd'] = np.sqrt(df['noise_power'])
        df['name'] = path.stem
        frames.append(df)
    pool_df = pd.concat(frames, ignore_index=True)
    pool_df.to_csv(
        save_path / 'Pooled_Noise_Characteristics.csv', index=False
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(
        pool_df['noise_ratio'],
        bins=BIN_COUNT,
        range=BIN_RANGE,
        color=HIST_COLOR,
        alpha=0.7
    )
    ax.set_title(
        f"{group_name} - Noise to Raw Power Ratio (%)"
    )
    fig.savefig(save_path / f'{group_name}_Pooled_Noise_Histogram.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.boxplot(
        pool_df['noise_ratio'],
        vert=False,
        boxprops=dict(color=HIST_COLOR)
    )
    ax.set_title(
        f"{group_name} - Noise to Raw Power Ratio (%)"
    )
    fig.savefig(save_path / f'{group_name}_Pooled_Noise_Boxplot.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, frame in enumerate(frames):
        ax.plot(
            frame['noise_ratio'],
            np.full_like(frame['noise_ratio'], i),
            'o'
        )
    ax.set_xlabel('Noise to Raw Power Ratio (%)')
    ax.set_yticks(np.arange(len(frames)))
    ax.set_yticklabels([f['name'].iloc[0] for f in frames])

    fig.savefig(
        save_path / f"{group_name}_noise_ratio_scatter.svg",
        format='svg'
    )
    plt.close(fig)
