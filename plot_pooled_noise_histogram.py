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
NOISE_THRESHOLD = 0.0025

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
    pool_df.to_excel(
        save_path / 'Pooled_Noise_Characteristics.xlsx', index=False
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if NOISE_THRESHOLD is None:
        ax.hist(
            pool_df['noise_ratio'],
            bins=BIN_COUNT,
            range=BIN_RANGE,
            color=HIST_COLOR,
            alpha=0.7
        )
    else:
        flags = [
            pool_df['noise_ratio'] <= NOISE_THRESHOLD,
            pool_df['noise_ratio'] > NOISE_THRESHOLD
        ]
        for flag in flags:
            if not np.any(flag):
                continue
            ax.hist(
                pool_df[flag]['noise_ratio'],
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
    if NOISE_THRESHOLD is not None:
        ax.axvline(
            NOISE_THRESHOLD,
            color='r',
            linestyle='--',
            label='Noise Threshold'
        )
        ax.legend()
    ax.set_xlabel('Noise to Raw Power Ratio (%)')
    ax.set_yticks(np.arange(len(frames)))
    ax.set_yticklabels([f['name'].iloc[0] for f in frames])

    fig.savefig(
        save_path / f"{group_name}_noise_ratio_scatter.svg",
        format='svg'
    )
    plt.close(fig)

    aggregates = []
    for frame in frames:
        iqr = np.diff(np.percentile(frame['noise_ratio'], [25, 75]))
        outlier_threshold = np.percentile(frame['noise_ratio'], 75) + 1.5 * iqr
        frame_aggregate = pd.DataFrame({
            'name': frame['name'].iloc[0],
            'mean_noise_ratio': frame['noise_ratio'].mean(),
            'std_noise_ratio': frame['noise_ratio'].std(),
            'median_noise_ratio': frame['noise_ratio'].median(),
            'q1_noise_ratio': np.percentile(frame['noise_ratio'], 25),
            'q3_noise_ratio': np.percentile(frame['noise_ratio'], 75),
            'outlier_threshold': outlier_threshold,
            'outlier_ratio': np.mean(frame['noise_ratio'] > outlier_threshold)
        })
        if NOISE_THRESHOLD is not None:
            frame_aggregate['above_threshold_ratio'] = np.mean(
                frame['noise_ratio'] > NOISE_THRESHOLD
            )
        aggregates.append(frame_aggregate)
    aggregate_df = pd.concat(aggregates, ignore_index=True)
    aggregate_df.to_excel(
        save_path / 'Cellwise_Noise_Aggregates.xlsx', index=False
    )
