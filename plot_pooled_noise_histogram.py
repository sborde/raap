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
    pool_df.to_csv(
        save_path / f'{group_name}_Pooled_Noise_Characteristics.csv',
        index=False
    )

    if np.any(np.isnan(pool_df['noise_ratio'])):
        n_nans = np.sum(pool_df['noise_ratio'].isna())
        nan_ratio = n_nans / len(pool_df) * 100
        print(f"Number of NaNs: {n_nans}/{len(pool_df)} ({nan_ratio:.2f}%)")
        print(pool_df.loc[pool_df['noise_ratio'].isna()])
        pool_df = pool_df.loc[~pool_df['noise_ratio'].isna()]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].boxplot(
        pool_df['noise_ratio'],
        vert=False,
        boxprops={'color': HIST_COLOR},
        showfliers=False
    )
    ax[0].set_title(
        f"{group_name} - Noise-to-Raw Power Ratio (%)"
    )

    pool_q3 = np.percentile(pool_df['noise_ratio'], 75)
    pool_iqr = np.diff(np.percentile(pool_df['noise_ratio'], [25, 75]))
    outlier_threshold = pool_q3 + 1.5 * pool_iqr

    if NOISE_THRESHOLD is None:
        ax[1].hist(
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
            ax[1].hist(
                pool_df[flag]['noise_ratio'],
                bins=BIN_COUNT,
                range=BIN_RANGE,
                color=HIST_COLOR,
                alpha=0.7
            )
    ax[1].axvline(
        outlier_threshold,
        color='orange',
        linestyle='--'
    )

    fig.savefig(save_path / f'{group_name}_Pooled_Noise_Distribution.png')
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
    aggregate_df.to_csv(
        save_path / 'Cellwise_Noise_Aggregates.csv', index=False
    )
