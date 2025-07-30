
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data.spikes import load_spike_shapes


SPIKE_EXTRACTOR_PARAM = {
    'signal_name': 'wideband',
    'event_name': 'spk',
    'cut_window': [-0.001, 0.002],
    'fs': 40000
}

SMOOTH_PARAMS = {
    'window_length': 7,
    'polyorder': 3
}

OUTPUT_DIR = Path('E:/Data/BogiSejtek/PutRAAP/Results/20250714/SmoothResidualOfAP')
INPUT_DIR = Path('E:/Data/BogiSejtek/PutRAAP/20250701/juxta')
FILE_LIST = {}
for file in INPUT_DIR.glob('*.mat'):
    FILE_NAME = file.stem
    CELL_NAME = '_'.join(FILE_NAME.split('_')[:2]).lower()
    FILE_LIST[CELL_NAME] = FILE_LIST.get(CELL_NAME, []) + [file.name]

# fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# for i, file in enumerate(TEST_FILES):
#     file_path = INPUT_DIR / file
#     print(f'Processing file: {file_path.name}')
#     spikes = load_spike_shapes(file_path, SPIKE_EXTRACTOR_PARAM)

#     avg_spike = np.mean(np.stack([spike.values for spike in spikes]), axis=0)
#     residuals = np.stack([avg_spike - spike.values for spike in spikes])

#     # axs[i, 0].fill_between(
#     #     spikes[0].times * 1000,
#     #     avg_spike - np.std(residuals, axis=0),
#     #     avg_spike + np.std(residuals, axis=0),
#     #     color='lightgray'
#     # )
#     for j in range(residuals.shape[0]):
#         axs[i, 0].scatter(
#             spikes[0].times * 1000,
#             residuals[j, :],
#             marker='.',
#             color='lightgray',
#             alpha=0.5
#         )

#     # residual_heatmap = np.zeros(shape=(40, 121))
#     # for j in range(residuals.shape[1]):
#     #     counts, _ = np.histogram(
#     #         residuals[:, j],
#     #         bins=np.linspace(-1, 1, 41)
#     #     )
#     #     residual_heatmap[:, j] = counts
#     # axs[i, 0].imshow(
#     #     residual_heatmap,
#     #     aspect='auto',
#     #     extent=[spikes[0].times[0] * 1000, spikes[0].times[-1] * 1000, -1, 1],
#     #     cmap='Blues',
#     #     origin='lower',
#     #     interpolation='nearest'
#     # )

#     axs[i, 0].plot(
#         spikes[0].times * 1000,
#         avg_spike,
#         color='black'
#     )
#     axs[i, 0].set_title(f'Average Spike Shape - {file_path.name}')

#     axs[i, 1].hist(
#         residuals.flatten(),
#         bins=30,
#         color='blue',
#         alpha=0.7
#     )

# plt.show()

for cell_name, file_list in FILE_LIST.items():
    print(f'Processing cell: {cell_name}')

    cell_results = []
    for file in file_list:
        file_path = INPUT_DIR / file
        print(f'Processing file: {file_path.name}')
        spikes = load_spike_shapes(file_path, SPIKE_EXTRACTOR_PARAM)

        results = []
        best_snr_spike = {'snr': -np.inf}
        worst_snr_spike = {'snr': np.inf}
        sum_snr = 0
        pool_residuals = []
        for j, spike in enumerate(spikes):
            v = spike.values

            smoothed_v = savgol_filter(
                v,
                SMOOTH_PARAMS['window_length'],
                SMOOTH_PARAMS['polyorder']
            )
            signal_power = np.mean(smoothed_v ** 2)

            residuals = v - smoothed_v
            noise_power = np.mean(residuals ** 2)
            snr = signal_power / noise_power if noise_power > 0 else np.inf

            smooth_data = {
                'times': spike.times,
                'raw': v,
                'smoothed': smoothed_v,
                'spike_id': j,
                'snr': snr
            }

            if snr > best_snr_spike['snr']:
                best_snr_spike = smooth_data
            if snr < worst_snr_spike['snr']:
                worst_snr_spike = smooth_data

            sum_snr += snr
            pool_residuals.append(residuals)

        cell_results.append(
            {
                'file': file_path.name,
                'smooth_params': SMOOTH_PARAMS,
                'best_spike': best_snr_spike,
                'worst_spike': worst_snr_spike,
                'average_snr': 10 * np.log10(sum_snr / len(spikes)),
                'residuals': np.concatenate(pool_residuals)
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for i, cell_result in enumerate(cell_results):
        for j, spike_data in enumerate([
                cell_result['best_spike'],
                cell_result['worst_spike']
        ]):
            axes[i, j].plot(
                spike_data['times'] * 1000,
                spike_data['raw'],
                label='Raw Spike',
                color='#afafaf'
            )
            axes[i, j].plot(
                spike_data['times'] * 1000,
                spike_data['smoothed'],
                label='Smoothed Spike',
                color='orange'
            )
            axes[i, j].set_title(
                f"{'Best' if j == 0 else 'Worst'} Spike, "
                f"SNR: {10 * np.log10(spike_data['snr']):.2f} dB"
            )

        axes[i, 2].hist(
            cell_result['residuals'],
            bins=10,
            alpha=0.7
        )
        axes[i, 2].set_title(
            f"Residuals: {np.mean(cell_result['residuals']):.4f} " +
            r"$\pm$" +
            f"{np.std(cell_result['residuals']):.4f} \n" +
            f"Average SNR: {cell_result['average_snr']:.2f} dB"
        )
        if i > 0:
            axes[i, 2].sharex(axes[0, 2])

    plt.suptitle(
        f'{file_list[0]} and {file_list[1]} - ' +
        'Smoothing Parameters: ' +
        f'w_{SMOOTH_PARAMS["window_length"]}, ' +
        f'p_{SMOOTH_PARAMS["polyorder"]}'
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        OUTPUT_DIR / f'{cell_name}_smoothing_residuals.png',
        dpi=300
    )
    plt.close(fig)
