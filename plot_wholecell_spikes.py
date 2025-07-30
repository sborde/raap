
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from analyses.feature_based import find_start_index
from analyses.preprocessing import prepare_data

RAW_INPUT_BASE = Path('E:/Data/BogiSejtek/PutRAAP/20250701')

RESULT_DIR = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results/20250702') /
    'InVitroJuxta' /
    'DvMaxNorm' /
    'Search_0.15-0.85' /
    'w11_p2'
)
FEATURE_TABLE = RESULT_DIR / 'invitro_juxta_dataset.csv'
MODE = 'Smoothed'

OUTPUT_DIR = (
    RESULT_DIR /
    'Figures' /
    'WholeJuxtaPairs'
) / MODE

X_THRESH = 0.01
Y_THRESH = 0.24

for PREDICTED_GROUP in ['zero', 'belowthresh', 'abovethresh']:
    for whole_group in ['RAAP', 'AISAP']:
        (OUTPUT_DIR / f'{PREDICTED_GROUP}_{whole_group}').mkdir(parents=True, exist_ok=True)

PALETTE = {
    'RAAP': {
        'juxta': '#ff5000',
        'whole': '#ff0000'
    },
    'AISAP': {
        'juxta': '#0000ff',
        'whole': '#008000'
    }
}

MEASUREMENT_PALETTE = {
    'juxta': {
        'RAAP': PALETTE['RAAP']['juxta'],
        'AISAP': PALETTE['AISAP']['juxta']
    },
    'whole': {
        'RAAP': PALETTE['RAAP']['whole'],
        'AISAP': PALETTE['AISAP']['whole']
    }
}

SCATTER_PALETTE = {
    'RAAP': '#ff5000',
    'AISAP': '#0000ff'
}

SPIKE_CONFIG = {
    'extractor': {
        'signal_name': 'wideband',
        'event_name': 'spk',
        'cut_window': [-0.001, 0.002],
        'fs': 40000
    },
    'preprocess': {
        'sg_filter': None,
        'normalize': None
    }
}

if MODE == 'Smoothed':
    SPIKE_CONFIG['preprocess']['sg_filter'] = {
        'window_length': 11,
        'poly_order': 2
    }

data = pd.read_csv(FEATURE_TABLE)
cell_names = data['cell_name'].unique()

for cell_name in cell_names:
    cell_data = data[data['cell_name'] == cell_name]
    cell_spikes = {
        'RAAP': {},
        'AISAP': {}
    }

    for spike_type in ['RAAP', 'AISAP']:
        for measurement in ['juxta', 'whole']:
            SUFFIX = f'{measurement}{spike_type}'
            cell_spikes[spike_type][measurement] = prepare_data(
                RAW_INPUT_BASE / measurement / f'{cell_name}_{SUFFIX}.mat',
                config=SPIKE_CONFIG
            )
    print('All spikes loaded for cell:', cell_name)
    # raap_like_spikes = cell_data.loc[
    #     np.logical_and(cell_data['relative_dv_drop'] > 0,
    #     cell_data['relative_rebound_time'] > 0)
    # ]

    for i, row in cell_data.iterrows():
        spike_type = row.spike_type

        juxta_spikes = cell_spikes[spike_type]['juxta']
        wholecell_spikes = cell_spikes[spike_type]['whole']

        closest_juxta_time = np.argmin(np.abs(
            juxta_spikes.timestamps - row.timestamp
        ))
        closest_wholecell_time = np.argmin(np.abs(
            wholecell_spikes.timestamps - row.timestamp
        ))

        print(
            f'Cell name: {cell_name}, Spike type: {spike_type}, ' +
            f'Timestamp: {row.timestamp:.3f}, ' +
            f'Juxta spike time: {juxta_spikes.timestamps[closest_juxta_time]:.3f}, ' +
            f'Wholecell spike time: {wholecell_spikes.timestamps[closest_wholecell_time]:.3f}'
        )

        wholecell_start_i = find_start_index(
            wholecell_spikes.segments.values[closest_wholecell_time]
        )
        wholecell_start_v = wholecell_spikes.segments.values[closest_wholecell_time][wholecell_start_i] * 1000

        juxta_start_i = find_start_index(
            juxta_spikes.segments.values[closest_juxta_time]
        )
        juxta_start_v = juxta_spikes.segments.values[closest_juxta_time][juxta_start_i]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(
            wholecell_spikes.segments.times * 1000,
            wholecell_spikes.segments.values[closest_wholecell_time] * 1000,
            '-',
            c=PALETTE[spike_type]['whole'],
            label=f'{spike_type} wholecell'
        )
        axes[0].hlines(
            wholecell_start_v,
            wholecell_spikes.segments.times[0] * 1000,
            wholecell_spikes.segments.times[-1] * 1000,
            colors='black',
            linestyles='--'
        )
        axes[0].vlines(
            wholecell_spikes.segments.times[wholecell_start_i] * 1000,
            wholecell_start_v,
            wholecell_spikes.segments.values[closest_wholecell_time].max() * 1000,
            colors='black',
            linestyles='--'
        )
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title(f'Wholecell startV={wholecell_start_v:.3f}mV')

        axes[1].plot(
            juxta_spikes.segments.times * 1000,
            juxta_spikes.segments.values[closest_juxta_time],
            '-',
            alpha=0.5,
            c=PALETTE[spike_type]['juxta']
        )
        axes[1].hlines(
            juxta_start_v,
            juxta_spikes.segments.times[0] * 1000,
            juxta_spikes.segments.times[-1] * 1000,
            colors='black',
            linestyles='--'
        )
        axes[1].vlines(
            juxta_spikes.segments.times[juxta_start_i] * 1000,
            juxta_start_v,
            juxta_spikes.segments.values[closest_juxta_time].max(),
            colors='black',
            linestyles='--'
        )
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Voltage (mV)')

        PROCESS_STRING = 'Raw'
        if MODE == 'Smoothed':
            SMOOTH_CONFIG = SPIKE_CONFIG['preprocess']['sg_filter']
            if not SMOOTH_CONFIG or not isinstance(SMOOTH_CONFIG, dict):
                raise ValueError('Smoothing filter configuration is missing.')
            PROCESS_STRING = (
                f'w={SMOOTH_CONFIG["window_length"]}' +
                f'p={SMOOTH_CONFIG["poly_order"]}'
            )

        axes[1].set_title(f'Juxta ({PROCESS_STRING}) startV={juxta_start_v}')

        sns.scatterplot(
            data=data,
            x='relative_dv_drop',
            y='relative_rebound_time',
            hue='spike_type',
            alpha=0.4,
            palette=SCATTER_PALETTE,
            ax=axes[2]
        )

        axes[2].plot(
            row.relative_dv_drop,
            row.relative_rebound_time,
            'o',
            markersize=10,
            c=SCATTER_PALETTE[spike_type]
        )
        axes[2].set_xlabel('Relative DV Drop')
        axes[2].set_ylabel('Relative Rebound Time')
        axes[2].set_title(
            f'rdv={row.relative_dv_drop:.3f}, ' +
            f'rrt={row.relative_rebound_time:.3f}'
        )

        if row.relative_dv_drop <= 0 and row.relative_rebound_time <= 0:
            PREDICTED_GROUP = 'zero'
        elif (
            row.relative_dv_drop <= X_THRESH or
            row.relative_rebound_time <= Y_THRESH
        ):
            PREDICTED_GROUP = 'belowthresh'
        else:
            PREDICTED_GROUP = 'abovethresh'

        plt.suptitle(f'{cell_name} {spike_type} t={row.timestamp:.3f}s')
        plt.tight_layout(pad=1.0)

        plt.savefig(
            (
                OUTPUT_DIR /
                f'{PREDICTED_GROUP}_{spike_type}' /
                f'{cell_name}_{spike_type}_{i:03d}.png'
            ),
            dpi=300
        )
        plt.close(fig)

# spikes = []
# for cell_name in cell_names:
#     cell_data = data[data['cell_name'] == cell_name]
#     cell_spikes = {
#         'RAAP': {},
#         'AISAP': {}
#     }

#     for spike_type in ['RAAP', 'AISAP']:
#         for measurement in ['juxta', 'whole']:
#             SUFFIX = f'{measurement}{spike_type}'
#             cell_spikes[spike_type][measurement] = prepare_data(
#                 RAW_INPUT_BASE / f'{SUFFIX}' / f'{cell_name}_{SUFFIX}.mat',
#                 config=SPIKE_CONFIG
#             )

#     print('All spikes loaded for cell:', cell_name)

#     for i, row in cell_data.iterrows():
#         spike_type = row.spike_type

#         juxta_spikes = cell_spikes[spike_type]['juxta']
#         wholecell_spikes = cell_spikes[spike_type]['whole']

#         closest_juxta_time = np.argmin(np.abs(
#             juxta_spikes.timestamps - row.timestamps
#         ))
#         closest_wholecell_time = np.argmin(np.abs(
#             wholecell_spikes.timestamps - row.timestamps
#         ))

#         wholecell_v =  wholecell_spikes.segments.values[closest_wholecell_time]
#         juxta_v = juxta_spikes.segments.values[closest_juxta_time]

#         wholecell_start_i = find_start_index(wholecell_v)
#         wholecell_start_v = wholecell_v[wholecell_start_i] * 1000

#         juxta_start_i = find_start_index(juxta_v)
#         juxta_start_v = juxta_v[juxta_start_i]

#         spikes.append({
#             'cell_name': cell_name,
#             'spike_type': spike_type,
#             'timestamp': row.timestamps,
#             'whole_start_v': wholecell_start_v,
#             'juxta_start_v': juxta_start_v
#         })

# start_voltages = pd.DataFrame(spikes)

# for measurement in ['whole', 'juxta']:
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     sns.boxplot(
#         data=start_voltages,
#         x='spike_type',
#         y=f'{measurement}_start_v',
#         hue='spike_type',
#         fill=False,
#         dodge=True,
#         width=0.3,
#         gap=0.2,
#         palette=MEASUREMENT_PALETTE[measurement],
#         ax=ax
#     )
#     plt.savefig(
#         OUTPUT_DIR / f'{measurement}_start_voltages_boxplot.png',
#         dpi=300
#     )
#     plt.close(fig)
