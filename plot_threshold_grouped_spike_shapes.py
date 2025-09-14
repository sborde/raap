
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyses.preprocessing import prepare_data
from raapy.workflows.processors.spike_transformers import find_start_index

TH_X = 0.01
TH_Y = 0.24

PLOT_CONFIG = {
    'superimposed': {
        'color': '#dedede',
        'alpha': 0.3,
        'linewidth': 0.5,
        'linestyle': '-'
    },
    'average': {
        'color': '#000000',
        'alpha': 1.0,
        'linewidth': 2,
        'linestyle': '-'
    }
}

CATEGORIES = [
    {
        'name': 'AIS-AP',
        'low': [-np.inf, -np.inf],
        'high': [0, 0],
        'linestyle': 'solid'
    },
    {
        'name': 'UNKNOWN',
        'low': [0, 0],
        'high': [TH_X, TH_Y],
        'linestyle': 'solid'
    },
    {
        'name': 'RA-AP',
        'low': [TH_X, TH_Y],
        'high': [np.inf, np.inf],
        'linestyle': 'solid'
    }
]

CELL_TYPES = {
    'FSI_L2-3': {
        'name': 'FSI_L2-3',
        'color': '#d54500ff',
        'xlim': [-1, 1.5]
    },
    'FSI_L4-6': {
        'name': 'FSI_L4-6',
        'color': '#7a0000ff',
        'xlim': [-1, 1.5]
    },
    'RSI_L2-3': {
        'name': 'RSI_L2-3',
        'color': '#ff48a3ff',
        'xlim': [-1, 1.5]
    },
    'RSI_L4-6': {
        'name': 'RSI_L4-6',
        'color': '#800080ff',
        'xlim': [-1, 1.5]
    },
    'PYR_L2-3': {
        'name': 'PYR_L2-3',
        'color': '#803300ff',
        'xlim': [-1, 1.5]
    },
    'PYR_L4-6': {
        'name': 'PYR_L4-6',
        'color': '#502d16ff',
        'xlim': [-1, 1.5]
    }
}

SPIKE_CONFIG = {
    'extractor': {
        'signal_name': 'wideband',
        'event_name': 'spk',
        'cut_window': [-0.001, 0.002],
        'fs': 40000
    },
    'preprocess': {
        'sg_filter': {
            'window_length': 5,
            'polyorder': 3
        },
        'normalize': None
    }
}
START_CONFIG = {
    'smooth': SPIKE_CONFIG['preprocess']['sg_filter'],
    'start_threshold': 0.1,
}

DT = 1 / 40  # Sampling interval in seconds

PROJECT_DIR = Path('E:/Data/BogiSejtek/PutRAAP')
RAW_DATA_DIR = PROJECT_DIR
RESULT_DIR = (
    PROJECT_DIR /
    'Results' /
    '20250703' /
    'InVivoJuxta' /
    'DvMaxNorm' /
    'Search_0.15-0.85' /
    'w5_p3'
)
FEATURES_DIR = (
    RESULT_DIR /
    'Preprocessed' /
    'Individual'
)
FIGURE_DIR = (
    RESULT_DIR /
    'Figures' /
    'Individual' /
    'ThresholdGroupedSpikeShapes'
)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

for file_name in FEATURES_DIR.glob('*.csv'):
    df = pd.read_csv(file_name)
    cell_name = df['cell_name'].iloc[0]
    cell_type = df['cell_type'].iloc[0]

    print(f'Processing {cell_type} cell: {cell_name}')

    raw_file_path = RAW_DATA_DIR / cell_type / f'{cell_name}.mat'
    spikes = prepare_data(raw_file_path, config=SPIKE_CONFIG)
    times = spikes.times * 1000
    n_spikes = df.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey='row', sharex='row')
    for category_id, category in enumerate(CATEGORIES):
        low = category['low']
        high = category['high']
        if category_id == 0 or category_id == 2:
            flag = (
                (df['relative_dv_drop'] > low[0]) &
                (df['relative_dv_drop'] <= high[0]) &
                (df['relative_rebound_time'] > low[1]) &
                (df['relative_rebound_time'] <= high[1])
            )
        else:
            flag = (
                (df['relative_dv_drop'] > low[0]) &
                (df['relative_dv_drop'] <= high[0]) |
                (df['relative_rebound_time'] > low[1]) &
                (df['relative_rebound_time'] <= high[1])
            )
        category_timestamps = df.loc[flag, 'timestamp'].values
        category_size = np.sum(flag)

        ax1 = axes[0, category_id]
        ax2 = axes[1, category_id]

        if category_size > 0:

            valid_values = []
            for i in range(spikes.values.shape[0]):
                if np.min(np.abs(category_timestamps - spikes.timestamps[i])) < 0.001:
                    v = spikes.values[i, :]
                    start_i = find_start_index(v, START_CONFIG)
                    if start_i is None:
                        continue
                    valid_values.append(v - v[start_i])

            values = np.stack(valid_values, axis=0)

            print(f'#spike={category_size}, #shape={values.shape[0]}')

            for i in range(values.shape[0]):
                ax1.plot(
                    times,
                    values[i, :],
                    color="#dedede",
                    alpha=0.3
                )
                ax2.plot(
                    values[i, :-1],
                    np.diff(values[i, :]) / DT,
                    color="#dedede",
                    alpha=0.3
                )
                mean_shape = np.mean(values, axis=0)
                mean_derivative = np.diff(mean_shape) / DT
                ax1.plot(
                    times,
                    mean_shape,
                    color=CELL_TYPES[cell_type]['color'],
                    linewidth=2,
                    linestyle=category['linestyle']
                )
                ax2.plot(
                    mean_shape[:-1],
                    mean_derivative,
                    color=CELL_TYPES[cell_type]['color'],
                    linewidth=2,
                    linestyle=category['linestyle']
                )

        ax1.set_xlim(CELL_TYPES[cell_type]['xlim'])
        x_ticks = ax1.get_xticks()
        # ax1.set_xticks([x_ticks[0], 0, x_ticks[-1]])
        ax1.set_yticks([])
        ax1.set_xticks([])
        # ax1.set_title(
        #     category['name'] +
        #     f' {category_size}/{n_spikes}' +
        #     f' ({100 * category_size / n_spikes:.1f}%)'
        # )
        x_ticks = ax2.get_xticks()
        # ax2.set_xticks([x_ticks[0], 0, x_ticks[-1]])
        ax2.set_yticks([])
        ax2.set_xticks([])

    plt.tight_layout(pad=1.0)
    plt.savefig(
        FIGURE_DIR / f'{cell_name}_grouped_spike_shapes.png',
        dpi=300
    )
    plt.close(fig)
    # plt.show()
