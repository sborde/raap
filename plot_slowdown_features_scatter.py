"""
This script shows a scatter plot of the two slowdown features.
Each point represents a 2D bin, the size of the point represents
the relative number of that type of spike in that bin.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_bins(values, bin_size):
    """
    Get the minimum and maximum values from a list of values.
    """
    min_val, max_val = min(values), max(values)

    data_range = max_val - min_val
    n_bins = int(np.ceil(data_range / bin_size))
    binned_range = n_bins * bin_size
    diff = binned_range - data_range
    min_val -= diff / 2
    max_val += diff / 2

    return np.linspace(min_val, max_val, n_bins + 1)

PROJECT_DIR = Path('E:/Data/BogiSejtek/PutRAAP/Results')

INPUT_PATH = PROJECT_DIR / '20250625' / 'whole_smooth_w5_p3_dataset.csv'
OUTPUT_DIR = PROJECT_DIR / '20250625' / 'Figures' / 'InVitroWhole' / 'w5_p3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'RAAP': '#ff5000',
    'AISAP': '#0000ff'
}

PREDICTOR_FEATURES = ['relative_dv_drop', 'relative_rebound_time']
LABEL_FEATURE = 'spike_type'

BIN_SIZE = 0.005

data = pd.read_csv(INPUT_PATH)

x1_bins = generate_bins(data['relative_dv_drop'], BIN_SIZE)
x2_bins = generate_bins(data['relative_rebound_time'], BIN_SIZE)

count_by_type = data[LABEL_FEATURE].value_counts()

bin_counts = []
for x11, x12 in zip(x1_bins[:-1], x1_bins[1:]):
    for x21, x22 in zip(x2_bins[:-1], x2_bins[1:]):
        for spike_type in PALETTE.keys():
            count = data[
                (data['relative_dv_drop'] >= x11) &
                (data['relative_dv_drop'] < x12) &
                (data['relative_rebound_time'] >= x21) &
                (data['relative_rebound_time'] < x22) &
                (data[LABEL_FEATURE] == spike_type)
            ].shape[0]

            if count > 0:
                # Only add bins with at least one spike of the type
                bin_counts.append({
                    'x1': np.mean([x11, x12]),
                    'x2': np.mean([x21, x22]),
                    'count': count,
                    'ratio': count / count_by_type[spike_type],
                    'spike_type': spike_type
                })

aggregated_data = pd.DataFrame(bin_counts).sort_values(
    by=['spike_type'],
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(
    data=aggregated_data,
    x='x1',
    y='x2',
    size='ratio',
    hue='spike_type',
    sizes=(10, 1000),
    palette=PALETTE,
    ax=ax
)
ax.set_xlabel('Relative DV Drop')
ax.set_ylabel('Relative Rebound Time')

# Remove the hue from the legend
handles, labels = ax.get_legend_handles_labels()
# Keep only the style (marker) entries in the legend
components_to_keep = [
    (h, l) for h, l 
    in zip(handles, labels)
    if l in ['RAAP', 'AISAP']
]
style_handles = [h for h, _ in components_to_keep]
style_labels = [l for _, l in components_to_keep]
ax.legend(
    style_handles,
    style_labels,
    title='Spike Type',
    loc='lower right'
)
ax.set_title(
    'Relative ratio of each spike type ' +
    f'in each 2D bin of size {BIN_SIZE}'
)

plt.savefig(
    OUTPUT_DIR /
    'slowdown_feature_scatterplot_2d_with_sizes.png',
    dpi=300
)
plt.close(fig)
