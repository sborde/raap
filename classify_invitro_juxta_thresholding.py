"""
This script performs cross validated classification of in-vitro juxta
cells' spikes into RA-AP and AIS-AP classes.
"""

from pathlib import Path
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from analyses.slowdown_descriptors import calculate_sparse_2d_histogram

Prediction2D = namedtuple('Prediction2D', ['x', 'y', 'p'])


def generate_meshgrid_for_dataset(x, margin=0.1, n_points=200):
    """
    Generate a meshgrid for the given feature space.
    """
    x1_min, x1_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    x2_min, x2_max = x[:, 1].min() - margin, x[:, 1].max() + margin
    return np.meshgrid(
        np.linspace(x1_min, x1_max, n_points),
        np.linspace(x2_min, x2_max, n_points)
    )

def plot_classified_space(
    axis: Axes,
    x: np.ndarray,
    labels: np.ndarray,
    predictions: Prediction2D,
    classes: list[dict]
):
    """
    Plot the classification result with decision areas and instances.
    """
    colors = [c['color'] for c in classes]
    ids = [str(c['id']) for c in classes]
    names = [c['name'] for c in classes]

    print(names)

    colormap = LinearSegmentedColormap.from_list('class_gradient', colors)
    axis.contourf(
        predictions.x,
        predictions.y,
        predictions.p,
        levels=np.linspace(0, 1, 100),
        cmap=colormap,
        antialiased=True,
        alpha=0.5
    )
    binned_data = calculate_sparse_2d_histogram(x, labels, bin_size=0.005)
    sns.scatterplot(
        data=binned_data,
        x='x1',
        y='x2',
        size='ratio',
        hue='label',
        linewidth=0,
        sizes=(10, 1000),
        palette=colors,
        ax=axis
    )
    axis.set_xlabel('Relative dV Drop')
    axis.set_ylabel('Relative Rebound Time')

    legend_handles, legend_labels = axis.get_legend_handles_labels()

    print(legend_labels)

    components_to_keep = [
        (handle, names[int(label)]) for handle, label
        in zip(legend_handles, legend_labels)
        if label in ids
    ]
    style_handles = [h for h, _ in components_to_keep]
    style_labels = [l for _, l in components_to_keep]

    print(style_labels)

    axis.legend(
        style_handles,
        style_labels,
        title='Spike Type',
        loc='lower right'
    )

    axis.set_aspect('equal')

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score for the predictions."""
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def display_prediction_space(x, labels, model, classes):
    """Display the prediction results on a 2D plot."""

    xx, yy = generate_meshgrid_for_dataset(x, 0.005, 200)
    p = model(np.c_[xx.ravel(), yy.ravel()])
    p = p.reshape(xx.shape)

    prediction = Prediction2D(xx, yy, p)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    plot_classified_space(
        ax,
        x=x,
        labels=labels,
        predictions=prediction,
        classes=classes
    )
    return fig

def classify_by_thresholds(x, th1, th2):
    """
    Classify the data points based on the given thresholds.
    """
    return np.where(
        (x[:, 0] >= th1) & (x[:, 1] >= th2),
        1,  # RAAP
        0   # AISAP
    )

def find_best_threshold(
    x,
    y_real,
    thresholds_x1,
    thresholds_x2,
    target_precision=0.99,
    verbose=0
):
    """
    Find the best thresholds for classification based on precision and recall.
    """
    result = {
        'x1': None,
        'x2': None,
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }

    for th_x1 in thresholds_x1:
        if np.sum(x[:, 0] >= th_x1) == 0:
            continue
        for th_x2 in thresholds_x2:
            if np.sum(x[:, 1] >= th_x2) == 0:
                continue

            if verbose > 0:
                print(
                    f'Testing thresholds: x1={th_x1:.2f}, x2={th_x2:.2f}',
                    end='\r'
                )

            y_pred = classify_by_thresholds(x, th_x1, th_x2)
            precision = precision_score(y_real, y_pred, zero_division=0)
            recall = recall_score(y_real, y_pred, zero_division=0)
            f1 = f1_score(y_real, y_pred, zero_division=0)

            if precision < result['precision']:
                continue
            if result['precision'] >= target_precision and result['recall'] > recall:
                continue

            result = {
                'x1': th_x1,
                'x2': th_x2,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

    return result

RUN_CV = False
N_FOLD = 2

DIR_PATH = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results') /
    '20250702' /
    'InVitroJuxta' /
    'DvMaxNorm' /
    'Search_0.15-0.85' /
    'w11_p2'
)

CLASSES = [
    {
        'name': 'AISAP',
        'color': '#0000ff',
        'id': 0
    },
    {
        'name': 'RAAP',
        'color': '#ff5000',
        'id': 1
    }
]
CLASS_MAPPER = {'RAAP': 1, 'AISAP': 0}

PREDICTOR_FEATURES = ['relative_dv_drop', 'relative_rebound_time']
TARGET_FEATURE = 'spike_type'

splitter = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)

data_frames = []
for file in DIR_PATH.glob('*.csv'):
    print(f'Load {file.name}')
    df = pd.read_csv(file)
    df['cell_name'] = file.stem
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)
data = data.sort_values(by=['spike_type'], ascending=False)

X = data[PREDICTOR_FEATURES].values
y = data[TARGET_FEATURE].map(CLASS_MAPPER).values.astype(int)

x1_thresholds = np.linspace(0, 1, 201)
x2_thresholds = np.linspace(0, 1, 201)

if RUN_CV:
    print(f'Running {N_FOLD}-fold cross-validation...')

    fold_results = []
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_threshold = find_best_threshold(
            X_train,
            y_train,
            x1_thresholds,
            x2_thresholds
        )

        x1_th = best_threshold['x1']
        x2_th = best_threshold['x2']

        metrics = {
            'train': calculate_metrics(y_train, classify_by_thresholds(X_train, x1_th, x2_th)),
            'val': calculate_metrics(y_test, classify_by_thresholds(X_test, x1_th, x2_th))
        }

        print(f'Fold {fold + 1} - (Training/Validation)')
        print('========================================')
        print(
            f'\tprecision: {metrics['train']["precision"]:.2f}/' +
            f'{metrics['val']["precision"]:.2f}\n' +
            f'\trecall: {metrics['train']["recall"]:.2f}/' +
            f'{metrics['val']["recall"]:.2f}\n' +
            f'\tf1_score: {metrics['train']["f1_score"]:.2f}/' +
            f'{metrics['val']["f1_score"]:.2f}'
        )

        fold_results.extend([
            {
                'fold': fold + 1,
                'threshold_x1': x1_th,
                'threshold_x2': x2_th,
                'set': set_name,
                'metric': metric_name,
                'value': metric_value
            } for set_name, set_metrics in metrics.items()
            for metric_name, metric_value in set_metrics.items()
        ])

    cv_results = pd.DataFrame(fold_results)
    cv_results.to_csv(
        DIR_PATH / 'classification_thresholding_cv_results.csv',
        index=False
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(
        data=cv_results,
        x='metric',
        y='value',
        hue='set',
        fill=False,
        dodge=True,
        width=0.3,
        gap=0.2,
        ax=ax
    )
    sns.stripplot(
        data=cv_results,
        x='metric',
        y='value',
        hue='set',
        dodge=True,
        jitter=True,
        alpha=0.5,
        ax=ax
    )

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(f'{N_FOLD}-fold CV metrics')
    ax.set_ylim(0.5, 1)
    ax.legend()
    plt.savefig(
        DIR_PATH / 'classification_thresholding_cv_results_boxplot.png',
        dpi=300
    )
    plt.close(fig)

best_threshold = find_best_threshold(
    X,
    y,
    x1_thresholds,
    x2_thresholds,
    target_precision=0.99,
    verbose=1
)

th_x1 = best_threshold['x1']
th_x2 = best_threshold['x2']

fig = display_prediction_space(
    X,
    y,
    lambda x: classify_by_thresholds(x, th_x1, th_x2),
    CLASSES
)
plt.title(
    'Classification with thresholds: ' +
    f'x1={best_threshold["x1"]:.2f}, x2={best_threshold["x2"]:.2f}\n' +
    f'Precision: {best_threshold["precision"]:.2f}, ' +
    f'Recall: {best_threshold["recall"]:.2f}, ' +
    f'F1 Score: {best_threshold["f1_score"]:.2f}'
)

FIGURE_NAME = 'classification_thresholding_full_results'

plt.savefig(
    DIR_PATH /
    f'{FIGURE_NAME}.png',
    format='png',
    dpi=300
)
with plt.rc_context({'svg.fonttype': 'none'}):
    plt.savefig(
        DIR_PATH /
        f'{FIGURE_NAME}.svg',
        format='svg',
        dpi=300
    )

plt.close(fig)
