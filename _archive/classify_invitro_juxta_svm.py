"""
This script performs cross validated classification of in-vitro juxta
cells' spikes into RA-AP and AIS-AP classes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from collections import namedtuple


Prediction2D = namedtuple('Prediction2D', ['x', 'y', 'p'])


def generate_meshgrid(x1, x2, margin=0.1, n_points=200):
    """
    Generate a meshgrid for the given feature space.
    """
    x1_min, x1_max = x1.min() - margin, x1.max() + margin
    x2_min, x2_max = x2.min() - margin, x2.max() + margin
    return np.meshgrid(
        np.linspace(x1_min, x1_max, n_points),
        np.linspace(x2_min, x2_max, n_points)
    )

def plot_classified_space(
    ax,
    data,
    labels,
    prediction,
    class_palette,
    gradient_colormap
):
    """
    Plot the classification result with decision areas and instances.
    """
    ax.contourf(
        prediction.x,
        prediction.y,
        prediction.p,
        levels=np.linspace(0, 1, 100),
        cmap=gradient_colormap,
        alpha=0.6,
        antialiased=True
    )
    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=labels,
        edgecolor=None,
        palette=class_palette,
        ax=ax
    )

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score for the predictions."""
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def display_prediction_space(x, y, model, palette):
    """Display the prediction results on a 2D plot."""

    xx, yy = generate_meshgrid(x[:, 0], x[:, 1], 0.1, 200)
    p = model.predict(np.c_[xx.ravel(), yy.ravel()])
    p = p.reshape(xx.shape)

    prediction = Prediction2D(xx, yy, p)

    gradient_colormap = LinearSegmentedColormap.from_list(
        'spike_type_gradient', palette
    )

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    plot_classified_space(
        ax,
        data=x,
        labels=y,
        prediction=prediction,
        class_palette=palette,
        gradient_colormap=gradient_colormap
    )
    return fig

N_FOLD = 2

DIR_PATH = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results') /
    '20250702' /
    'InVitroJuxta' /
    'DvMaxNorm' /
    'Search_0.15-0.85' /
    'w11_p2'
)

INPUT_PATH = DIR_PATH / 'invitro_juxta_dataset.csv'

PALETTE = {'RAAP': '#ff5000', 'AISAP': '#0000ff'}
CLASS_MAPPER = {'RAAP': 1, 'AISAP': 0}



PREDICTOR_FEATURES = ['relative_dv_drop', 'relative_rebound_time']
TARGET_FEATURE = 'spike_type'

splitter = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)

data = pd.read_csv(INPUT_PATH).sort_values(by=['spike_type'], ascending=False)

X = data[PREDICTOR_FEATURES].values
y = data[TARGET_FEATURE].map(CLASS_MAPPER).values.astype(int)
CLASS_WEIGHT = {
    0: np.sum(y == 0) / len(y),
    1: np.sum(y == 1) / len(y)
}

print(f'Class weights: {CLASS_WEIGHT}')

fold_results = []
for fold, (train_index, test_index) in enumerate(splitter.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma=0.5,
        random_state=42,
        class_weight=CLASS_WEIGHT
    )
    clf.fit(X_train, y_train)

    metrics = {
        'train': calculate_metrics(y_train, clf.predict(X_train)),
        'val': calculate_metrics(y_test, clf.predict(X_test))
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
            'set': set_name,
            'metric': metric_name,
            'value': metric_value
        } for set_name, set_metrics in metrics.items()
        for metric_name, metric_value in set_metrics.items()
    ])

# cv_results = pd.DataFrame(fold_results)
# cv_results.to_csv(
#     DIR_PATH / 'classification_rbf_svm_cv_results.csv',
#     index=False
# )

# fig, ax = plt.subplots(figsize=(10, 10))

# sns.boxplot(
#     data=cv_results,
#     x='metric',
#     y='value',
#     hue='set',
#     fill=False,
#     dodge=True,
#     width=0.3,
#     gap=0.2,
#     ax=ax
# )

# sns.stripplot(
#     data=cv_results,
#     x='metric',
#     y='value',
#     hue='set',
#     dodge=True,
#     jitter=True,
#     alpha=0.5,
#     ax=ax
# )

# ax.set_xlabel('Metric')
# ax.set_ylabel('Value')
# ax.set_title(f'{N_FOLD}-fold CV metrics')
# ax.set_ylim(0.5, 1)
# ax.legend()
# plt.show()
# plt.savefig(
#     DIR_PATH / 'classification_logreg_cv_results_boxplot.png',
#     dpi=300
# )
# plt.close(fig)

clf_full = SVC(
    kernel='rbf',
    C=1.0,
    gamma=0.5,
    random_state=42
)
clf_full.fit(X, y)
display_prediction_space(X, y, clf_full, [PALETTE['AISAP'], PALETTE['RAAP']])
plt.show()

# fig = plot_logreg_result(X, y, clf)
# plt.savefig(
#     DIR_PATH /
#     'classification_logreg_selected_full.png',
#     dpi=300
# )
# plt.close(fig)
