"""
This script performs cross validated classification of in-vitro juxta
cells' spikes into RA-AP and AIS-AP classes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
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

def color_decision_areas(prediction, ax, color_map):
    """
    Fill the area with a gradient color map
    according to the predicted probabilities.
    """
    ax.contourf(
        prediction.x,
        prediction.y,
        prediction.p,
        levels=np.linspace(0, 1, 100),
        cmap=color_map,
        alpha=0.6,
        antialiased=True
    )

def plot_decision_boundary(prediction, ax, decision_threshold=0.5):
    """
    Plot the decision boundary and fill the areas with colors.
    """
    ax.contour(
        prediction.x,
        prediction.y,
        prediction.p,
        levels=[decision_threshold],
        linewidths=2,
        colors='k',
        linestyles='--'
    )

def plot_instances(
        data,
        x_col,
        y_col,
        hue_col,
        ax,
        palette,
        edgecolor=None
):
    """
    Plot instances with specified x and y columns, colored by hue.
    """
    sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        edgecolor=edgecolor,
        palette=palette,
        ax=ax
    )

def plot_classification_result(
    ax,
    data,
    feature_names,
    target_name,
    prediction,
    class_palette,
    gradient_colormap
):
    """
    Plot the classification result with decision areas and instances.
    """
    
    color_decision_areas(prediction, ax, gradient_colormap)
    plot_decision_boundary(prediction, ax, decision_threshold=0.5)
    plot_instances(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        hue_col=target_name,
        ax=ax,
        palette=class_palette
    )

def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    precision = true_positive / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
    recall = true_positive / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def evaluate_logreg(X, y, clf):
    """
    Evaluate logistic regression model at various thresholds
    and calculate precision, recall, and F1 score.
    """
    probs = clf.predict_proba(X)
    metrics = []
    for threshold in np.arange(0, 1, 0.01):
        y_pred = (probs[:, 1] >= threshold).astype(int)
        precision, recall, f1_score = calculate_metrics(y, y_pred)
        metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
    return pd.DataFrame(metrics)

def evaluate_decision_tree(X, y, clf):
    """
    Evaluate decision tree classifier and return metrics.
    """
    y_pred = clf.predict(X)
    precision, recall, f1_score = calculate_metrics(y, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def plot_thresholded_classification_result(ax, metrics):
    """
    Plot the classification result with decision areas and instances
    at a specific threshold.
    """
    for metric, label in zip(
        ['f1_score', 'precision', 'recall'],
        ['F1 Score', 'Precision', 'Recall']
    ):
        sns.lineplot(
            x='threshold',
            y=metric,
            data=metrics,
            label=label,
            ax=ax
        )

    for threshold, linestyle in zip([0.95, 0.99], ['--', '-.']):
        ax.hlines(
            threshold,
            0,
            1,
            color='black',
            linestyle=linestyle,
            label=f'Threshold {100 * threshold}%')

    ax.legend()

def plot_logreg_result(
    X,
    y,
    clf
):
    metrics = evaluate_logreg(X, y, clf)

    xx, yy = generate_meshgrid(X[:, 0], X[:, 1], 0.1, 200)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    prediction = Prediction2D(xx, yy, Z)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    plot_thresholded_classification_result(ax[0], metrics)

    plot_classification_result(
        ax[1],
        data,
        feature_names=PREDICTOR_FEATURES,
        target_name=TARGET_FEATURE,
        prediction=prediction,
        class_palette=PALETTE,
        gradient_colormap=GRADIENT_CMAP
    )

    return fig

INPUT_PATH = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results') /
    '20250701' / 'InVitroJuxta' / 'DvMaxNorm' / 'w11_p3' /
    'invitro_juxta_dataset.csv'
)
OUTPUT_DIR = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results') /
    '20250701' / 'InVitroJuxta' / 'DvMaxNorm' / 'w11_p3'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'RAAP': '#ff5000',
    'AISAP': '#0000ff'
}

GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    'spike_type_gradient',
    [
        PALETTE['AISAP'],
        PALETTE['RAAP']
    ]
)

PREDICTOR_FEATURES = ['relative_dv_drop', 'relative_rebound_time']
TARGET_FEATURE = 'spike_type'

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

data = pd.read_csv(INPUT_PATH).sort_values(by=['spike_type'], ascending=False)

# Prepare features and target
X = data[PREDICTOR_FEATURES].values
y = np.asarray(LabelEncoder().fit_transform(data[TARGET_FEATURE]))

fold_results = []
for fold, (train_index, test_index) in enumerate(splitter.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)

    training_metrics = evaluate_logreg(X_train, y_train, clf)
    th_idx = np.argmin(np.abs(training_metrics['threshold'] - 0.5))
    th_95_idx = np.argmin(np.abs(training_metrics['precision'] - 0.95))
    th_99_idx = np.argmin(np.abs(training_metrics['precision'] - 0.99))

    print(f'Fold {fold + 1}\n==========================')
    print(
        f'threshold to achieve precision\n'
        f'\t- 0.95: {training_metrics["threshold"][th_95_idx]} ' +
        f'({training_metrics["precision"][th_95_idx]:.2f})\n'
        f'\t- 0.99: {training_metrics["threshold"][th_99_idx]}' +
        f' ({training_metrics["precision"][th_99_idx]:.2f})'
    )
    print(
        f'with threshold at 0.5\n'
        f'\t- precision: {training_metrics["precision"][th_idx]:.2f}\n'
    )

    validation_metrics = evaluate_logreg(X_test, y_test, clf)
    print(
        f'precision of the validation set at threshold\n'
        f'\t- 0.95: {validation_metrics["precision"][th_95_idx]:.2f}\n'
        f'\t- 0.99: {validation_metrics["precision"][th_99_idx]:.2f}'
    )
    print(
        f'with threshold at 0.5\n'
        f'\t- precision: {validation_metrics["precision"][th_idx]:.2f}\n'
    )

    for set_name, set_metrics in zip(
        ['train', 'validation'],
        [training_metrics, validation_metrics]
    ):
        for threshold_name, threshold_idx in zip(
            ['thresh_05', 'thresh_95', 'thresh_99'],
            [th_idx, th_95_idx, th_99_idx]
        ):
            fold_results.append({
                'fold': fold + 1,
                'set': set_name,
                'threshold': threshold_name,
                'precision': set_metrics['precision'][threshold_idx]
            })

cv_results = pd.DataFrame(fold_results)
cv_results.to_csv(
    OUTPUT_DIR / 'classification_logreg_cv_results.csv',
    index=False
)

fig, ax = plt.subplots(figsize=(10, 10))

sns.boxplot(
    x='threshold',
    y='precision',
    hue='set',
    data=cv_results,
    fill=False,
    dodge=True,
    width=0.3,  # Reduced width to create more space between boxes
    gap=0.2,    # Add gap between grouped boxes (if supported in your seaborn version)
    ax=ax
)
# Scatter plot of all points
sns.stripplot(
    x='threshold',
    y='precision',
    hue='set',
    data=cv_results,
    dodge=True,
    jitter=True,
    alpha=0.5,
    ax=ax
)
ax.set_xticklabels(
    ['Threshold 0.5', 'Threshold for 0.95 precision',
     'Threshold for 0.99 precision']
)
ax.set_xlabel('Threshold')
ax.set_ylabel('Precision')
ax.set_title('Cross-validated Classification Results with different thresholds')
ax.set_ylim(0.5, 1)
ax.legend()
plt.savefig(
    OUTPUT_DIR / 'classification_logreg_cv_results_boxplot.png',
    dpi=300
)
plt.close(fig)

# # Calculate median and whiskers (25th and 75th percentiles) for each group
# grouped = cv_results.groupby(['threshold', 'set'])['precision']
# medians = grouped.median().reset_index()
# q1 = grouped.quantile(0.25).reset_index().rename(columns={'precision': 'q1'})
# q3 = grouped.quantile(0.75).reset_index().rename(columns={'precision': 'q3'})
# summary = medians.merge(q1, on=['threshold', 'set']).merge(q3, on=['threshold', 'set'])

# # Plot median and whiskers only (no box body)
# colors = sns.color_palette()[:len(cv_results['set'].unique())]
# for i, (set_name, color) in enumerate(zip(cv_results['set'].unique(), colors)):
#     subset = summary[summary['set'] == set_name]
    
#     # Create offset for different sets to avoid overlap
#     x_positions = np.arange(len(subset))
#     offset = (i - 0.5) * 0.1  # Adjust offset as needed

#     # Plot median line
#     # ax.plot(x_positions + offset, subset['precision'], 
#     #         color=color, linewidth=3, marker='o', markersize=15,
#     #         label=f"{set_name} median")
#     ax.scatter(x_positions + offset, subset['precision'],
#             color=color, s=100,
#             label=f"{set_name} median")


#     # Plot whiskers (vertical lines from Q1 to Q3)
#     for j, (_, row) in enumerate(subset.iterrows()):
#         ax.plot([j + offset, j + offset], [row['q1'], row['q3']], 
#                 color=color, linewidth=2, alpha=0.8)
        
#         # Add whisker caps (small horizontal lines at Q1 and Q3)
#         cap_width = 0.02
#         ax.plot([j + offset - cap_width, j + offset + cap_width], [row['q1'], row['q1']], 
#                 color=color, linewidth=2, alpha=0.8)
#         ax.plot([j + offset - cap_width, j + offset + cap_width], [row['q3'], row['q3']], 
#                 color=color, linewidth=2, alpha=0.8)

# # Set x-axis labels
# ax.set_xticks(range(len(summary['threshold'].unique())))
# ax.set_xticklabels(summary['threshold'].unique())

# ax.set_ylim(0.5, 1)
# ax.legend()
# plt.savefig(
#     OUTPUT_DIR / 'classification_logreg_cv_results.png',
#     dpi=300
# )
# plt.close(fig)

fig = plot_logreg_result(X, y, clf)
plt.savefig(
    OUTPUT_DIR /
    'classification_logreg_selected_full.png',
    dpi=300
)
plt.close(fig)
