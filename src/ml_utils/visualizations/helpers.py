import numpy as np
from sklearn.metrics import confusion_matrix


def cum_gain_curve(y_true, y_proba, positive_label=1):
    n = len(y_true)
    n_true = np.sum(y_true == positive_label)

    idx = np.argsort(y_proba)[::-1]  # Reverse sort to get descending values
    cum_gains = np.cumsum(y_true[idx]) / n_true
    percents = np.arange(1, n + 1) / n
    return percents, cum_gains


def sorted_feature_importance(labels, importance):
    idx = np.argsort(np.abs(importance))
    return labels[idx], importance[idx]


def generate_text_labels(ax, horizontal=False, padding=0.005):
    for (i, patch) in enumerate(ax.patches):
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()

        if horizontal is True:
            padded = ax.get_xbound()[1] * padding
            x_value = width + padded
            y_value = y + (height / 2)
        else:
            padded = ax.get_ybound()[1] * padding
            x_value = x + (width / 2)
            y_value = height + padded

        yield x_value, y_value


def create_confusion_matrix(y_true, y_pred, normalized=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalized is True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
