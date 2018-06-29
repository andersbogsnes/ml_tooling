"""
Helper functions for vizualisations
"""
import numpy as np


class VizError(Exception):
    """Base Exception for visualization errors"""
    pass


def cum_gain_curve(y_true, y_proba, positive_label=1):
    n = len(y_true)
    n_true = np.sum(y_true == positive_label)

    idx = np.argsort(y_proba)[::-1]  # Reverse sort to get descending values
    cum_gains = np.cumsum(y_true[idx]) / n_true
    percents = np.arange(1, n + 1) / n
    return percents, cum_gains


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


def get_feature_importance(model):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_

    elif hasattr(model, 'coef_'):
        importance = model.coef_
        if importance.ndim > 1:
            importance = importance[0]
    else:
        raise VizError(f"{model.__class__.__name__} does not have either coef_ or feature_importances_")

    return importance
