"""
Helper functions for vizualisations
"""
from typing import Tuple

import numpy as np


class VizError(Exception):
    """Base Exception for visualization errors"""
    pass


def cum_gain_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   positive_label=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a cumulative gain curve of how many positives are captured
    per percent of sorted data.

    :param y_true:
        True labels

    :param y_proba:
        Predicted label

    :param positive_label:
        Which class is considered positive class in multiclass settings

    :return:
        array of data percents and cumulative gain
    """
    n = len(y_true)
    n_true = np.sum(y_true == positive_label)

    idx = np.argsort(y_proba)[::-1]  # Reverse sort to get descending values
    cum_gains = np.cumsum(y_true[idx]) / n_true
    percents = np.arange(1, n + 1) / n
    return percents, cum_gains


def generate_text_labels(ax, horizontal=False, padding=0.005):
    """
    Helper for generating text labels for bar charts

    :param ax:
        Ax which has patches on it

    :param horizontal:
        Whether or not the graph is a barh or a regular bar

    :param padding:
        How much padding to multiply by

    :return:
        x and y values for ax.text
    """
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


def get_feature_importance(model) -> np.ndarray:
    """
    Helper function for extracting importances.
    Checks for coef_ or feature_importances_ on model

    :param model:
        A sklearn estimator

    :return:
        array of importances
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_

    elif hasattr(model, 'coef_'):
        importance = model.coef_
        if importance.ndim > 1:
            importance = importance[0]
    else:
        model_name = model.__class__.__name__
        raise VizError(f"{model_name} does not have either coef_ or feature_importances_")

    return importance
