import math
from typing import Union

import numpy as np
from sklearn import metrics

from .utils import _is_percent, DataType


class MetricError(Exception):
    pass


def lift_score(y_target: DataType, y_predicted: DataType) -> float:
    """
    Calculates lift score for a given model. The lift score quantifies how much better
    the model is compared to a random baseline.

    The formula is defined as follows:
        lift = (TP/(TP+FN)(TP+FP)/(TP+TN+FP+FN)
        Source: https://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score

    :param y_target:
        Target labels

    :param y_predicted:
        Predicted labels

    :return:
        Lift score
    """
    y_target = np.array(y_target)
    y_predicted = np.array(y_predicted)

    if y_target.ndim > 1 or y_predicted.ndim > 1:
        raise MetricError("Input must be 1-dimensional")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalized=True) -> np.ndarray:
    """
    Generates a confusion matrix using sklearn's confusion_matrix function, with the ability
    to add normalization.

    Calculates a matrix of
        [[True Positives, False Positives],
         [False Negatives, True Negatives]]

    :param y_true:
        True labels

    :param y_pred:
        Predicted labels
    :param normalized:
        Whether or not to normalize counts

    :return:
        Confusion matrix array
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalized is True:
        cm = cm / cm.sum()
        cm = np.around(cm, 2)
        cm[np.isnan(cm)] = 0.0
    return cm


def sorted_feature_importance(labels: np.ndarray,
                              importance: np.ndarray,
                              top_n: Union[int, float] = None,
                              bottom_n: Union[int, float] = None):
    """
    Calculates a sorted array of importances and corresponding labels by absolute values

    :param labels:
        List of feature labels

    :param importance:
        List of importance values

    :param top_n:
        If top_n is an int return top n features
        If top_n is a float between 0 and 1 return top top_n percent of features

    :param bottom_n:
        If bottom_n is an int return bottom n features
        If bottom_n is a float between 0 and 1 return bottom bottom_n percent of features

    :return:
        List of labels and list of feature importances sorted by importance
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    idx = np.argsort(np.abs(importance))[::-1]

    sorted_idx = []

    if top_n:
        if _is_percent(top_n):
            top_n = math.floor(top_n * len(idx)) or 1  # If floor rounds to 0, use 1 instead
        sorted_idx.extend(idx[:top_n])

    if bottom_n:
        if _is_percent(bottom_n):
            bottom_n = math.floor(bottom_n * len(idx)) or 1  # If floor rounds to 0, use 1 instead
        sorted_idx.extend(idx[::-1][:bottom_n])

    if sorted_idx:
        idx = sorted_idx

    return labels[idx], importance[idx]
