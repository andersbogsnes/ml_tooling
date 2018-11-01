"""
Contains all viz functions
"""
from typing import Tuple, Sequence, Union

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
import numpy as np
import itertools
from matplotlib.axes import Axes
from sklearn.pipeline import Pipeline

from . import metrics
from .utils import DataType, _is_percent


def plot_roc_auc(y_true: DataType, y_proba: DataType, title: str = None, ax: Axes = None) -> Axes:
    """
    Plot ROC AUC curve. Works only with probabilities

    :param y_true:
        True labels

    :param y_proba:
        Probability estimate from model

    :param title:
        Plot title

    :param ax:
        Pass in your own ax

    :return:
        matplotlib.Axes
    """
    title = 'ROC AUC curve' if title is None else title

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    score = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"ROC Score: {score}")
    ax.plot([0, 1], '--')
    ax.set_title(title)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc='best')
    plt.tight_layout()
    return ax


def plot_confusion_matrix(y_true: DataType,
                          y_pred: DataType,
                          normalized: bool = True,
                          title: str = None,
                          ax: Axes = None,
                          labels: Sequence[str] = None) -> Axes:
    """
    Plots a confusion matrix of predicted labels vs actual labels

    :param y_true:
        True labels

    :param y_pred:
        Predicted labels from model

    :param normalized:
        Whether to normalize counts in matrix

    :param title:
        Title for plot

    :param ax:
        Pass your own ax

    :param labels:
        Pass custom list of labels

    :return:
        matplotlib.Axes
    """

    title = 'Confusion Matrix' if title is None else title

    if normalized:
        title = f"{title} - Normalized"

    cm = metrics.confusion_matrix(y_true, y_pred, normalized=normalized)

    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        unique_labels = np.unique(y_true)
        labels = list(unique_labels)

    cax = ax.matshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    labels.insert(0, '')
    ax.set_title(title)
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')

    plt.colorbar(cax, ax=ax)
    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    return ax


def plot_residuals(y_true: DataType,
                   y_pred: DataType,
                   title: str = None,
                   ax: Axes = None) -> Axes:
    """
    Plots residuals from a regression.

    :param y_true:
        True values

    :param y_pred:
        Models predicted value

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """
    title = f'Residual Plot' if title is None else title

    residuals = y_pred - y_true
    r2 = r2_score(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(y_pred, residuals, label=f'$R^2 = {r2:0.3f}$')
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted Value')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax


def plot_prediction_error(y_true: DataType,
                          y_pred: DataType,
                          title: str = None,
                          ax: Axes = None) -> Axes:
    """
    Plots prediction error of regression model

    :param y_true:
        True values

    :param y_pred:
        Model's predicted values

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    title = f"Prediction Error" if title is None else title

    r2 = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, label=f"$R^2 = {r2}$")
    ax.set_ylabel('$\hat{y}$')
    ax.set_xlabel('$y$')
    ax.set_title(title)
    min_x, min_y = np.min(y_true), np.min(y_pred)
    max_x, max_y = np.max(y_true), np.max(y_pred)
    ax.plot((min_x, max_x), (min_y, max_y), c='grey', linestyle='--')
    return ax


def plot_feature_importance(importance: DataType,
                            labels: DataType,
                            values: bool = None,
                            title: str = None,
                            ax: Axes = None,
                            top_n: Union[int, float] = None,
                            bottom_n: Union[int, float] = None
                            ) -> Axes:
    """
    Plot a horizontal bar chart of labelled feature importance

    :param importance:
        Importance measure - typically feature importance or coefficient

    :param labels:
        Name of feature

    :param title:
        Plot title

    :param values:
        Add value labels to end of each bar

    :param ax:
        Pass your own ax

    :param top_n:
        If top_n is an integer, return top_n features
        If top_n is a float between 0 and 1, return top_n percent of features

    :param bottom_n:
        If bottom_n is an integer, return bottom_n features
        If bottom_n is a float between 0 and 1, return bottom_n percent of features


    :return:
        matplotlib.Axes
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = f"Feature Importance" if title is None else title

    if top_n:
        if _is_percent(top_n):
            title = f"{title} - Top {top_n:.0%}"
        else:
            title = f"{title} - Top {top_n}"

    if bottom_n:
        if _is_percent(bottom_n):
            title = f"{title} - Bottom {bottom_n:.0%}"
        else:
            title = f"{title} - Bottom {bottom_n}"

    labels, importance = metrics.sorted_feature_importance(labels,
                                                           importance,
                                                           top_n,
                                                           bottom_n
                                                           )

    ax.barh(labels, np.abs(importance))
    ax.set_title(title)
    ax.set_ylabel('Features')
    ax.set_xlabel('Importance')
    if values:
        for i, (x, y) in enumerate(_generate_text_labels(ax, horizontal=True)):
            ax.text(x, y, f"{importance[i]:.2f}", va='center')

    return ax


def plot_lift_curve(y_true, y_proba, title=None, ax=None):
    """
    Plot a lift chart from results. Also calculates lift score based on a .5 threshold

    :param y_true:
        True labels

    :param y_proba:
        Model's predicted probability

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """

    if y_proba.ndim > 1:
        raise VizError("Only works in binary classification. Pass a 1d list")

    if ax is None:
        fig, ax = plt.subplots()

    title = "Lift Curve" if title is None else title

    percents, gains = _cum_gain_curve(y_true, y_proba)
    positives = np.where(y_proba > .5, 1, 0)
    score = metrics.lift_score(y_true, positives)

    ax.plot(percents, gains / percents, label=f'$Lift = {score:.2f}$ ')
    ax.axhline(y=1, color='grey', linestyle='--', label='Baseline')
    ax.set_title(title)
    ax.set_ylabel("Lift")
    ax.set_xlabel("% of Data")
    ax.legend()
    return ax


class VizError(Exception):
    """Base Exception for visualization errors"""
    pass


def _cum_gain_curve(y_true: np.ndarray,
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


def _generate_text_labels(ax, horizontal=False, padding=0.005):
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


def _get_feature_importance(model) -> np.ndarray:
    """
    Helper function for extracting importances.
    Checks for coef_ or feature_importances_ on model

    :param model:
        A sklearn estimator

    :return:
        array of importances
    """
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]

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
