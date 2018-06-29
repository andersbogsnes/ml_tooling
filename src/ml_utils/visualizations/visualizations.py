"""
Contains all viz functions
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
import numpy as np
import itertools

from . import helpers
from .. import metrics
from .helpers import VizError


def plot_roc_auc(y_true, y_proba, title=None, ax=None):
    """
    Plot ROC AUC curve. Works only with probabilities
    :param y_true: True labels
    :param y_proba: Probability estimate from model
    :param title: Plot title
    :param ax: Pass in your own ax
    :return: matplotlib.Axes
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


def plot_confusion_matrix(y_true, y_pred, normalized=True, title=None, ax=None):
    """
    Plots a confusion matrix of predicted labels vs actual labels
    :param y_true: True labels
    :param y_pred: Predicted labels from model
    :param normalized: Whether to normalize counts in matrix
    :param title: Title for plot
    :param ax: Pass your own ax
    :return: matplotlib.Axes
    """

    title = 'Confusion Matrix' if title is None else title

    if normalized:
        title = f"{title} - Normalized"

    cm = metrics.confusion_matrix(y_true, y_pred, normalized=normalized)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(cm, interpolation='nearest')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)

    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    return ax


def plot_residuals(y_true, y_pred, title=None, ax=None):
    """
    Plots residuals from a regression.
    :param y_true: True value
    :param y_pred: Models predicted value
    :param title: Plot title
    :param ax: Pass your own ax
    :return: matplotlib.Axes
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


def plot_prediction_error(y_true, y_pred, title=None, ax=None):
    """
    Plots prediction error of regression model
    :param y_true: True values
    :param y_pred: Model's predicted values
    :param title: Plot title
    :param ax: Pass your own ax
    :return: matplotlib.Axes
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


def plot_feature_importance(importance, labels, values=None, title=None, ax=None):
    """
    Plot a horizontal bar chart of labelled feature importance
    :param importance: Importance measure - typically feature importance or coefficient
    :param labels: Name of feature
    :param title: Plot title
    :param values: Add value labels to end of each bar
    :param ax: Pass your own ax
    :return: matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    title = f"Feature Importance" if title is None else title

    labels, importance = metrics.sorted_feature_importance(labels, importance)

    ax.barh(labels, np.abs(importance))
    ax.set_title(title)
    ax.set_ylabel('Features')
    ax.set_xlabel('Importance')
    if values:
        for i, (x, y) in enumerate(helpers.generate_text_labels(ax, horizontal=True)):
            ax.text(x, y, f"{importance[i]:.2f}", va='center')

    return ax


def plot_lift_curve(y_true, y_proba, title=None, ax=None):
    """
    Plot a lift chart from results. Also calculates lift score based on a .5 threshold
    :param y_true: True labels
    :param y_proba: Model's predicted probability
    :param title: Plot title
    :param ax: Pass your own ax
    :return: matplotlib.Axes
    """

    if y_proba.ndim > 1:
        raise VizError("Only works in binary classification. Pass a 1d list")

    if ax is None:
        fig, ax = plt.subplots()

    title = "Lift Curve" if title is None else title

    percents, gains = helpers.cum_gain_curve(y_true, y_proba)
    positives = np.where(y_proba > .5, 1, 0)
    score = metrics.lift_score(y_true, positives)

    ax.plot(percents, gains / percents, label=f'$Lift = {score:.2f}$ ')
    ax.axhline(y=1, color='grey', linestyle='--', label='Baseline')
    ax.set_title(title)
    ax.set_ylabel("Lift")
    ax.set_xlabel("% of Data")
    ax.legend()
    return ax


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, model, config, train_x, train_y, test_x, test_y):
        self._model = model
        self._model_name = model.__class__.__name__
        self._config = config
        self._train_x = train_x
        self._test_x = test_x
        self._train_y = train_y
        self._test_y = test_y
        self._feature_labels = self._get_labels()

    def _get_labels(self):
        if hasattr(self._train_x, 'columns'):
            labels = self._train_x.columns
        else:
            labels = np.arange(self._train_x.shape[1])

        return labels

    def feature_importance(self, values=True, **kwargs):
        """
        Visualizes feature importance of the model. Model must have either feature_importance_
        or coef_ attribute
        :param values: Toggles value labels on end of each bar
        :return: matplotlib.Axes
        """

        title = f"Feature Importance - {self._model_name}"
        importance = helpers.get_feature_importance(self._model)

        with plt.style.context(self._config['STYLE_SHEET']):
            return plot_feature_importance(importance,
                                           self._feature_labels,
                                           values=values,
                                           title=title,
                                           **kwargs)


class RegressionVisualize(BaseVisualize):
    """
    Visualization class for Regression models
    """

    def residuals(self, **kwargs):
        """
        Visualizes residuals of a regression model
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f"Residual Plot - {self._model_name}"
            y_pred = self._model.predict(self._test_x)
            return plot_residuals(self._test_y, y_pred, title, **kwargs)

    def prediction_error(self, **kwargs):
        """
        Visualizes prediction error of a regression model
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f"Prediction Error - {self._model_name}"
            y_pred = self._model.predict(self._test_x)
            return plot_prediction_error(self._test_y, y_pred, title=title, **kwargs)


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """

    def confusion_matrix(self, normalized=True, **kwargs):
        """
        Visualize a confusion matrix for a classification model
        :param normalized: Whether or not to normalize annotated class counts
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f'Confusion Matrix - {self._model_name}'
            y_pred = self._model.predict(self._test_x)
            return plot_confusion_matrix(self._test_y, y_pred, normalized, title, **kwargs)

    def roc_curve(self, **kwargs):
        """
        Visualize a ROC curve for a classification model.
        Model must implement a `predict_proba` method
        :return: matplotlib.Axes
        """
        if not hasattr(self._model, 'predict_proba'):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config['STYLE_SHEET']):
            title = f'ROC AUC - {self._model_name}'
            y_proba = self._model.predict_proba(self._test_x)[:, 1]
            return plot_roc_auc(self._test_y, y_proba, title=title, **kwargs)

    def lift_curve(self, **kwargs):
        """
        Visualize a Lift Curve for a classification model
        Model must implement a `predict_proba` method
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config["STYLE_SHEET"]):
            title = f'Lift Curve - {self._model_name}'
            y_proba = self._model.predict_proba(self._test_x)[:, 1]
            return plot_lift_curve(self._test_y, y_proba, title=title, **kwargs)
