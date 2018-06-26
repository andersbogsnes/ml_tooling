import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, r2_score
import numpy as np
import itertools


class VizError(Exception):
    """Base Exception for visualization errors"""
    pass


def plot_roc_auc(y_true, y_proba, title=None):
    """
    Plot ROC AUC curve. Works only with probabilities
    :param y_true: True labels
    :param y_proba: Probability estimate from model
    :param title: Plot title
    :return: matplotlib.Axes
    """
    title = 'ROC AUC curve' if title is None else title

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    score = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Score: {score}")
    ax.plot([0, 1], '--')
    ax.set_title(title)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc='best')
    plt.tight_layout()
    return ax


def plot_confusion_matrix(y_true, y_pred, normalized=True, title=None):
    """
    Plots a confusion matrix of predicted labels vs actual labels
    :param y_true: True labels
    :param y_pred: Predicted labels from model
    :param normalized: Whether to normalize counts in matrix
    :param title: Title for plot
    :return: matplotlib.Axes
    """
    cm = confusion_matrix(y_true, y_pred)
    title = 'Confusion Matrix' if title is None else title

    if normalized is True:
        title = f"{title} - Normalized"
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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


def plot_residuals(y_true, y_pred, title=None):
    """
    Plots residuals from a regression.
    :param y_true: True value
    :param y_pred: Models predicted value
    :param title: Plot title
    :return: matplotlib.Axes
    """
    residuals = y_pred - y_true
    title = f'Residual Plot' if title is None else title
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, label=f'$R^2 = {r2:0.3f}$')
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted Value')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax


def plot_prediction_error(y_true, y_pred, title=None):
    """
    Plots prediction error of regression model
    :param y_true: True values
    :param y_pred: Model's predicted values
    :param title: Plot title
    :return: matplotlib.Axes
    """
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


def plot_feature_importance(importance, labels, values=None, title=None):
    """
    Plot a horizontal bar chart of labelled feature importance
    :param importance: Importance measure - typically feature importance or coefficient
    :param labels: Name of feature
    :param title: Plot title
    :param values: Add value labels to end of each bar
    :return: matplotlib.Axes
    """
    fig, ax = plt.subplots()
    title = f"Feature Importance" if title is None else title
    idx = np.argsort(np.abs(importance))
    y_values, x_values = labels[idx], importance[idx]
    ax.barh(y_values, np.abs(x_values))
    ax.set_title(title)
    ax.set_ylabel('Features')
    ax.set_xlabel('Importance')
    if values:
        for (i, patch) in enumerate(ax.patches):
            y = patch.get_y()
            width = patch.get_width()
            height = y + (patch.get_height() / 2)
            padding = ax.get_xbound()[1] * 0.005
            ax.text(width + padding, height, f"{x_values[i]:.2f}", va='center')

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

    def feature_importance(self, values=True):
        """
        Visualizes feature importance of the model. Model must have either feature_importance_
        or coef_ attribute
        :param values: Toggles value labels on end of each bar
        :return: matplotlib.Axes
        """
        labels = self._train_x.columns

        if hasattr(self._model, 'feature_importances_'):
            importance = self._model.feature_importances_

        elif hasattr(self._model, 'coef_'):
            importance = self._model.coef_
            if importance.ndim > 1:
                importance = importance[0]

        else:
            raise VizError(f"{self._model_name} does not have either coef_ or feature_importances_")

        if len(labels) != len(importance):
            raise VizError(f"Must have equal number of labels as features: "
                           f"You have {len(labels)} labels and {len(importance)} features")

        title = f"Feature Importance - {self._model_name}"
        with plt.style.context(self._config['STYLE_SHEET']):
            return plot_feature_importance(importance, labels, values=values, title=title)


class RegressionVisualize(BaseVisualize):
    """
    Visualization class for Regression models
    """
    def residuals(self):
        """
        Visualizes residuals of a regression model
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f"Residual Plot - {self._model_name}"
            y_pred = self._model.predict(self._test_x)
            return plot_residuals(self._test_y, y_pred, title)

    def prediction_error(self):
        """
        Visualizes prediction error of a regression model
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f"Prediction Error - {self._model_name}"
            y_pred = self._model.predict(self._test_x)
            return plot_prediction_error(self._test_y, y_pred, title=title)


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """
    def confusion_matrix(self, normalized=True):
        """
        Visualize a confusion matrix for a classification model
        :param normalized: Whether or not to normalize annotated class counts
        :return: matplotlib.Axes
        """
        with plt.style.context(self._config['STYLE_SHEET']):
            title = f'Confusion Matrix - {self._model_name}'
            y_pred = self._model.predict(self._test_x)
            return plot_confusion_matrix(self._test_y, y_pred, normalized, title)

    def roc_curve(self):
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
            return plot_roc_auc(self._test_y, y_proba, title=title)
