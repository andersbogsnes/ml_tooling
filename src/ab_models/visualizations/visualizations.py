import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, r2_score
import numpy as np
import itertools


class VizError(Exception):
    pass


def plot_roc_auc(y_true, y_proba, title=None):
    """
    Plot ROC AUC curve. Works only with probabilities
    :param y_true:
    :param y_proba:
    :param title:
    :return:
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
    residuals = y_pred - y_true
    title = f'Residual Plot' if title is None else title
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, label=f'$R^2 = {r2:0.3f}$')
    ax.axhline(y=0)
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted Value')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax


def plot_prediction_error(y_true, y_pred, title=None):
    fig, ax = plt.subplots()
    title = f"Prediction Error" if title is None else title
    r2 = r2_score(y_true, y_pred)
    ax.scatter(y_true, y_pred, label=f"$R^2 = {r2}$")
    ax.set_ylabel('$\hat{y}$')
    ax.set_xlabel('$y$')
    ax.set_title(title)
    return ax


class RegressionVisualize:
    def __init__(self, model, config, train_x, train_y, test_x, test_y):
        self._model = model
        self._model_name = model.__class__.__name__
        self._config = config
        self._train_x = train_x
        self._test_x = test_x
        self._train_y = train_y
        self._test_y = test_y

    def residuals(self):
        title = f"Residual Plot - {self._model_name}"
        y_pred = self._model.predict(self._test_x)
        return plot_residuals(self._test_y, y_pred, title)

    def prediction_error(self):
        title = f"Prediction Error - {self._model_name}"
        y_pred = self._model.predict(self._test_x)
        return plot_prediction_error(self._test_y, y_pred, title=title)


class ClassificationVisualize:
    def __init__(self, model, config, train_x, train_y, test_x, test_y):
        self._model = model
        self._model_name = model.__class__.__name__
        self._config = config
        self._train_x = train_x
        self._test_x = test_x
        self._train_y = train_y
        self._test_y = test_y

    def confusion_matrix(self, normalized=True):
        with plt.style.context(self._config["STYLESHEET"]):
            title = f'Confusion Matrix - {self._model_name}'
            y_pred = self._model.predict(self._test_x)
            return plot_confusion_matrix(self._test_y, y_pred, normalized, title)

    def roc_curve(self):
        if not hasattr(self._model, 'predict_proba'):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config["STYLESHEET"]):
            title = f'ROC AUC - {self._model_name}'
            y_proba = self._model.predict_proba(self._test_x)[:, 1]
            return plot_roc_auc(self._test_y, y_proba, title=title)
