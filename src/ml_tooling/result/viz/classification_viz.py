from matplotlib import pyplot as plt

from ml_tooling.plots import plot_confusion_matrix, plot_roc_auc, plot_lift_curve, plot_pr_curve
from ml_tooling.plots.utils import VizError
from ml_tooling.result.viz import BaseVisualize


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """

    def confusion_matrix(self, normalized: bool = True, **kwargs) -> plt.Axes:
        """
        Visualize a confusion matrix for a classification model
        Any kwargs are passed onto matplotlib
        :param normalized:
            Whether or not to normalize annotated class counts

        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f'Confusion Matrix - {self._model_name}'
            y_pred = self._model.predict(self._data.test_x)
            return plot_confusion_matrix(self._data.test_y, y_pred, normalized, title, **kwargs)

    def roc_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a ROC curve for a classification model.
        Model must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        if not hasattr(self._model, 'predict_proba'):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config.STYLE_SHEET):
            title = f'ROC AUC - {self._model_name}'
            y_proba = self._model.predict_proba(self._data.test_x)[:, 1]
            return plot_roc_auc(self._data.test_y, y_proba, title=title, **kwargs)

    def lift_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Lift Curve for a classification model
        Model must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f'Lift Curve - {self._model_name}'
            y_proba = self._model.predict_proba(self._data.test_x)[:, 1]
            return plot_lift_curve(self._data.test_y, y_proba, title=title, **kwargs)

    def pr_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Precision-Recall curve for a classification model.
        Model must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        if not hasattr(self._model, 'predict_proba'):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config.STYLE_SHEET):
            title = f'Precision-Recall - {self._model_name}'
            y_proba = self._model.predict_proba(self._data.test_x)[:, 1]
            return plot_pr_curve(self._data.test_y, y_proba, title=title, **kwargs)
