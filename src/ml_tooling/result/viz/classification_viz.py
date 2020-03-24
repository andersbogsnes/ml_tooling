import numpy as np
from matplotlib import pyplot as plt

from ml_tooling.plots import (
    plot_confusion_matrix,
    plot_roc_auc,
    plot_lift_curve,
    plot_pr_curve,
)
from ml_tooling.utils import VizError
from ml_tooling.result.viz import BaseVisualize


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """

    def confusion_matrix(
        self, normalized: bool = True, threshold: float = 0.5, **kwargs
    ) -> plt.Axes:
        """
        Visualize a confusion matrix for a classification estimator
        Any kwargs are passed onto matplotlib

        Parameters
        ----------

        normalized: bool
            Whether or not to normalize annotated class counts
        threshold: float
            Threshold to use for classification - defaults to 0.5

        Returns
        -------
        matplotlib.Axes
        """

        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Confusion Matrix - {self._estimator_name}"
            y_prob = self._estimator.predict_proba(self._data.test_x)
            y_pred = np.where(
                (y_prob > threshold) & (y_prob == y_prob.max(axis=1, keepdims=True)),
                1,
                0,
            )
            y_pred = self._estimator.classes_[np.argmax(y_pred, axis=1)]
            return plot_confusion_matrix(
                self._data.test_y, y_pred, normalized, title, **kwargs
            )

    def roc_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a ROC curve for a classification estimator.
        Estimator must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib

        Returns
        -------
        matplotlib.Axes
        """
        if not hasattr(self._estimator, "predict_proba"):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(self._config.STYLE_SHEET):
            title = f"ROC AUC - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)[:, 1]
            return plot_roc_auc(self._data.test_y, y_proba, title=title, **kwargs)

    def lift_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Lift Curve for a classification estimator
        Estimator must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib

        Returns
        -------
        matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Lift Curve - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)[:, 1]
            return plot_lift_curve(self._data.test_y, y_proba, title=title, **kwargs)

    def pr_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Precision-Recall curve for a classification estimator.
        Estimator must implement a `predict_proba` method.
        Any kwargs are passed onto matplotlib.

        Parameters
        ----------
        kwargs : optional
            Keyword arguments to pass on to matplotlib

        Returns
        -------
        plt.Axes
            Plot of precision-recall curve
        """

        if not hasattr(self._estimator, "predict_proba"):
            raise VizError("Estimator must provide a 'predict_proba' method")

        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Precision-Recall - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)[:, 1]
            return plot_pr_curve(self._data.test_y, y_proba, title=title, **kwargs)
