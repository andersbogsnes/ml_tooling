from typing import Optional, List

from matplotlib import pyplot as plt

from ml_tooling.plots import (
    plot_confusion_matrix,
    plot_roc_auc,
    plot_lift_curve,
    plot_pr_curve,
)
from ml_tooling.utils import VizError, _classify
from ml_tooling.plots.viz.baseviz import BaseVisualize
from ml_tooling.config import MPL_STYLESHEET


class ClassificationVisualize(BaseVisualize):
    """
    Visualization class for Classification models
    """

    def confusion_matrix(
        self, normalized: bool = True, threshold: Optional[float] = None, **kwargs
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
        plt.Axes
            Returns a Confusion Matrix plot
        """

        with plt.style.context(MPL_STYLESHEET):
            title = f"Confusion Matrix - {self._estimator_name}"
            y_pred = _classify(self._data.test_x, self._estimator, threshold=threshold)
            return plot_confusion_matrix(
                self._data.test_y, y_pred, normalized, title, **kwargs
            )

    def roc_curve(self, labels: List[str] = None, **kwargs) -> plt.Axes:
        """
        Visualize a ROC curve for a classification estimator.
        Estimator must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib

        Parameters
        ----------
        labels: List of str
            Labels to use for the class names if multi-class

        kwargs : optional
            Keyword arguments to pass on to matplotlib

        Returns
        -------
        plt.Axes
            Returns a ROC AUC plot
        """
        if not hasattr(self._estimator, "predict_proba"):
            raise VizError("Model must provide a 'predict_proba' method")

        with plt.style.context(MPL_STYLESHEET):
            title = f"ROC AUC - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)
            return plot_roc_auc(
                self._data.test_y, y_proba, title=title, labels=labels, **kwargs
            )

    def lift_curve(self, **kwargs) -> plt.Axes:
        """
        Visualize a Lift Curve for a classification estimator
        Estimator must implement a `predict_proba` method
        Any kwargs are passed onto matplotlib

        Parameters
        ----------
        kwargs : optional
            Keyword arguments to pass on to matplotlib

        Returns
        -------
        plt.Axes
        """
        with plt.style.context(MPL_STYLESHEET):
            title = f"Lift Curve - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)
            return plot_lift_curve(self._data.test_y, y_proba, title=title, **kwargs)

    def precision_recall_curve(self, labels: List[str] = None, **kwargs) -> plt.Axes:
        """
        Visualize a Precision-Recall curve for a classification estimator.
        Estimator must implement a `predict_proba` method.
        Any kwargs are passed onto matplotlib.

        Parameters
        ----------
        labels: List of str
            Labels to use for the class names if multi-class

        kwargs : optional
            Keyword arguments to pass on to matplotlib

        Returns
        -------
        plt.Axes
            Plot of precision-recall curve
        """

        if not hasattr(self._estimator, "predict_proba"):
            raise VizError("Estimator must provide a 'predict_proba' method")

        with plt.style.context(MPL_STYLESHEET):
            title = f"Precision-Recall - {self._estimator_name}"
            y_proba = self._estimator.predict_proba(self._data.test_x)
            return plot_pr_curve(
                self._data.test_y, y_proba, title=title, labels=labels, **kwargs
            )
