from functools import total_ordering
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from .plots import (_get_feature_importance,
                    plot_feature_importance,
                    plot_residuals,
                    plot_prediction_error,
                    plot_confusion_matrix,
                    VizError,
                    plot_roc_auc,
                    plot_lift_curve,
                    )
from .utils import get_model_name


@total_ordering
class Result:
    def __init__(self,
                 model,
                 viz=None,
                 model_params=None,
                 score=None,
                 metric=None
                 ):
        self.model = model
        self.model_name = get_model_name(model)
        self.score = score
        self.model_params = model_params
        self.metric = metric
        self.plot = viz

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"<Result {self.model_name}: " \
               f"{self.metric}: {np.round(self.score, 2)} >"


class CVResult(Result):
    """
    Data class for holding results of model testing.
    Also implements comparison operators for finding max mean score
    """

    def __init__(self,
                 model,
                 viz=None,
                 cv=None,
                 model_params=None,
                 cross_val_scores=None,
                 metric=None
                 ):
        self.cv = cv
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = np.mean(cross_val_scores)
        self.cross_val_std = np.std(cross_val_scores)
        super().__init__(model, viz, model_params, self.cross_val_mean, metric)

    def __repr__(self):
        cross_val_type = f"{self.cv}-fold " if isinstance(self.cv, int) else ''
        return f"<Result {self.model_name}: " \
               f"{cross_val_type}Cross-validated {self.metric}: {np.round(self.score, 2)} " \
               f"± {np.round(self.cross_val_std, 2)}>"


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, model, config, data):
        self._model = model
        self._model_name = get_model_name(model)
        self._config = config
        self._data = data
        self._feature_labels = self._get_labels()

    def _get_labels(self):
        """
        If data is a DataFrame, use columns attribute - else use [0...n] np.array
        :return:
            list-like of labels
        """
        if hasattr(self._data.train_x, 'columns'):
            labels = self._data.train_x.columns
        else:
            labels = np.arange(self._data.train_x.shape[1])

        return labels

    def feature_importance(self,
                           values: bool = True,
                           top_n: Union[int, float] = None,
                           bottom_n: Union[int, float] = None,
                           **kwargs) -> plt.Axes:
        """
        Visualizes feature importance of the model. Model must have either feature_importance_
        or coef_ attribute

        :param values:
            Toggles value labels on end of each bar

        :param top_n:
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        :param bottom_n:
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        :return:
            matplotlib.Axes
        """

        title = f"Feature Importance - {self._model_name}"
        importance = _get_feature_importance(self._model)

        with plt.style.context(self._config.STYLE_SHEET):
            return plot_feature_importance(importance,
                                           self._feature_labels,
                                           values=values,
                                           title=title,
                                           top_n=top_n,
                                           bottom_n=bottom_n,
                                           **kwargs)


class RegressionVisualize(BaseVisualize):
    """
    Visualization class for Regression models
    """

    def residuals(self, **kwargs) -> plt.Axes:
        """
        Visualizes residuals of a regression model.
        Any kwargs are passed onto matplotlib

        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Residual Plot - {self._model_name}"
            y_pred = self._model.predict(self._data.test_x)
            return plot_residuals(self._data.test_y, y_pred, title, **kwargs)

    def prediction_error(self, **kwargs) -> plt.Axes:
        """
        Visualizes prediction error of a regression model
        Any kwargs are passed onto matplotlib
        :return:
            matplotlib.Axes
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Prediction Error - {self._model_name}"
            y_pred = self._model.predict(self._data.test_x)
            return plot_prediction_error(self._data.test_y, y_pred, title=title, **kwargs)


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
